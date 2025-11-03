import os
import sys
import re
import json
import requests
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)

###############################################################################
# CONFIG
###############################################################################

BASE_MODEL = "microsoft/Phi-4-mini-instruct"

MAX_SEQ_LEN = 512
LR = 2e-4
EPOCHS = 25               # <-- per your request
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0

OUTPUT_DIR = "./frc_continued_checkpoints"

USE_GRADIENT_CHECKPOINTING = True

# GitHub scrape targets:
# These are public FRC codebases / WPILib examples that teams publish specifically
# for reuse (swerve drivebases, RobotContainer wiring, Constants, etc.). Teams like
# HuskieRobotics (3061) and Stryke Force (2767), and WPILib itself, openly publish
# swerve drivetrains, odometry, gyro integration, CAN bus configs, etc., for other
# teams to learn from. :contentReference[oaicite:5]{index=5}
FRC_REPOS = [
    (
        "HuskieRobotics",
        "frc-software-2025",
        "main",
        [
            "src/main/java/frc/robot",
            "swerve",
        ],
    ),
    (
        "strykeforce",
        "thirdcoast",
        "main",
        [
            "swerve",
            "src/main/java",
        ],
    ),
    (
        "wpilibsuite",
        "allwpilib",
        "main",
        [
            "wpilibjExamples",
            "examples/swerve",
            "examples/swervebot",
            "RobotContainer",
            "Constants",
        ],
    ),
]

GITHUB_API_TREE = (
    "https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
)
RAW_FILE_URL = (
    "https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
)

###############################################################################
# BASIC SAFETY FILTER
###############################################################################
BLOCK_PATTERNS = [
    r"suicide",
    r"kill yourself",
    r"how to make a bomb",
    r"how to make explosives",
    r"pipe bomb",
    r"harm.*yourself",
    r"self-harm",
    r"commit suicide",
    r"molotov cocktail",
]
block_regex = re.compile("|".join(BLOCK_PATTERNS), re.IGNORECASE)

def is_safe(text: str) -> bool:
    return not bool(block_regex.search(text))

###############################################################################
# SCRAPE HELPERS
###############################################################################

def fetch_repo_tree(owner: str, repo: str, branch: str):
    """Return list of {path, type} using GitHub's git/trees?recursive=1 API.
    This is a documented pattern: GET /repos/:owner/:repo/git/trees/:branch?recursive=1. :contentReference[oaicite:6]{index=6}
    """
    url = GITHUB_API_TREE.format(owner=owner, repo=repo, branch=branch)
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        print(f"[WARN] tree {owner}/{repo}@{branch} -> HTTP {r.status_code}")
        return []
    data = r.json()
    if "tree" not in data:
        return []
    # GitHub can truncate large trees; in that case `truncated` may be true and
    # we'd only get partial list. For our use, partial is still fine. :contentReference[oaicite:7]{index=7}
    return data["tree"]

def download_raw_file(owner: str, repo: str, branch: str, path: str) -> str:
    """Grab raw text for a file via raw.githubusercontent.com."""
    url = RAW_FILE_URL.format(owner=owner, repo=repo, branch=branch, path=path)
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        print(f"[WARN] raw {url} -> HTTP {r.status_code}")
        return ""
    return r.text

def clean_code_for_training(src: str) -> str:
    # drop giant /* ... */ license headers at file top to save tokens
    cleaned = re.sub(r"^/\*.*?\*/", "", src, flags=re.DOTALL)
    cleaned = cleaned.strip()
    return cleaned

def wrap_as_supervised_example(filename: str, code_text: str) -> str:
    # We frame each chunk like instruction-tuning data:
    # "### Instruction: ...  ### Response: <actual FRC code chunk>"
    #
    # This teaches the model:
    # - when asked for "RobotContainer.java" or "Constants.java", respond with valid,
    #   WPILib-style Java code (subsystems, bindings, swerve kinematics, gyro usage,
    #   CAN IDs, etc.). WPILib docs + public FRC codebases promote this structure. :contentReference[oaicite:8]{index=8}
    inst = (
        f"Write high quality FRC robot code for {filename}. "
        "Follow WPILib Command-based structure with RobotContainer, subsystem classes, "
        "SwerveDrive odometry using SwerveDriveKinematics and a gyro, "
        "and clean Constants for CAN IDs / PID gains / geometry."
    )

    return (
        "### Instruction:\n"
        + inst
        + "\n\n### Input:\n(none)\n\n### Response:\n"
        + code_text.strip()
        + "\n"
    )

def build_frc_corpus() -> list[str]:
    """Collect many Java chunks from public FRC repos."""
    examples = []
    for (owner, repo, branch, include_substrings) in FRC_REPOS:
        tree = fetch_repo_tree(owner, repo, branch)
        for node in tree:
            if node.get("type") != "blob":
                continue
            path = node.get("path", "")
            if not path.endswith(".java"):
                continue

            # only grab robot-ish files
            if not any(sub in path for sub in include_substrings):
                continue

            raw_code = download_raw_file(owner, repo, branch, path)
            if not raw_code.strip():
                continue
            if not is_safe(raw_code):
                continue

            cleaned = clean_code_for_training(raw_code)

            # chunk to avoid blowing past token window
            chunk_size_chars = 3000
            for i in range(0, len(cleaned), chunk_size_chars):
                chunk = cleaned[i : i + chunk_size_chars].strip()
                if not chunk:
                    continue
                ex = wrap_as_supervised_example(os.path.basename(path), chunk)
                examples.append(ex)

    print(f"[frc-data] collected {len(examples)} code chunks from public FRC repos")
    return examples

###############################################################################
# DATASET
###############################################################################

class FRCDataset(Dataset):
    """Dataset of new FRC samples scraped RIGHT NOW."""
    def __init__(self, tokenizer, max_len: int):
        self.examples = build_frc_corpus()
        self.tok = tokenizer
        self.max_len = max_len

        if len(self.examples) == 0:
            print("[FATAL] scraped 0 usable FRC samples")
            # we won't sys.exit() here, we let caller decide

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        enc = self.tok(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }

class Collator:
    def __call__(self, batch):
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}

###############################################################################
# MODEL LOADING / RESUME LOGIC
###############################################################################

def mark_lora_trainable(peft_model: torch.nn.Module):
    """
    Force LoRA adapter params to require grad. This helps ensure
    that backward() will compute grads for the adapter layers only,
    which is the point of PEFT/LoRA. PEFT docs and community guides
    note that only a tiny fraction of params are updated vs the
    frozen base model. :contentReference[oaicite:9]{index=9}
    """
    for name, param in peft_model.named_parameters():
        # Heuristic: LoRA adapter params contain 'lora' or are already unfrozen.
        if "lora" in name.lower():
            param.requires_grad = True
        # If it's already True, leave it True.
    return peft_model

def load_base_model_4bit():
    """Load Phi-4-mini-instruct in 4-bit and prep it for QLoRA training.
    QLoRA uses quantization + LoRA to let you fine-tune larger LLMs on a
    single GPU by freezing the base weights and only training low-rank
    adapters. Hugging Face's PEFT docs show calling
    prepare_model_for_kbit_training() before adding LoRA. :contentReference[oaicite:10]{index=10}
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_cfg,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Prep base for k-bit finetune:
    base_model = prepare_model_for_kbit_training(base_model)

    return base_model, tokenizer

def attach_or_resume_lora(base_model, adapter_path: str | None):
    """
    If adapter_path is provided (a folder with an existing LoRA adapter),
    load that adapter on top of the base. This is how you "continue"
    training instead of starting from scratch. Hugging Face PEFT integrations
    let you reload LoRA weights on the base model and keep fine-tuning. :contentReference[oaicite:11]{index=11}

    If adapter_path is None or doesn't exist, we make a fresh LoRA.
    """
    if adapter_path and os.path.isdir(adapter_path):
        print(f"[resume] Loading existing adapter from: {adapter_path}")
        # Wrap base with a dummy LoRA first, then merge in adapter weights using PeftModel.from_pretrained.
        # Simpler path: directly call PeftModel.from_pretrained to create a PEFT model that
        # has the LoRA weights loaded.
        lora_model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=True,  # mark adapter trainable if supported
        )

    else:
        print("[resume] No valid adapter path given, creating NEW adapter from scratch.")
        lora_cfg = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        lora_model = get_peft_model(base_model, lora_cfg)

    # Make sure LoRA params can get gradients
    lora_model = mark_lora_trainable(lora_model)

    # Enable gradient checkpointing AFTER LoRA is attached.
    if USE_GRADIENT_CHECKPOINTING and hasattr(lora_model, "gradient_checkpointing_enable"):
        lora_model.gradient_checkpointing_enable()
        # transformers will force use_cache=False automatically when checkpointing is on

    lora_model.train()
    lora_model.print_trainable_parameters()
    return lora_model

###############################################################################
# TRAIN LOOP
###############################################################################

def train_loop(
    model,
    tokenizer,
    train_loader,
    epochs,
    lr,
    warmup_steps,
    grad_accum_steps,
    max_grad_norm,
    save_root,
):
    """
    Classic single-GPU PEFT/QLoRA loop:
    - only LoRA params get gradients
    - mixed precision via torch.amp.autocast + GradScaler
    - gradient accumulation so effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
    - cosine LR schedule w/ warmup, which is standard in modern LLM finetunes
      to avoid blowing out weights early. :contentReference[oaicite:12]{index=12}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable params found in LoRA model (nothing requires_grad=True).")

    opt = AdamW(trainable_params, lr=lr)

    steps_per_epoch = max(1, len(train_loader) // grad_accum_steps + 1)
    total_steps = max(1, epochs * steps_per_epoch)

    sched = get_cosine_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    os.makedirs(save_root, exist_ok=True)

    global_step = 0
    running_loss = 0.0

    for ep in range(epochs):
        for step_idx, batch in enumerate(train_loader):
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = out.loss / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum_steps

            if (step_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

                global_step += 1
                if global_step % 10 == 0:
                    avg = running_loss / 10.0
                    print(f"[train] step {global_step}/{total_steps} loss={avg:.4f}")
                    running_loss = 0.0

        # save adapter snapshot each epoch
        save_dir = os.path.join(save_root, f"epoch_{ep+1}")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"[train] saved adapters -> {save_dir}")

    print("[train] done ✅")

###############################################################################
# MAIN
###############################################################################

if __name__ == "__main__":
    # 1. Ask user which LoRA adapter to continue from.
    print("Path example: C:\\Users\\User\\Desktop\\Autocomplete\\llm_tuned_any\\epoch_2")
    adapter_path = input("Enter path to existing LoRA adapter folder (or leave blank to start new): ").strip()

    # 2. Load base Phi-4-mini-instruct in 4-bit and prep for QLoRA
    base_model, tokenizer = load_base_model_4bit()

    # 3. Attach previous adapter (continue training) OR create new LoRA
    model = attach_or_resume_lora(base_model, adapter_path)

    # 4. Scrape fresh FRC data right now
    dataset = FRCDataset(tokenizer, MAX_SEQ_LEN)
    if len(dataset) == 0:
        print("[FATAL] scraped 0 samples. Can't train.")
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        collate_fn=Collator(),
    )

    # 5. Train for 100 epochs on ONLY this new scrape
    train_loop(
        model=model,
        tokenizer=tokenizer,
        train_loader=loader,
        epochs=EPOCHS,
        lr=LR,
        warmup_steps=WARMUP_STEPS,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        save_root=OUTPUT_DIR,
    )
