import os
import sys
import re
import requests
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

############################################################
# GLOBAL CONFIG
############################################################

BASE_MODEL = "microsoft/Phi-4-mini-instruct"
MAX_SEQ_LEN = 512

LR = 2e-4
EPOCHS = 2
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0

OUTPUT_DIR = "./llm_tuned_any"

USE_GRADIENT_CHECKPOINTING = True

############################################################
# BASIC SAFETY FILTER
############################################################
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

############################################################
# HUGGINGFACE DOMAINS
############################################################
DATASET_REGISTRY = {
    "general": [
        {"hf_repo": "yizhongw/self_instruct", "split": "train"},
        {"hf_repo": "teknium/unnatural_instructions", "split": "train"},
    ],
    "safety": [
        {"hf_repo": "PKU-Alignment/BeaverTails", "split": "train"},
    ],
    "reasoning": [
        # this might 404 / be private / etc., and we handle it gracefully
        {"hf_repo": "ontocord/mixturevitae_reasoning", "split": "train"},
    ],
    # frc_code handled separately
}

############################################################
# FRC REPO SCRAPE CONFIG
############################################################
# We automatically scrape public FRC code from known "please reuse this"
# sources like HuskieRobotics frc-software-2025 (swerve starter for MK4/MK4i,
# Kraken/Falcon, CANCoder, Pigeon2), Stryke Force 'thirdcoast' (swerve drive +
# telemetry library), and WPILib official examples (RobotContainer, Constants,
# odometry setups). These projects are publicly shared specifically to teach
# and bootstrap other FRC teams. :contentReference[oaicite:2]{index=2}
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

############################################################
# SMALL HELPERS
############################################################
def clean_code_for_training(src: str) -> str:
    # drop giant /* ... */ license headers at file top to save tokens
    cleaned = re.sub(r"^/\*.*?\*/", "", src, flags=re.DOTALL)
    cleaned = cleaned.strip()
    return cleaned

def make_frc_example(filename: str, code_text: str) -> str:
    # We wrap the code as instruction/response pairs, so the model learns
    # "when asked for RobotContainer / Constants / swerve subsystem,
    # output Java code in FRC style."
    #
    # WPILib and top teams organize subsystems, RobotContainer button bindings,
    # drivetrain odometry (SwerveDriveKinematics + gyro), and Constants for CAN IDs,
    # PID gains, geometry, etc. That's exactly the style we want the model to mimic,
    # which is why we present it as "Write high quality FRC robot code for <file> ...".
    # :contentReference[oaicite:3]{index=3}
    inst = (
        f"Write high quality FRC robot code for {filename}. "
        "Follow WPILib Command-based structure with RobotContainer, subsystem classes, "
        "SwerveDrive odometry using SwerveDriveKinematics and a gyro, and clean Constants."
    )

    return (
        "### Instruction:\n"
        + inst
        + "\n\n### Input:\n(none)\n\n### Response:\n"
        + code_text.strip()
        + "\n"
    )

def fetch_repo_tree(owner: str, repo: str, branch: str):
    """Use GitHub public tree API to list repo files."""
    url = GITHUB_API_TREE.format(owner=owner, repo=repo, branch=branch)
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        print(f"[WARN] tree {owner}/{repo}@{branch} -> HTTP {r.status_code}")
        return []
    data = r.json()
    if "tree" not in data:
        return []
    return data["tree"]

def download_raw_file(owner: str, repo: str, branch: str, path: str) -> str:
    url = RAW_FILE_URL.format(owner=owner, repo=repo, branch=branch, path=path)
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        print(f"[WARN] raw {url} -> HTTP {r.status_code}")
        return ""
    return r.text

def build_frc_corpus() -> list[str]:
    """
    For each FRC repo:
    - list repo tree
    - keep .java files from robot-y dirs
    - download, clean, chunk
    - wrap into instruction/response format
    """
    examples = []
    for (owner, repo, branch, include_substrings) in FRC_REPOS:
        tree = fetch_repo_tree(owner, repo, branch)
        for node in tree:
            if node.get("type") != "blob":
                continue
            path = node.get("path", "")
            if not path.endswith(".java"):
                continue
            if not any(sub in path for sub in include_substrings):
                continue

            raw_code = download_raw_file(owner, repo, branch, path)
            if not raw_code.strip():
                continue
            if not is_safe(raw_code):
                continue

            cleaned = clean_code_for_training(raw_code)

            # break huge files so they fit in MAX_SEQ_LEN after tokenization
            chunk_size_chars = 3000
            for i in range(0, len(cleaned), chunk_size_chars):
                chunk = cleaned[i : i + chunk_size_chars].strip()
                if not chunk:
                    continue
                ex = make_frc_example(os.path.basename(path), chunk)
                examples.append(ex)

    print(
        f"[frc-data] collected {len(examples)} code chunks from public FRC repos"
    )
    return examples

############################################################
# NORMALIZE HF DATA (general / safety / reasoning)
############################################################
def normalize_sample(sample: dict) -> str:
    instr_fields = ["instruction", "query", "question", "prompt", "task"]
    input_fields = ["input", "context", "history"]
    out_fields = ["output", "response", "answer", "completion"]

    instr_txt = ""
    input_txt = ""
    out_txt = ""

    for k in instr_fields:
        if k in sample and isinstance(sample[k], str):
            instr_txt = sample[k].strip()
            break
    for k in input_fields:
        if k in sample and isinstance(sample[k], str) and sample[k].strip():
            input_txt = sample[k].strip()
            break
    for k in out_fields:
        if k in sample and isinstance(sample[k], str):
            out_txt = sample[k].strip()
            break

    # fallback for datasets that just have "text"
    if not instr_txt and "text" in sample and isinstance(sample["text"], str):
        instr_txt = sample["text"].strip()
        out_txt = ""

    full_prompt = "### Instruction:\n" + instr_txt
    if input_txt:
        full_prompt += "\n\n### Input:\n" + input_txt
    full_prompt += "\n\n### Response:\n" + out_txt + "\n"
    return full_prompt

############################################################
# DATASET CLASSES
############################################################
class HFDomainDataset(Dataset):
    """
    For 'general', 'safety', 'reasoning' from Hugging Face Hub.
    We skip samples that fail safety filter.
    """
    def __init__(self, domain: str, tokenizer, max_len: int):
        if domain not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown domain '{domain}'. "
                f"Available: {list(DATASET_REGISTRY.keys())} plus 'frc_code'"
            )

        self.examples = []
        for cfg in DATASET_REGISTRY[domain]:
            repo = cfg["hf_repo"]
            split = cfg.get("split", "train")
            try:
                ds = load_dataset(repo, split=split)
            except Exception as e:
                print(f"[WARN] couldn't load {repo}: {e}")
                continue

            for row in ds:
                text = normalize_sample(dict(row))
                if is_safe(text):
                    self.examples.append(text)

        self.tok = tokenizer
        self.max_len = max_len
        print(f"[data:{domain}] {len(self.examples)} safe samples")

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
        attn = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }

class FRCDataset(Dataset):
    """
    For 'frc_code' (GitHub scrape).
    """
    def __init__(self, tokenizer, max_len: int):
        self.examples = build_frc_corpus()
        self.tok = tokenizer
        self.max_len = max_len

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
        attn = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }

class Collator:
    def __call__(self, batch):
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}

############################################################
# LORA / QLORA MODEL LOADER
############################################################
def mark_lora_trainable(peft_model: torch.nn.Module):
    """
    Safety belt: make sure LoRA adapter params actually require grad.
    People reported that after mixing PEFT + gradient checkpointing,
    they hit RuntimeError: 'element 0 ... does not require grad'
    because no params were trainable. Explicitly flipping
    requires_grad on LoRA params fixes it. :contentReference[oaicite:4]{index=4}
    """
    for name, param in peft_model.named_parameters():
        if "lora_" in name or "loraA" in name or "loraB" in name or "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        elif getattr(param, "requires_grad", False):
            # leave True as-is
            param.requires_grad = True
    return peft_model

def build_lora_model(base_model_name):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 1. load base in 4-bit
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_cfg,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # 2. prepare for k-bit (QLoRA training hook)
    # Hugging Face PEFT docs + forum answers say: you MUST run this before LoRA
    # when training a quantized model, or gradients won't flow and backward()
    # will crash with that grad_fn error. :contentReference[oaicite:5]{index=5}
    base = prepare_model_for_kbit_training(base)

    # 3. configure LoRA adapters
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

    # 4. inject LoRA
    lora_model = get_peft_model(base, lora_cfg)

    # 5. mark LoRA params trainable explicitly
    lora_model = mark_lora_trainable(lora_model)

    # 6. AFTER LoRA attach, enable gradient checkpointing if requested.
    # People see fewer "no grad_fn" issues when checkpointing is enabled
    # after the PEFT wrap, not before. :contentReference[oaicite:6]{index=6}
    if USE_GRADIENT_CHECKPOINTING and hasattr(lora_model, "gradient_checkpointing_enable"):
        lora_model.gradient_checkpointing_enable()
        # transformers will auto-set use_cache=False when checkpointing is on

    lora_model.train()
    lora_model.print_trainable_parameters()
    return lora_model, tokenizer

############################################################
# TRAIN LOOP
############################################################
def train_model(
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(
            "No trainable params found after LoRA prep. "
            "Check requires_grad on adapter weights."
        )

    opt = AdamW(trainable_params, lr=lr)

    steps_per_epoch = max(1, len(train_loader) // grad_accum_steps + 1)
    total_steps = max(1, epochs * steps_per_epoch)

    sched = get_cosine_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # use new torch.amp APIs so we don't get deprecation warnings
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    os.makedirs(save_root, exist_ok=True)

    global_step = 0
    running_loss = 0.0

    for ep in range(epochs):
        for step_idx, batch in enumerate(train_loader):
            for k in batch:
                batch[k] = batch[k].to(device)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                # after prepare_model_for_kbit_training + LoRA, out.loss
                # is attached to trainable LoRA weights, so .backward()
                # will produce grads instead of crashing. :contentReference[oaicite:7]{index=7}
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
                    print(
                        f"[train] step {global_step}/{total_steps} loss={avg:.4f}"
                    )
                    running_loss = 0.0

        # save adapter snapshot at end of each epoch
        save_dir = os.path.join(save_root, f"epoch_{ep+1}")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"[train] saved adapters -> {save_dir}")

    print("[train] done")

############################################################
# MAIN
############################################################
if __name__ == "__main__":
    print("Available domains: general, safety, reasoning, frc_code")
    domain = input(
        "Enter a domain to train on (example: general / safety / reasoning / frc_code): "
    ).strip().lower()

    model, tok = build_lora_model(BASE_MODEL)

    if domain == "frc_code":
        ds = FRCDataset(tok, MAX_SEQ_LEN)
    else:
        ds = HFDomainDataset(domain, tok, MAX_SEQ_LEN)

    if len(ds) == 0:
        print(f"[FATAL] 0 usable samples for domain '{domain}'. Exiting.")
        sys.exit(1)

    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        collate_fn=Collator(),
    )

    train_model(
        model,
        tok,
        dl,
        epochs=EPOCHS,
        lr=LR,
        warmup_steps=WARMUP_STEPS,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        save_root=OUTPUT_DIR,
    )
