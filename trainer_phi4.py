import os
import math
import json
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.amp import autocast, GradScaler  # new AMP API (PyTorch now prefers torch.amp.autocast("cuda")). :contentReference[oaicite:3]{index=3}

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

#########################################
# 0. CONFIG
#########################################

# Phi-4-mini-instruct is lighter and explicitly intended for instruction/chat
# and PEFT/LoRA style fine-tuning on a single GPU. Microsoft and the community
# show this as the go-to target instead of full Phi-4, because you can freeze
# base weights and just train ~1% adapter params (QLoRA style). This makes it
# fit on consumer GPUs. :contentReference[oaicite:4]{index=4}
MODEL_NAME = "microsoft/Phi-4-mini-instruct"

# Your dataset path. Each line in this file should be valid JSON like:
# {"text": "some training string ..."}
JSONL_PATH = r"C:\Users\User\Desktop\Autocomplete\data.jsonl"

OUTPUT_DIR = "./phi4_lora_checkpoints"

MAX_SEQ_LEN = 512           # sequence length cap (tokens)
LR = 2e-4                   # LoRA adapter LR can be higher than full-model LR
NUM_EPOCHS = 3
BATCH_SIZE = 2              # microbatch per step
GRAD_ACCUM_STEPS = 8        # effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0

# IMPORTANT: gradient checkpointing + LoRA on quantized models can break the
# backward graph and cause "element 0 of tensors does not require grad".
# This is a known issue in PEFT discussions for LoRA fine-tuning where
# checkpointing sometimes detaches grads. Disabling it fixes the crash. :contentReference[oaicite:5]{index=5}
USE_GRADIENT_CHECKPOINTING = False


#########################################
# 1. DATA: JSONL loader for causal LM
#########################################

class Phi4JsonlDataset(Dataset):
    """
    Expects .jsonl like:
        {"text": "blah blah ..."}
        {"text": "another sample..."}
    one JSON object per line, no commas, no outer [].

    We tokenize and train causal LM on that text.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_len: int):
        self.samples: List[str] = []
        self.tok = tokenizer
        self.max_len = max_len

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    # empty line; super common cause of JSONDecodeError. :contentReference[oaicite:6]{index=6}
                    continue

                # strip UTF-8 BOM if present
                if line.startswith("\ufeff"):
                    line = line.replace("\ufeff", "", 1)

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] bad JSON on line {lineno}: {e}")
                    print(f"[WARN] content starts with: {repr(line[:200])}")
                    continue

                text = obj.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    print(f"[WARN] line {lineno} missing/empty 'text', skipping")
                    continue

                self.samples.append(text.strip())

        if len(self.samples) == 0:
            print("[FATAL] No usable training samples found in your JSONL.")
            print("        Make sure each line is something like {\"text\": \"...\"}")
        else:
            print(f"[data] loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.samples[idx]

        # classic causal LM setup:
        # input_ids -> model; labels = same tokens shifted internally
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
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


#########################################
# 2. Collator (everything already padded)
#########################################

@dataclass
class Collator:
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        keys = batch[0].keys()
        out = {}
        for k in keys:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        return out


#########################################
# 3. Load Phi-4-mini in 4-bit + attach LoRA
#########################################

def load_phi4_lora(model_name: str):
    """
    - Load Phi-4-mini-instruct in 4-bit NF4 via bitsandbytes (QLoRA style).
      This is how people fine-tune large LLMs on consumer GPUs with low VRAM.
      QLoRA does: quantize base weights to 4-bit + train only LoRA adapters. :contentReference[oaicite:7]{index=7}
    - Inject LoRA adapters into attention/MLP proj layers.
      Training ~1% of params is normal for LoRA and is what lets this run on 1 GPU. :contentReference[oaicite:8]{index=8}
    """

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization config
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",      # NF4 is standard for QLoRA high-quality 4-bit
        bnb_4bit_compute_dtype=torch.float16,
    )

    # base model (frozen weights loaded in 4-bit on GPU)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        torch_dtype=torch.float16,
        device_map="auto",   # map layers to your GPU automatically
    )

    # DO NOT enable gradient checkpointing here. Leaving it on is known to
    # trigger the grad break you just hit (loss ends up not requiring grad).
    # if USE_GRADIENT_CHECKPOINTING and hasattr(base_model, "gradient_checkpointing_enable"):
    #     base_model.gradient_checkpointing_enable()

    # LoRA config
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
    lora_model.print_trainable_parameters()
    # You'll see something like:
    # trainable params: ~35M || all params: ~3.8B || trainable%: ~0.9
    # <1% trainable is absolutely normal for LoRA and is how LoRA keeps VRAM low. :contentReference[oaicite:9]{index=9}

    return lora_model, tokenizer


#########################################
# 4. Trainer (single GPU, LoRA adapters only)
#########################################

class Phi4Trainer:
    """
    - single GPU
    - mixed precision (autocast + GradScaler)
    - gradient accumulation
    - cosine LR schedule w/ warmup
    - gradient clipping
    - saves ONLY LoRA adapter weights (not the giant base model)
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_loader,
        num_epochs,
        lr,
        warmup_steps,
        grad_accum_steps,
        max_grad_norm,
        output_dir,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = output_dir

        # Optimizer over ONLY the LoRA params (the base model stays frozen)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=lr)

        total_update_steps = (
            num_epochs * math.ceil(len(train_loader) / grad_accum_steps)
        )
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_update_steps,
        )

        # New AMP API: torch.amp.GradScaler("cuda"), recommended in current PyTorch
        # instead of torch.cuda.amp.GradScaler, which is deprecated. :contentReference[oaicite:10]{index=10}
        self.scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

        os.makedirs(output_dir, exist_ok=True)
        self.model.train()

        self.global_step = 0
        self.running_loss = 0.0
        self.total_update_steps = total_update_steps

    def _save_checkpoint(self):
        # Save ONLY the LoRA adapter weights + tokenizer.
        # This is the normal PEFT workflow: ship adapters, not full model. :contentReference[oaicite:11]{index=11}
        save_path = os.path.join(self.output_dir, f"step_{self.global_step}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"[checkpoint] saved adapters at {save_path}")

    def train(self):
        for epoch in range(self.num_epochs):
            for batch_idx, batch in enumerate(self.train_loader):
                # move batch to GPU
                for k in batch:
                    batch[k] = batch[k].to(self.device, non_blocking=True)

                # forward pass with mixed precision
                # torch.amp.autocast("cuda") is the new recommended style,
                # replacing torch.cuda.amp.autocast (deprecated). :contentReference[oaicite:12]{index=12}
                with autocast("cuda", enabled=torch.cuda.is_available()):
                    out = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = out.loss / self.grad_accum_steps

                # safety check: if loss has no grad_fn here, bail early with a clear message
                if not loss.requires_grad:
                    raise RuntimeError(
                        "loss has no grad_fn (likely gradient checkpointing / graph detach issue). "
                        "Make sure USE_GRADIENT_CHECKPOINTING is False."
                    )

                # backward through scaled loss
                self.scaler.scale(loss).backward()
                self.running_loss += loss.item() * self.grad_accum_steps

                # gradient accumulation boundary
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    # unscale, clip, step optimizer
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.lr_scheduler.step()

                    self.global_step += 1

                    # log every 10 optimizer steps
                    if self.global_step % 10 == 0:
                        avg_loss = self.running_loss / 10
                        lr_now = self.lr_scheduler.get_last_lr()[0]
                        print(
                            f"[epoch {epoch+1}] step {self.global_step}/{self.total_update_steps} "
                            f"loss={avg_loss:.4f} lr={lr_now:.2e}"
                        )
                        self.running_loss = 0.0

                    # periodic checkpoint
                    if self.global_step % 500 == 0:
                        self._save_checkpoint()

            # end-of-epoch checkpoint
            self._save_checkpoint()

        print("Done ✅ — LoRA adapters saved.")


#########################################
# 5. main()
#########################################

if __name__ == "__main__":
    # 1. load model (4-bit base + LoRA adapters)
    model, tokenizer = load_phi4_lora(MODEL_NAME)

    # 2. build dataset / dataloader
    train_ds = Phi4JsonlDataset(JSONL_PATH, tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        collate_fn=Collator(),
    )

    # 3. train
    trainer = Phi4Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        warmup_steps=WARMUP_STEPS,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        output_dir=OUTPUT_DIR,
    )

    trainer.train()
