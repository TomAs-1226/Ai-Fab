# evaluator.py
import math
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate  # Hugging Face evaluate lib. :contentReference[oaicite:12]{index=12}

########################
# 1. Perplexity Dataset
########################
class PerplexityDataset(Dataset):
    """
    JSONL with {"text": "..."} for held-out eval.
    We'll compute log-likelihood and turn that into perplexity.
    Perplexity = exp( avg loss per token ), standard LM metric. :contentReference[oaicite:13]{index=13}
    """
    def __init__(self, path, tokenizer, max_len=512):
        self.samples = []
        self.tok = tokenizer
        self.max_len = max_len
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append(obj["text"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        txt = self.samples[idx]
        enc = self.tok(
            txt,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def compute_perplexity(model, loader, device):
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            # out.loss is average cross-entropy over all tokens that aren't ignored
            # multiply by token count to get total nll
            ce_loss = out.loss
            n_tokens = batch["attention_mask"].sum().item()
            total_loss += ce_loss.item() * n_tokens
            total_tokens += n_tokens
    avg_nll = total_loss / total_tokens
    ppl = math.exp(avg_nll)
    return ppl  # lower perplexity = better next-token prediction. :contentReference[oaicite:14]{index=14}

########################
# 2. ROUGE / BLEU eval
########################
def compute_text_metrics(preds, refs):
    """
    preds[i] = model's generated text
    refs[i] = reference "correct" text
    We'll compute BLEU, ROUGE.
    Hugging Face evaluate supports these directly. :contentReference[oaicite:15]{index=15}
    """
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    bleu_score = bleu_metric.compute(predictions=preds, references=refs)
    rouge_score = rouge_metric.compute(predictions=preds, references=refs)

    return {"bleu": bleu_score, "rouge": rouge_score}

########################
# 3. Example usage
########################
if __name__ == "__main__":
    BASE = "microsoft/Phi-4-mini-instruct"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(BASE).to(DEVICE)

    # Perplexity eval on held-out jsonl:
    ppl_ds = PerplexityDataset(r"C:\path\to\heldout_eval.jsonl", tok)
    ppl_dl = DataLoader(ppl_ds, batch_size=2)
    ppl = compute_perplexity(mdl, ppl_dl, DEVICE)
    print("perplexity:", ppl)

    # ROUGE/BLEU eval on QA/summarization pairs:
    # Suppose you have gold pairs in a list:
    gold = [
        {
            "input": "Summarize: The robot failed to grab coral because ...",
            "reference": "The robot failed to grab coral due to misaligned claw ...",
        },
    ]

    preds = []
    refs = []
    mdl.eval()
    for ex in gold:
        prompt = ex["input"]
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=128)
        pred_text = tok.decode(out[0], skip_special_tokens=True)
        preds.append(pred_text)
        refs.append(ex["reference"])

    scores = compute_text_metrics(preds, refs)
    print("gen metrics:", scores)
