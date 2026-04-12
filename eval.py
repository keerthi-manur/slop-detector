"""
Cross-Model Evaluation
Tests each detector against all three AI datasets to produce a 3x3 results matrix.
Rows    = which model the detector was trained on
Columns = which dataset it is being tested on
Saves results to results/cross_eval_results.csv
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MODELS_DIR  = "models"
DATA_DIR    = "data"
RESULTS_DIR = "results"
MAX_LEN     = 256
BATCH_SIZE  = 8

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

DETECTORS = {
    "claude":  "models/detector_claude",
    "chatgpt": "models/detector_chatgpt",
    "gemini":  "models/detector_gemini",
}

AI_DATASETS = ["claude", "chatgpt", "gemini"]


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }


def evaluate(model, tokenizer, ai_source):
    """Test a loaded model against one AI dataset + human samples."""
    ai_df    = pd.read_csv(os.path.join(DATA_DIR, ai_source, "samples.csv"))
    human_df = pd.read_csv(os.path.join(DATA_DIR, "human", "samples.csv"))
    df       = pd.concat([ai_df, human_df], ignore_index=True).sample(frac=1, random_state=99)

    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    dataset = TextDataset(texts, labels, tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            preds = torch.argmax(
                model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE)
                ).logits, dim=1
            ).cpu().tolist()
            all_preds.extend(preds)

            if (i + 1) % 50 == 0:
                print(f"    Batch {i+1}/{len(loader)}")

    acc  = accuracy_score(labels, all_preds)
    f1   = f1_score(labels, all_preds, zero_division=0)
    prec = precision_score(labels, all_preds, zero_division=0)
    rec  = recall_score(labels, all_preds, zero_division=0)

    return {"acc": acc, "f1": f1, "precision": prec, "recall": rec}


# main
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for detector_name, model_path in DETECTORS.items():
        print(f"\n=== Loading detector trained on: {detector_name} ===")
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model     = RobertaForSequenceClassification.from_pretrained(model_path)
        model.to(DEVICE)

        for ai_source in AI_DATASETS:
            print(f"  Testing on: {ai_source} data...")
            metrics = evaluate(model, tokenizer, ai_source)
            print(f"  Acc: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")

            rows.append({
                "detector_trained_on": detector_name,
                "tested_on":           ai_source,
                "accuracy":            round(metrics["acc"],  4),
                "f1":                  round(metrics["f1"],   4),
                "precision":           round(metrics["prec"] if "prec" in metrics else metrics["precision"], 4),
                "recall":              round(metrics["recall"], 4),
            })

    results_df = pd.DataFrame(rows)
    out_path   = os.path.join(RESULTS_DIR, "cross_eval_results.csv")
    results_df.to_csv(out_path, index=False)

    print("\n=== Cross-Evaluation Matrix (F1) ===")
    matrix = results_df.pivot(index="detector_trained_on", columns="tested_on", values="f1")
    print(matrix.to_string())
    print(f"\nFull results saved to {out_path}")