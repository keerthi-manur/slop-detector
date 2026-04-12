"""
Train Gemini Detector
Binary classifier: Gemini (label=1) vs Human (label=0)
Saves best model to models/detector_gemini/
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_DIR   = "data"
SAVE_PATH  = "models/detector_gemini"
MAX_LEN    = 256
BATCH_SIZE = 8
EPOCHS     = 1
LR         = 2e-5
TEST_SIZE  = 0.2

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple MPS (Metal)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

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


# main
if __name__ == "__main__":
    print("=== Training Gemini Detector ===")

    ai_df    = pd.read_csv(os.path.join(DATA_DIR, "gemini", "samples.csv"))
    human_df = pd.read_csv(os.path.join(DATA_DIR, "human", "samples.csv"))
    df       = pd.concat([ai_df, human_df], ignore_index=True).sample(frac=1, random_state=42)

    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=42, stratify=labels
    )
    print(f"Train: {len(train_texts)} | Val: {len(val_texts)}")

    print("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    print("Loading model...")
    model     = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    model.to(DEVICE)

    print("Tokenizing training data...")
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    print("Tokenizing validation data...")
    val_dataset   = TextDataset(val_texts,   val_labels,   tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    print(f"Batches per epoch: {len(train_loader)}")

    optimizer   = AdamW(model.parameters(), lr=LR)
    best_val_f1 = 0

# training
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS}: Training ---")
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE)
            )
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()

            if (i + 1) % 50 == 0:
                print(f"  Batch {i+1}/{len(train_loader)} | Avg loss so far: {total_loss/(i+1):.4f}")

# validation
        print(f"--- Epoch {epoch+1}/{EPOCHS}: Validating ---")
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                preds = torch.argmax(
                    model(input_ids=batch["input_ids"].to(DEVICE),
                          attention_mask=batch["attention_mask"].to(DEVICE)).logits, dim=1
                ).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, zero_division=0)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            os.makedirs(SAVE_PATH, exist_ok=True)
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            print(f"  Saved best model (F1: {best_val_f1:.4f}) -> {SAVE_PATH}")

    print(f"\nBest F1: {best_val_f1:.4f}")