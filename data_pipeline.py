import os
import pandas as pd
from datasets import load_dataset

SAMPLES_PER_CLASS = 1000 
CLAUDE_RAW  = "data/claude_raw.csv"
GEMINI_RAW  = "data/gemini_raw.csv"
OUTPUT_DIR  = "data"

def clean(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())

def save_csv(rows, folder):
    path = os.path.join(OUTPUT_DIR, folder, "samples.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows, columns=["text", "label", "source"])
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows -> {path}")

# Claude
def collect_claude():
    print("\nLoading Claude data (Hello Claude CSV)...")
    df = pd.read_csv(CLAUDE_RAW)
    rows = []
    for _, row in df.iterrows():
        text = clean(row.get("essay_text", ""))
        if len(text) < 100:
            continue
        rows.append([text, 1, "hello_claude"])
        if len(rows) >= SAMPLES_PER_CLASS:
            break
    print(f"  Claude: {len(rows)} samples")
    return rows

# Gemini
def collect_gemini():
    print("\nLoading Gemini data (DAIGT Gemini CSV)...")
    df = pd.read_csv(GEMINI_RAW)
    rows = []
    for _, row in df.iterrows():
        text = clean(row.get("text", ""))
        if len(text) < 100:
            continue
        rows.append([text, 1, "daigt_gemini_pro"])
        if len(rows) >= SAMPLES_PER_CLASS:
            break
    print(f"  Gemini: {len(rows)} samples")
    return rows

# OpenAI
def collect_raid():
    print("\nLoading RAID dataset (ChatGPT + Human)...")
    ds = load_dataset("liamdugan/raid", split="train")

    chatgpt_rows = []
    human_rows   = []

    for row in ds:
        text  = clean(row.get("generation") or row.get("text", ""))
        model = str(row.get("model", "")).lower()

        if len(text) < 100:
            continue

        if model in ("chatgpt", "gpt4") and len(chatgpt_rows) < SAMPLES_PER_CLASS:
            chatgpt_rows.append([text, 1, f"raid/{model}"])
        elif model == "human" and len(human_rows) < SAMPLES_PER_CLASS:
            human_rows.append([text, 0, "raid/human"])

        if len(chatgpt_rows) >= SAMPLES_PER_CLASS and len(human_rows) >= SAMPLES_PER_CLASS:
            break

    print(f"  ChatGPT: {len(chatgpt_rows)} samples")
    print(f"  Human:   {len(human_rows)} samples")
    return chatgpt_rows, human_rows

# main code

if __name__ == "__main__":
    claude_rows              = collect_claude()
    gemini_rows              = collect_gemini()
    chatgpt_rows, human_rows = collect_raid()

    save_csv(claude_rows,  "claude")
    save_csv(gemini_rows,  "gemini")
    save_csv(chatgpt_rows, "chatgpt")
    save_csv(human_rows,   "human")

    for folder in ["claude", "gemini", "chatgpt", "human"]:
        path = os.path.join(OUTPUT_DIR, folder, "samples.csv")
        df = pd.read_csv(path)
        print(f"  {folder}: {len(df)} rows")

    print("\nData pipeline finished")