# 📁 File: ~/indiana_summarization_thesis/scripts/evaluate_bertsum.py

import pandas as pd
import evaluate
from pathlib import Path

# Load ROUGE evaluator
rouge = evaluate.load("rouge")

# clean and strip text 
def clean(text):
    return str(text).strip().replace("\n", " ")

# load data
data_dir = Path("/home/spourgholamali/indiana_summarization_thesis/data/processed")

splits = {
    "train": pd.read_csv(data_dir / "train_extractive.csv"),
    "val": pd.read_csv(data_dir / "val_extractive.csv"),
    "test": pd.read_csv(data_dir / "test_extractive.csv")
}

for split_name, df in splits.items():
    print(f"\n Evaluating {split_name} split ({len(df)} samples)...")

    preds = df["extractive_summary"].apply(clean).tolist()
    targets = df["impression"].apply(clean).tolist()

    result = rouge.compute(predictions=preds, references=targets, use_stemmer=True)
    result = {k: round(v * 100, 2) for k, v in result.items()}

    print(f" ROUGE scores for {split_name}:")
    for metric, value in result.items():
        print(f"  {metric.upper()}: {value}")

