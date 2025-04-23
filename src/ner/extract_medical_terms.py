# src/ner/extract_medical_terms.py

import pandas as pd
import spacy
from pathlib import Path

# Load SciSpaCy model
nlp = spacy.load("en_core_sci_md")

def extract_terms(text):
    doc = nlp(text)
    terms = [ent.text for ent in doc.ents]
    return list(set(terms))  # remove duplicates

# Define paths
splits = {
    "train": "data/processed/train_extractive.csv",
    "val": "data/processed/val_extractive.csv",
    "test": "data/processed/test_extractive.csv"
}

for split, path in splits.items():
    print(f" Extracting medical terms from {split.upper()} set...")
    df = pd.read_csv(path)
    df["medical_terms"] = df["extractive_summary"].apply(extract_terms)

    output_path = Path(f"data/processed/{split}_extractive_ner.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f" Saved: {output_path}")