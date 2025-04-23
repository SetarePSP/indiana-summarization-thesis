import pandas as pd
from datasets import Dataset
from transformers import BartTokenizer

# Load test set
test_df = pd.read_csv("/home/spourgholamali/indiana_summarization_thesis/data/processed/test_extractive_ner.csv")

# Rename and clean
test_df = test_df.rename(columns={'extractive_summary': 'input_text', 'impression': 'target_text'}).dropna()

# Convert to HF dataset
test_dataset = Dataset.from_pandas(test_df)

# Tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

def tokenize(example):
    input_enc = tokenizer(example['input_text'], max_length=512, truncation=True, padding='max_length')
    target_enc = tokenizer(example['target_text'], max_length=128, truncation=True, padding='max_length')
    input_enc['labels'] = target_enc['input_ids']
    return input_enc

# Tokenize
test_dataset = test_dataset.map(tokenize, batched=True)

# Save tokenized test set
test_dataset.save_to_disk("/home/spourgholamali/indiana_summarization_thesis/data/tokenized/test")
print(" Tokenized test set saved.")
