
import pandas as pd
from datasets import Dataset
from transformers import BartTokenizer

# Load data
train_df = pd.read_csv('/content/drive/MyDrive/indiana_summarization_thesis/data/processed/train_extractive_ner.csv')
val_df = pd.read_csv('/content/drive/MyDrive/indiana_summarization_thesis/data/processed/val_extractive_ner.csv')

# Rename and clean
train_df = train_df.rename(columns={'extractive_summary': 'input_text', 'impression': 'target_text'}).dropna()
val_df = val_df.rename(columns={'extractive_summary': 'input_text', 'impression': 'target_text'}).dropna()

# Convert to HF datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

def tokenize(example):
    input_enc = tokenizer(example['input_text'], max_length=512, truncation=True, padding='max_length')
    target_enc = tokenizer(example['target_text'], max_length=128, truncation=True, padding='max_length')
    input_enc['labels'] = target_enc['input_ids']
    return input_enc

# Tokenize
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# Save
train_dataset.save_to_disk('/content/drive/MyDrive/indiana_summarization_thesis/data/tokenized/train')
val_dataset.save_to_disk('/content/drive/MyDrive/indiana_summarization_thesis/data/tokenized/val')

print('Tokenization complete and saved.')
