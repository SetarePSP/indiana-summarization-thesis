import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import nltk
from nltk.tokenize import sent_tokenize
from torch.nn.functional import cosine_similarity
from sklearn.model_selection import train_test_split
from pathlib import Path

# Download sentence tokenizer
nltk.download("punkt")

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Helper Functions
def get_sentence_embeddings(text):
    sentences = sent_tokenize(text)
    embeddings = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[0][0]  # CLS token embedding
        embeddings.append(cls_embedding)

    return sentences, embeddings

def extract_top_k_sentences(text, k=3):
    sentences, embeddings = get_sentence_embeddings(text)
    if len(sentences) <= k:
        return " ".join(sentences)

    doc_embedding = torch.stack(embeddings).mean(dim=0)
    scores = [cosine_similarity(e.unsqueeze(0), doc_embedding.unsqueeze(0)).item() for e in embeddings]
    top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    selected = [sentences[i] for i in sorted(top_k_idx)]
    return " ".join(selected)

#  Split Data
df = pd.read_csv("data/processed/indiana_clean.csv")

train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

print(f" Split sizes → Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# BERTSUM to Each Split
print(" Extracting from TRAIN split...")
train["extractive_summary"] = train["findings"].apply(lambda x: extract_top_k_sentences(x, k=3))

print(" Extracting from VAL split...")
val["extractive_summary"] = val["findings"].apply(lambda x: extract_top_k_sentences(x, k=3))

print(" Extracting from TEST split...")
test["extractive_summary"] = test["findings"].apply(lambda x: extract_top_k_sentences(x, k=3))

# Save
Path("data/processed").mkdir(parents=True, exist_ok=True)
train.to_csv("data/processed/train_extractive.csv", index=False)
val.to_csv("data/processed/val_extractive.csv", index=False)
test.to_csv("data/processed/test_extractive.csv", index=False)

print(" All processed splits saved with extractive summaries!")