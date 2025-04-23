import evaluate
import numpy as np
from datasets import load_from_disk
from transformers import BartTokenizer, BartForConditionalGeneration

import torch
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load tokenized validation data
val_dataset = load_from_disk("/home/spourgholamali/indiana_summarization_thesis/data/tokenized/val")

# Load tokenizer and model
logger.info(" Loading tokenizer and fine-tuned model...")
tokenizer = BartTokenizer.from_pretrained("/home/spourgholamali/indiana_summarization_thesis/outputs/models/bart_finetuned/checkpoint-2586")
model = BartForConditionalGeneration.from_pretrained("/home/spourgholamali/indiana_summarization_thesis/outputs/models/bart_finetuned/checkpoint-2586")
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# ROUGE 
rouge = evaluate.load("rouge")

# predictions
logger.info("📤 Generating predictions...")
preds = []
labels = []

for example in val_dataset:
    inputs = tokenizer(example["input_text"], return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    label = example["target_text"]

    preds.append(pred.strip())
    labels.append(label.strip())

# ROUGE
logger.info(" Computing ROUGE scores...")
results = rouge.compute(predictions=preds, references=labels, use_stemmer=True)
results = {k: round(v * 100, 2) for k, v in results.items()}

print("\n Evaluation Results:")
for metric, value in results.items():
    print(f"{metric.upper()}: {value}")