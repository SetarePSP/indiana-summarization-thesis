import logging
from datasets import load_from_disk
from transformers import BartTokenizer, BartForConditionalGeneration
from bert_score import score


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import transformers
import torch
logger.info(f" Transformers version: {transformers.__version__}")
logger.info(f" Python: {torch.__version__}")
logger.info(" Loading tokenizer and fine-tuned model...")

# Load tokenizer and model checkpoint 2586
tokenizer = BartTokenizer.from_pretrained(
    "/home/spourgholamali/indiana_summarization_thesis/outputs/models/bart_finetuned/checkpoint-2586"
)
model = BartForConditionalGeneration.from_pretrained(
    "/home/spourgholamali/indiana_summarization_thesis/outputs/models/bart_finetuned/checkpoint-2586"
)

def evaluate_split(name, dataset_path):
    logger.info(f" Loading {name} set...")
    dataset = load_from_disk(dataset_path)

    preds = []
    labels = []

    logger.info(f"Generating summaries for {name} set...")
    for example in dataset:
        inputs = tokenizer(example["input_text"], return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        output_ids = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        label = example["target_text"]
        preds.append(pred.strip())
        labels.append(label.strip())

    logger.info(f" Computing BERTScore for {name} set...")
    P, R, F1 = score(preds, labels, lang="en", model_type="roberta-large")
    logger.info(f" {name.upper()} BERTScore F1: {F1.mean().item():.4f}")

#  evaluations on all splits
evaluate_split("train", "/home/spourgholamali/indiana_summarization_thesis/data/tokenized/train")
evaluate_split("val", "/home/spourgholamali/indiana_summarization_thesis/data/tokenized/val")
evaluate_split("test", "/home/spourgholamali/indiana_summarization_thesis/data/tokenized/test")

logger.info("BERTScore evaluation complete.")
