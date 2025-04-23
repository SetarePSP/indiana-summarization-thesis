import logging
import evaluate
import numpy as np
from datasets import load_from_disk
from transformers import BartTokenizer, BartForConditionalGeneration

# Set up
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(" Loading tokenizer and fine-tuned model...")
tokenizer = BartTokenizer.from_pretrained(
    "/home/spourgholamali/indiana_summarization_thesis/outputs/models/bart_finetuned/checkpoint-2586"
)
model = BartForConditionalGeneration.from_pretrained(
    "/home/spourgholamali/indiana_summarization_thesis/outputs/models/bart_finetuned/checkpoint-2586"
)

logger.info(" Loading tokenized test dataset...")
test_dataset = load_from_disk(
    "/home/spourgholamali/indiana_summarization_thesis/data/tokenized/test"
)

# ROUGE scorer
rouge = evaluate.load("rouge")

# Run inference
logger.info(" Generating predictions...")
preds = []
labels = []

for example in test_dataset:
    inputs = tokenizer(example["input_text"], return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    output_ids = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    label = example["target_text"]

    preds.append(pred.strip())
    labels.append(label.strip())

# ROUGE
logger.info(" Computing ROUGE scores...")
results = rouge.compute(predictions=preds, references=labels, use_stemmer=True)
results = {k: round(v * 100, 2) for k, v in results.items()}

print("\n Test Evaluation Results:")
for metric, value in results.items():
    print(f"{metric.upper()}: {value}")

# predictions 
with open("/home/spourgholamali/indiana_summarization_thesis/outputs/predictions_test.txt", "w") as f:
    for i, (inp, ref, hyp) in enumerate(zip(test_dataset["input_text"], labels, preds)):
        f.write(f"=== Sample {i + 1} ===\n")
        f.write(f"Input: {inp}\n")
        f.write(f"Reference: {ref}\n")
        f.write(f"Prediction: {hyp}\n\n")

print(" Predictions saved to predictions_test.txt")
