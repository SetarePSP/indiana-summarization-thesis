import evaluate
import numpy as np
from datasets import load_from_disk
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(" Loading tokenized datasets...")
train_dataset = load_from_disk("/home/spourgholamali/indiana_summarization_thesis/data/tokenized/train")
val_dataset = load_from_disk("/home/spourgholamali/indiana_summarization_thesis/data/tokenized/val")

logger.info(" Loading tokenizer and model...")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 2) for k, v in result.items()}

    logger.info(" Evaluation Results:")
    for metric, score in result.items():
        logger.info(f"{metric.upper()}: {score}")
    return result

training_args = TrainingArguments(
    output_dir="/home/spourgholamali/indiana_summarization_thesis/outputs/models/bart_finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_dir="/home/spourgholamali/indiana_summarization_thesis/logs",
    logging_steps=50,
    save_total_limit=2,
    save_strategy="epoch",
    report_to="none",
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

logger.info("🚀 Starting training...")

resume_from_checkpoint = "/home/spourgholamali/indiana_summarization_thesis/outputs/models/bart_finetuned/checkpoint-1000"
trainer.train(resume_from_checkpoint=resume_from_checkpoint)

logger.info(" Training complete.")
