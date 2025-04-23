import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from pathlib import Path


model_path = "outputs/models/bart_finetuned/checkpoint-2586"
input_csv_path = "data/processed/test_extractive.csv"
output_csv_path = "outputs/inference/predictions_side_by_side.csv"


tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


df = pd.read_csv(input_csv_path)

#  Select  samples
df = df.loc[[13, 12, 45, 50, 56, 74, 110, 203, 204, 220, 222, 224, 226]]

#  Generate abstractive summaries 
predictions = []
for idx, row in df.iterrows():
    input_text = row["extractive_summary"]
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            forced_bos_token_id=tokenizer.bos_token_id
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(summary)

#  Save results 
df["abstractive_summary"] = predictions


df = df.rename(columns={
    "extractive_summary": "Extractive Summary",
    "impression": "Ground Truth",
    "abstractive_summary": "BART Prediction"
})

Path("outputs/inference").mkdir(parents=True, exist_ok=True)
df.to_csv(output_csv_path, index=False)

print(f"Inference complete. Saved to: {output_csv_path}")