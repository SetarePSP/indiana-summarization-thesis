import pandas as pd
import logging
from bert_score import score


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# datasets
logger.info(" Loading BERTSUM evaluation data...")
train_df = pd.read_csv("/home/spourgholamali/indiana_summarization_thesis/data/processed/train_extractive.csv")
val_df = pd.read_csv("/home/spourgholamali/indiana_summarization_thesis/data/processed/val_extractive.csv")
test_df = pd.read_csv("/home/spourgholamali/indiana_summarization_thesis/data/processed/test_extractive.csv")

def evaluate_bertscore(df, split_name):
    logger.info(f" Evaluating {split_name} set with {len(df)} samples...")

    cands = df["extractive_summary"].astype(str).tolist()
    refs = df["impression"].astype(str).tolist()

    P, R, F1 = score(cands, refs, lang="en", verbose=True)
    logger.info(f" {split_name.upper()} BERTScore F1: {F1.mean().item():.4f}")

# Evaluate
evaluate_bertscore(train_df, "train")
evaluate_bertscore(val_df, "val")
evaluate_bertscore(test_df, "test")

logger.info(" Semantic evaluation complete.")
