#  Radiology Report Summarization Thesis

This project builds a hybrid summarization pipeline for radiology reports using extractive, medical term filtering, and abstractive summarization techniques.

## Model Architecture

1. **BERTSUM** – Extractive summarization using cosine similarity.
2. **SciSpacy NER** – Medical term filtering with `en_core_sci_md`.
3. **BART** – Abstractive summarization with fine-tuning.
4. **Evaluation** – ROUGE + BERTScore metrics on all data splits.

## Dataset

- **Indiana University Chest X-Ray Dataset**
- `findings` used as input; `impression` used as ground-truth summary.
- Data split: `train`, `val`, `test`.

## Components

- `src/extractive/bertsum_extractive.py` – BERTSUM implementation
- `src/ner/extract_medical_terms.py` – NER extraction
- `scripts/tokenize_data.py` – Converts CSV to tokenized HuggingFace datasets
- `scripts/train_bart.py` – Fine-tunes BART
- `scripts/evaluate_bart.py` – ROUGE on validation
- `scripts/eval_bart_bert_score.py` – BERTScore for BART
- `scripts/eval_bertsum_bert_score.py` – BERTScore for BERTSUM

##  Results

| Model   | Split | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|---------|-------|---------|---------|---------|---------------|
| BERTSUM | Val   | ~13.85  | ~5.34   | ~13.19  | 0.8559        |
| BART    | Val   | ~62.15  | ~53.79  | ~61.90  | 0.9355        |
| BART    | Test  | ~58.19  | ~49.53  | ~57.84  | 0.9293        |

##  Training Setup

-  Google Colab (initial fine-tuning)
-  Polito HPC (GPU training + full evaluation)
-  Conda environment (`thesis-gpu` on pc, `bartenv` on HPC)

##  Folder Structure
indiana_summarization_thesis/
├── data/
│   ├── processed/
│   └── tokenized/
├── outputs/
│   └── models/bart_finetuned/
├── scripts/
├── src/
├── logs/
├── README.md
└── environment.yml


##  License

This project is for academic purposes only.
