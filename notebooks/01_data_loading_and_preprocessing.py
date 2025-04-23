# notebooks/01_data_loading_and_preprocessing.py

import pandas as pd
from pathlib import Path

# Load the original Indiana reports dataset
raw_path = Path('data/raw/indiana_reports.csv')
df = pd.read_csv(raw_path)

# Show basic info
print("🔍 Original dataset shape:", df.shape)
print("🧼 Missing values:\n", df.isnull().sum())

# Drop rows where findings or impression is missing
df_clean = df.dropna(subset=['findings', 'impression'])
df_clean.reset_index(drop=True, inplace=True)

print("✅ Cleaned dataset shape:", df_clean.shape)

# Save cleaned dataset
processed_path = Path('data/processed/indiana_clean.csv')
processed_path.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(processed_path, index=False)

print(f"📁 Cleaned data saved to: {processed_path}")