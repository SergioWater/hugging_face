# clean_filter.py
#  - Removed debug prints.

import pandas as pd
import os

def drop_missing_audio_rows(df, audio_dir):
    # (Removed debug prints about ENTER/EXIT)
    keep_rows = []
    for idx, row in df.iterrows():
        audio_path = row["path"]
        full_path = os.path.join(audio_dir, audio_path)
        if os.path.exists(full_path):
            keep_rows.append(True)
        else:
            keep_rows.append(False)
    return df[keep_rows]

def clean_and_filter_tsv(input_file_path: str, output_file_path: str):
    # (Removed debug prints about ENTER/EXIT)
    data = pd.read_csv(input_file_path, sep='\t', dtype=str)

    numeric_cols = ["up_votes", "down_votes"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    if "up_votes" in data.columns:
        data["up_votes"] = data["up_votes"].fillna(0).astype(int)
    if "down_votes" in data.columns:
        data["down_votes"] = data["down_votes"].fillna(0).astype(int)

    for col in ["accents", "age", "gender"]:
        if col in data.columns:
            data[col] = data[col].fillna("unknown")

    for critical in ["sentence", "path"]:
        if critical in data.columns:
            data = data.dropna(subset=[critical])

    data = data.drop_duplicates()

    if "up_votes" in data.columns and "down_votes" in data.columns:
        data = data[data["up_votes"] > data["down_votes"]]

    if "sentence" in data.columns:
        data = data[data["sentence"].str.len() > 5]
        data = data[data["sentence"].str.len() < 200]

    data.to_csv(output_file_path, sep='\t', index=False)

def main():
    # (Removed debug prints)
    input_dir = "./Hugging_Face/data"
    output_dir = "./Hugging_Face/data/cleaned"

    os.makedirs(output_dir, exist_ok=True)

    files_to_process = ["train.tsv", "dev.tsv", "test.tsv"]
    for tsv_file in files_to_process:
        input_path = os.path.join(input_dir, tsv_file)
        output_path = os.path.join(output_dir, tsv_file)
        clean_and_filter_tsv(input_path, output_path)

if __name__ == "__main__":
    main()
