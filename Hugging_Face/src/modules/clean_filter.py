# [[ clean_filter.py ]]

import pandas as pd
import os

def clean_and_filter_tsv(input_file_path: str, output_file_path: str):
    """
    Reads a TSV file, cleans + filters its rows, then saves to output_file_path.
    This version avoids chained assignment warnings by not using 'inplace=True'
    when filling missing values.
    """
    # Load the TSV as strings (to prevent dtype issues)
    data = pd.read_csv(input_file_path, sep='\t', dtype=str)

    # Convert numeric columns (if present) to int safely
    numeric_cols = ["up_votes", "down_votes"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Fill missing numeric values with 0
    if "up_votes" in data.columns:
        data["up_votes"] = data["up_votes"].fillna(0).astype(int)
    if "down_votes" in data.columns:
        data["down_votes"] = data["down_votes"].fillna(0).astype(int)

    # Fill missing string columns with 'unknown'
    for col in ["accents", "age", "gender"]:
        if col in data.columns:
            data[col] = data[col].fillna("unknown")

    # Drop rows missing 'sentence' or 'path' if these columns exist
    for critical in ["sentence", "path"]:
        if critical in data.columns:
            data = data.dropna(subset=[critical])

    # Remove duplicates
    data = data.drop_duplicates()

    # Filter out rows where up_votes <= down_votes
    if "up_votes" in data.columns and "down_votes" in data.columns:
        data = data[data["up_votes"] > data["down_votes"]]

    # Filter by sentence length: >5 and <200
    if "sentence" in data.columns:
        data = data[data["sentence"].str.len() > 5]
        data = data[data["sentence"].str.len() < 200]

    # Write the cleaned data to output
    data.to_csv(output_file_path, sep='\t', index=False)
    print(f"Cleaned data saved to: {output_file_path}")

def main():
    """
    Example usage of the cleaning function:
      1) Adjust input_dir, output_dir
      2) List the TSVs to process
    """
    input_dir = "./Hugging_Face/data"
    output_dir = "./Hugging_Face/data/cleaned"

    os.makedirs(output_dir, exist_ok=True)

    files_to_process = ["train.tsv", "dev.tsv", "test.tsv"]

    for tsv_file in files_to_process:
        input_path = os.path.join(input_dir, tsv_file)
        output_path = os.path.join(output_dir, tsv_file)

        print(f"Processing {input_path}...")
        clean_and_filter_tsv(input_path, output_path)

    print("All files have been processed and saved.")

if __name__ == "__main__":
    main()
