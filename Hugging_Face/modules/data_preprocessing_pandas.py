"""
Loads the dataset using pandas, optionally checks file existence once, and returns a DatasetDict.
"""

import pandas as pd
import os
from datasets import Dataset, DatasetDict

def _drop_missing_audio_rows(df, data_dir):
    """
    Iterates over each row in the DataFrame, ensuring the audio file exists
    in data_dir/validated_clips. If missing, that row is excluded.
    """
    print("Size before dropping:", len(df))
    keep_rows = []

    for idx, row in df.iterrows():
        audio_path = row["path"]
        # "validated_clips" subfolder is inside data_dir
        full_path = os.path.join(data_dir, "validated_clips", audio_path)

        print(f"Checking {full_path}... Exists? {os.path.exists(full_path)}")

        if isinstance(audio_path, str) and os.path.exists(full_path):
            keep_rows.append(True)
        else:
            keep_rows.append(False)

    filtered_df = df[keep_rows]
    print("Size after dropping:", len(filtered_df))
    return filtered_df

def load_data_with_pandas(data_dir: str, skip_exist_check=True):
    """
    1) Reads 'validated.tsv' from data_dir.
    2) If skip_exist_check=False, drops rows whose audio files don't exist (one-time check).
       If skip_exist_check=True, we skip checking file existence entirely.
    3) Converts the DataFrame to a Hugging Face DatasetDict with a 'train' split.
    """
    validated_path = os.path.join(data_dir, "validated.tsv")
    validated_df = pd.read_csv(validated_path, sep='\t', dtype=str, quoting=3)
    print(f"Loaded validated.tsv with {len(validated_df)} rows")

    print("Sample sentences from validated.tsv:")
    print(validated_df["sentence"].head(10))

    # Conditionally skip the file-existence check
    if not skip_exist_check:
        validated_df = _drop_missing_audio_rows(validated_df, data_dir)
    else:
        print("Skipping file-existence check. Assuming data is already verified...")

    print("After optional dropping of missing audio rows, here are some sentences:")
    print(validated_df["sentence"].head(10))

    # Convert DataFrame -> Hugging Face Dataset
    validated_hf = Dataset.from_pandas(validated_df)

    # Put that into a DatasetDict so we have a "train" split
    dataset_dict = DatasetDict({"train": validated_hf})

    return dataset_dict