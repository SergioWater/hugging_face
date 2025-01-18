import pandas as pd
import os
from datasets import Dataset, DatasetDict

def _drop_missing_audio_rows(df, data_dir):
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

def load_data_with_pandas(data_dir: str):
    """
    data_dir might be something like /path/to/my_project/data
    which contains validated.tsv or train.tsv, dev.tsv, test.tsv, etc.
    """
    validated_path = os.path.join(data_dir, "validated.tsv")
    validated_df = pd.read_csv(validated_path, sep='\t', dtype=str, quoting=3)
    print(f"Loaded validated.tsv with {len(validated_df)} rows")

    print("Sample sentences from validated.tsv:")
    print(validated_df["sentence"].head(10))

    # Drop rows if the audio file doesn't physically exist
    validated_df = _drop_missing_audio_rows(validated_df, data_dir)

    print("After dropping missing audio rows, here are some of the sentences:")
    print(validated_df["sentence"].head(10))

    # Convert DataFrame -> Hugging Face Dataset
    validated_hf = Dataset.from_pandas(validated_df)

    # Put that into a DatasetDict so we have a "train" split
    dataset_dict = DatasetDict({"train": validated_hf})

    return dataset_dict