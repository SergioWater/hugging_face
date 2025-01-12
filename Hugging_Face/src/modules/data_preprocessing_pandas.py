# [[ data_preprocessing_pandas.py ]]

import pandas as pd
from datasets import Dataset, DatasetDict  # [HIGHLIGHT: Removed 'Audio' because we won't cast it]
import os

print("DEBUG: data_preprocessing_pandas loaded from:", __file__)

def _prepend_clips_if_missing(df):
    fixed_paths = []
    for idx, val in df["path"].items():
        if isinstance(val, str):
            if not val.startswith("clips/"):
                fixed_paths.append(f"clips/{val}")
            else:
                fixed_paths.append(val)
        else:
            fixed_paths.append(val)
    df["path"] = fixed_paths
    return df

def _drop_missing_audio_rows(df, data_dir):
    print(f"Before filtering: {len(df)} rows")
    keep_rows = []
    dropped_count = 0
    for idx, row in df.iterrows():
        audio_path = row["path"]
        if isinstance(audio_path, str):
            full_path = os.path.join(data_dir, audio_path)
            if os.path.exists(full_path):
                keep_rows.append(True)
            else:
                keep_rows.append(False)
                dropped_count += 1
        else:
            keep_rows.append(False)
            dropped_count += 1

    filtered_df = df[keep_rows]
    print(f"DEBUG: Dropped {dropped_count} rows referencing missing audio files.")
    print(f"DEBUG: After filtering, {len(filtered_df)} rows remain.")
    return filtered_df

def load_data_with_pandas(data_dir: str):
    """
    data_dir: Directory containing train.tsv, dev.tsv, test.tsv,
              plus the 'clips/' folder with audio files.

    Steps:
      1) Read each TSV with pandas.
      2) Prepend 'clips/' if missing.
      3) Drop any rows referencing missing audio.
      4) Convert each to a Hugging Face Dataset (string paths only).
    """

    train_path = os.path.join(data_dir, "train.tsv")
    dev_path   = os.path.join(data_dir, "dev.tsv")
    test_path  = os.path.join(data_dir, "test.tsv")

    train_df = pd.read_csv(train_path, sep="\t", dtype=str, quoting=3)
    dev_df   = pd.read_csv(dev_path,   sep="\t", dtype=str, quoting=3)
    test_df  = pd.read_csv(test_path,  sep="\t", dtype=str, quoting=3)

    # Prepend 'clips/' if missing + filter
    train_df = _prepend_clips_if_missing(train_df)
    train_df = _drop_missing_audio_rows(train_df, data_dir)
    dev_df   = _prepend_clips_if_missing(dev_df)
    dev_df   = _drop_missing_audio_rows(dev_df, data_dir)
    test_df  = _prepend_clips_if_missing(test_df)
    test_df  = _drop_missing_audio_rows(test_df, data_dir)

    # [HIGHLIGHT: We do NOT cast "path" to Audio anymore]
    train_hf = Dataset.from_pandas(train_df)
    dev_hf   = Dataset.from_pandas(dev_df)
    test_hf  = Dataset.from_pandas(test_df)

    return DatasetDict({
        "train": train_hf,
        "dev":   dev_hf,
        "test":  test_hf
    })
