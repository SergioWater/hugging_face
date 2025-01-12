# [[ data_preprocessing_pandas.py ]]

import pandas as pd
from datasets import Dataset, DatasetDict, Audio
import os

print("DEBUG: data_preprocessing_pandas loaded from:", __file__)

# [HIGHLIGHT: ADDED FUNCTION]
def _prepend_clips_if_missing(df):
    """
    If the 'path' column is like 'common_voice_en_XXXX.mp3',
    turn it into 'clips/common_voice_en_XXXX.mp3'.
    If it already starts with 'clips/', do nothing.
    """
    fixed_paths = []
    for idx, val in df["path"].items():
        if isinstance(val, str):
            if not val.startswith("clips/"):
                fixed_paths.append(f"clips/{val}")
            else:
                fixed_paths.append(val)
        else:
            # If it's somehow NaN or not a string, just keep it
            fixed_paths.append(val)
    df["path"] = fixed_paths
    return df
# [END HIGHLIGHT]

# [HIGHLIGHT: ADDED FUNCTION]
def _drop_missing_audio_rows(df, data_dir):
    """
    Remove rows if 'path' doesn't exist in 'data_dir/path'.
    """
    keep_rows = []
    for idx, row in df.iterrows():
        audio_path = row["path"]
        if isinstance(audio_path, str):
            full_path = os.path.join(data_dir, audio_path)
            if os.path.exists(full_path):
                keep_rows.append(True)
            else:
                keep_rows.append(False)
        else:
            keep_rows.append(False)
    return df[keep_rows]
# [END HIGHLIGHT]


def load_data_with_pandas(data_dir: str):
    """
    data_dir: Directory containing train.tsv, dev.tsv, test.tsv,
              plus a 'clips/' folder for actual audio files.

    Steps:
      1) Read each TSV with pandas (strings only).
      2) Fix 'path' if missing 'clips/' prefix.
      3) Drop any rows referencing non-existing audio.
      4) Convert each to a Hugging Face Dataset.
      5) Cast 'path' to Audio feature.
      6) Return a DatasetDict with 'train', 'dev', 'test'.
    """

    train_path = os.path.join(data_dir, "train.tsv")
    dev_path   = os.path.join(data_dir, "dev.tsv")
    test_path  = os.path.join(data_dir, "test.tsv")

    train_df = pd.read_csv(train_path, sep="\t", dtype=str, quoting=3)  # quoting=3 => csv.QUOTE_NONE
    dev_df   = pd.read_csv(dev_path,   sep="\t", dtype=str, quoting=3)
    test_df  = pd.read_csv(test_path,  sep="\t", dtype=str, quoting=3)

    # [HIGHLIGHT: FIX PATHS + FILTER MISSING]
    train_df = _prepend_clips_if_missing(train_df)
    train_df = _drop_missing_audio_rows(train_df, data_dir)

    dev_df = _prepend_clips_if_missing(dev_df)
    dev_df = _drop_missing_audio_rows(dev_df, data_dir)

    test_df = _prepend_clips_if_missing(test_df)
    test_df = _drop_missing_audio_rows(test_df, data_dir)
    # [END HIGHLIGHT]

    train_hf = Dataset.from_pandas(train_df)
    dev_hf   = Dataset.from_pandas(dev_df)
    test_hf  = Dataset.from_pandas(test_df)

    # Try to cast 'path' to Audio feature
    # If there's no 'path' column, just skip
    try:
        train_hf = train_hf.cast_column("path", Audio(sampling_rate=16000))
    except KeyError:
        pass

    try:
        dev_hf = dev_hf.cast_column("path", Audio(sampling_rate=16000))
    except KeyError:
        pass

    try:
        test_hf = test_hf.cast_column("path", Audio(sampling_rate=16000))
    except KeyError:
        pass

    # Build a DatasetDict
    dataset = DatasetDict({
        "train": train_hf,
        "dev":   dev_hf,
        "test":  test_hf
    })

    return dataset
