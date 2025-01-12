# [[ data_preprocessing_pandas.py ]]

import pandas as pd
from datasets import Dataset, DatasetDict
import os

print("========== [DEBUG] data_preprocessing_pandas loaded from:", __file__, "==========")

def _prepend_clips_if_missing(df):
    print("========== [DEBUG] ENTER _prepend_clips_if_missing ==========")
    fixed_paths = []
    for idx, val in df["path"].items():
        print(f"========== [DEBUG] path val={val} at idx={idx} ==========")
        if isinstance(val, str):
            # If your folder is "clips_data", handle that. Otherwise "clips/"
            if not val.startswith("clips/") and not val.startswith("clips_data/"):
                new_path = f"clips_data/{val}"
                print(f"========== [DEBUG] PREPEND: {val} -> {new_path} ==========")
                fixed_paths.append(new_path)
            else:
                print(f"========== [DEBUG] NO PREPEND needed for {val} ==========")
                fixed_paths.append(val)
        else:
            print(f"========== [DEBUG] path val is NOT a string => {val}. We'll keep as is.")
            fixed_paths.append(val)
    df["path"] = fixed_paths
    print("========== [DEBUG] EXIT _prepend_clips_if_missing ==========")
    return df

def _drop_missing_audio_rows(df, data_dir):
    print("========== [DEBUG] ENTER _drop_missing_audio_rows ==========")
    print(f"========== [DEBUG] df size: {len(df)} | data_dir={data_dir} ==========")
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
    print(f"========== [DEBUG] Dropped {dropped_count} rows referencing missing audio files ==========")
    print(f"========== [DEBUG] After filtering, {len(filtered_df)} rows remain ==========")
    print("========== [DEBUG] EXIT _drop_missing_audio_rows ==========")
    return filtered_df

def load_data_with_pandas(data_dir: str):
    print(f"========== [DEBUG] ENTER load_data_with_pandas. data_dir={data_dir} ==========")

    train_path = os.path.join(data_dir, "train.tsv")
    dev_path   = os.path.join(data_dir, "dev.tsv")
    test_path  = os.path.join(data_dir, "test.tsv")

    train_df = pd.read_csv(train_path, sep="\t", dtype=str, quoting=3)
    dev_df   = pd.read_csv(dev_path,   sep="\t", dtype=str, quoting=3)
    test_df  = pd.read_csv(test_path,  sep="\t", dtype=str, quoting=3)

    print(f"[DEBUG] train_df loaded with {len(train_df)} rows")
    print(f"[DEBUG] dev_df loaded with {len(dev_df)} rows")
    print(f"[DEBUG] test_df loaded with {len(test_df)} rows")

    train_df = _prepend_clips_if_missing(train_df)
    train_df = _drop_missing_audio_rows(train_df, data_dir)

    dev_df   = _prepend_clips_if_missing(dev_df)
    dev_df   = _drop_missing_audio_rows(dev_df, data_dir)

    test_df  = _prepend_clips_if_missing(test_df)
    test_df  = _drop_missing_audio_rows(test_df, data_dir)

    print("========== [DEBUG] Creating Hugging Face Datasets from pandas dataframes ==========")
    train_hf = Dataset.from_pandas(train_df)
    dev_hf   = Dataset.from_pandas(dev_df)
    test_hf  = Dataset.from_pandas(test_df)

    dataset_dict = DatasetDict({
        "train": train_hf,
        "dev":   dev_hf,
        "test":  test_hf
    })
    print("========== [DEBUG] EXIT load_data_with_pandas, returning DatasetDict ==========")
    return dataset_dict
