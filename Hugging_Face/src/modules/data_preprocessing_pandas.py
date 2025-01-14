# [HIGHLIGHT: CHANGED LINES]
# - We are removing references to train.tsv, dev.tsv, and test.tsv.
# - We only load validated.tsv, store it in validated_df, then treat that as "train" in the returned DatasetDict.
# - We also prepend "validated_clips/" if missing instead of "clips/".

import pandas as pd
from datasets import Dataset, DatasetDict
import os

# print("========== [DEBUG] data_preprocessing_pandas loaded from: $$$$$$$$$", __file__, "$$$$$$$$$ ==========")

def _prepend_validated_clips_if_missing(df):
    # print("========== [DEBUG] ENTER _prepend_validated_clips_if_missing ==========")
    fixed_paths = []
    for idx, val in df["path"].items():
        # print(f"========== [DEBUG] path val={val} at idx={idx} ==========")
        if isinstance(val, str):
            # Only check if not already starts with "validated_clips/"
            if not val.startswith("validated_clips/"):
                new_path = f"validated_clips/{val}"
                # print(f"========== [DEBUG] PREPEND: {val} -> {new_path} ==========")
                fixed_paths.append(new_path)
            else:
                # print(f"========== [DEBUG] NO PREPEND needed for {val} ==========")
                fixed_paths.append(val)
        else:
            # print(f"========== [DEBUG] path val is NOT a string => {val}. We'll keep as is.")
            fixed_paths.append(val)
    df["path"] = fixed_paths
    # print("========== [DEBUG] EXIT _prepend_validated_clips_if_missing ==========")
    return df

def _drop_missing_audio_rows(df, data_dir):
    # print("========== [DEBUG] ENTER _drop_missing_audio_rows ==========")
    # print(f"========== [DEBUG] Initial DataFrame size: {len(df)} ==========")
    # print(f"========== [DEBUG] Data directory: {data_dir} ==========")
    
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
    # print(f"========== [DEBUG] Dropped {dropped_count} rows referencing missing or invalid audio files ==========")
    # print(f"========== [DEBUG] Remaining DataFrame size: {len(filtered_df)} ==========")
    # print("========== [DEBUG] EXIT _drop_missing_audio_rows ==========")
    return filtered_df


def load_data_with_pandas(data_dir: str):
    # print(f"========== [DEBUG] ENTER load_data_with_pandas. data_dir={data_dir} ==========")

    # [HIGHLIGHT: CHANGED]
    # We only read validated.tsv. The name of the file is "validated.tsv" 
    # Then we create "train" from it. We do not do dev/test anymore.

    validated_path = os.path.join(data_dir, "validated.tsv")
    validated_df = pd.read_csv(validated_path, engine="python", sep=r"\t+(\s+)?", dtype=str, quoting=3)

    # print(f"[DEBUG] validated_df loaded with {len(validated_df)} rows")

    # [HIGHLIGHT: CHANGED]
    # Instead of prepending "clips/", we now prepend "validated_clips/" 
    validated_df = _prepend_validated_clips_if_missing(validated_df)
    validated_df = _drop_missing_audio_rows(validated_df, data_dir)

    # print("========== [DEBUG] Creating Hugging Face Dataset from pandas dataframe ==========")
    validated_hf = Dataset.from_pandas(validated_df)

    # [HIGHLIGHT: CHANGED]
    # Return a DatasetDict with only a 'train' key
    dataset_dict = DatasetDict({
        "train": validated_hf
    })

    # print("========== [DEBUG] EXIT load_data_with_pandas, returning DatasetDict with single train split ==========")
    return dataset_dict
