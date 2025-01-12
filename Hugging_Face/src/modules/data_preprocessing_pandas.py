# [[ data_preprocessing_pandas.py ]]

import pandas as pd
from datasets import Dataset, DatasetDict, Audio
import os


print("DEBUG: data_preprocessing_pandas loaded from:", __file__)
def load_data_with_pandas(data_dir: str):
    """
    data_dir: Directory containing train.tsv, dev.tsv, test.tsv
              plus any audio files in 'clips/' if needed.
    
    This function:
      1. Loads each TSV with pandas.
      2. Converts each to a Hugging Face Dataset.
      3. Casts the 'path' column to an Audio feature.
      4. Returns a DatasetDict with 'train', 'dev', 'test' splits.
    """

    # We'll define the paths explicitly
    train_path = os.path.join(data_dir, "train.tsv")
    dev_path   = os.path.join(data_dir, "dev.tsv")
    test_path  = os.path.join(data_dir, "test.tsv")

    # Load with pandas, forcing everything to string; ignore quotes
    # so that "Benchmark" or any weird tokens won't break us.
    train_df = pd.read_csv(train_path, sep="\t", dtype=str, quoting=3)  # quoting=3 => csv.QUOTE_NONE
    dev_df   = pd.read_csv(dev_path,   sep="\t", dtype=str, quoting=3)
    test_df  = pd.read_csv(test_path,  sep="\t", dtype=str, quoting=3)

    # Convert each pandas DataFrame into a Hugging Face Dataset
    train_hf = Dataset.from_pandas(train_df)
    dev_hf   = Dataset.from_pandas(dev_df)
    test_hf  = Dataset.from_pandas(test_df)

    # Cast the 'path' column to Audio feature if it exists
    # We do a try/except to avoid KeyErrors if 'path' doesn't exist for some reason
    try:
        train_hf = train_hf.cast_column("path", Audio(sampling_rate=16000))
    except:
        pass

    try:
        dev_hf = dev_hf.cast_column("path", Audio(sampling_rate=16000))
    except:
        pass

    try:
        test_hf = test_hf.cast_column("path", Audio(sampling_rate=16000))
    except:
        pass

    # Build a DatasetDict containing all three splits
    dataset = DatasetDict({
        "train": train_hf,
        "dev":   dev_hf,
        "test":  test_hf
    })

    return dataset
