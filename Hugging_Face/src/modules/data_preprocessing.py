# [[ data_preprocessing.py ]]

import pandas as pd
from datasets import load_dataset, Audio
from datasets import Features, Value
import csv

def load_data(data_dir: str):
    """
    data_dir: The path containing train.tsv, dev.tsv, test.tsv,
              plus the 'clips/' directory with the actual audio files.
    """
    # [HIGHLIGHT: CHANGED LINES BELOW]
    # Now we define all three (train, dev, test) so we can do dataset["train"], dataset["dev"], dataset["test"]
    data_files = {
        "train": f"{data_dir}/train.tsv",
        "dev":   f"{data_dir}/dev.tsv",
        "test":  f"{data_dir}/test.tsv"
    }
    print(f"========== [DEBUG] data_files = {data_files} ==========")

    dataset = load_dataset(
        "csv",
        data_files=data_files,
        delimiter="\t"
    )
    # [END OF CHANGED LINES]

    # Cast the 'path' column to an Audio feature (16kHz sample rate) for each split
    # If you'd like, you can do it in a loop for train/dev/test, but .cast_column is lazy and will only apply to existing columns.
    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

    return dataset

def preprocess_audio(examples, processor):
    """
    This function will load the audio from disk (via 'dataset.cast_column(Audio)'),
    and tokenize/encode it using the Wav2Vec2 processor.
    """
    audio_arrays = [x["array"] for x in examples["path"]]
    inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True)
    print("========== [DEBUG] EXIT preprocess_audio() ==========")
    return inputs
