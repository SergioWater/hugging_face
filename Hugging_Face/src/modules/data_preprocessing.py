# data_preprocessing.py
#  - Removed debug prints.

import pandas as pd
from datasets import load_dataset, Audio
from datasets import Features, Value
import csv

def load_data(data_dir: str):
    data_files = {
        "train": f"{data_dir}/train.tsv",
        "dev":   f"{data_dir}/dev.tsv",
        "test":  f"{data_dir}/test.tsv"
    }
    # (Removed debug print about data_files)

    dataset = load_dataset(
        "csv",
        data_files=data_files,
        delimiter="\t"
    )

    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))
    return dataset

def preprocess_audio(examples, processor):
    audio_arrays = [x["array"] for x in examples["path"]]
    inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True)
    # (Removed debug print "EXIT preprocess_audio()")
    return inputs
