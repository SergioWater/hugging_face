# [[ data_preprocessing.py ]]

import pandas as pd
from datasets import load_dataset, Audio

def load_data(data_dir: str):
    """
    data_dir: The path containing train.tsv, dev.tsv, test.tsv,
              plus the 'clips/' directory with the actual audio files.
    """
    # Load data using the datasets library, specifying CSV + tab delimiter
    dataset = load_dataset(
        "csv",
        data_files={"train": f"{data_dir}/train_head10k.tsv",},
        delimiter="\t"
    )
    # Cast the 'path' column to an Audio feature (16kHz sample rate)
    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

    return dataset


def preprocess_audio(examples, processor):
    """
    This function will load the audio from disk (via 'dataset.cast_column(Audio)'),
    and tokenize/encode it using the Wav2Vec2 processor.
    """
    audio_arrays = [x["array"] for x in examples["path"]]
    inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True)
    return inputs
