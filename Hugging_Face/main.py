"""
Entry point for training and inference using a Wav2Vec2 model.
This script loads the data, filters/shuffles it, and trains the model.
"""

import sys
import torch
from pathlib import Path

# Import functions from our modules
from modules.data_preprocessing_pandas import load_data_with_pandas
from modules.model_definition import get_model_and_processor
from modules.model_training import train_model
from modules.model_inference import predict

def main():
    """
    Main function that:
    1) Locates the data directory.
    2) Loads the dataset (skip file check if we want).
    3) Shuffles/filters the dataset.
    4) Creates the model and processor.
    5) Trains the model, optionally skipping the slow pre-check in train_model.
    6) Saves the model and runs inference on sample audio.
    """

    # Get the absolute path of the directory where THIS file (main.py) is located
    base_dir = Path(__file__).resolve().parent

    # 1) Path to data folder (this is the CORRECT path to your "data" directory)
    data_dir = base_dir / "data"

    # 2) Decide whether we want to skip the file-existence check in load_data_with_pandas
    skip_file_check_global = True  # <--- Toggle True/False here

    # 3) Load the dataset (skip existence check if skip_file_check_global is True)
    dataset = load_data_with_pandas(
        data_dir=str(data_dir),
        skip_exist_check=skip_file_check_global
    )
    train_dataset = dataset["train"]

    # Shuffle & optionally select half
    train_dataset = train_dataset.shuffle(seed=42)
    half_size = len(train_dataset) // 2
    train_dataset = train_dataset.select(range(half_size))

    print(f"Training on half the data: {half_size} samples")
    print("Number of training samples:", len(train_dataset))

    if len(train_dataset) == 0:
        print("ERROR: There is no data to train on. Exiting...")
        sys.exit(1)

    # 4) Get model & processor from Hugging Face
    model, processor = get_model_and_processor("facebook/wav2vec2-base-960h")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 5) Directory for saving checkpoints each epoch
    checkpoint_dir = base_dir / "checkpoints"

    # 6) Train the model
    #    We pass "root_data_dir" to ensure AudioDataset can build correct paths
    model = train_model(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        root_data_dir=str(data_dir),  # <--- pass the correct data folder
        epochs=10,
        batch_size=2,
        learning_rate=1e-5,
        save_checkpoint_dir=str(checkpoint_dir),
    )

    # 7) Final Save
    save_path = base_dir / "saved_model"
    save_path.mkdir(exist_ok=True)

    model.save_pretrained(str(save_path))
    processor.save_pretrained(str(save_path))
    print(f"Model and processor saved to {save_path}")

    # 8) Run inference on sample audio
    sample_audio_paths = [
        str(data_dir / "validated_clips" / "sample_clip_1.wav"),
        str(data_dir / "validated_clips" / "sample_clip_2.wav")
    ]
    predictions = predict(model, processor, sample_audio_paths, device)
    print("Sample predictions:", predictions)

if __name__ == "__main__":
    main()