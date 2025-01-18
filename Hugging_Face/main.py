import sys
import torch
from pathlib import Path

# Import functions from our modules
from modules.data_preprocessing_pandas import load_data_with_pandas
from modules.model_definition import get_model_and_processor
from modules.model_training import train_model
from modules.model_inference import predict

def main():
    # Get the absolute path of the directory where THIS file (main.py) is located
    base_dir = Path(__file__).resolve().parent

    # 1) Path to data folder
    data_dir = base_dir / "data"
    # 2) Path to the CSV/TSV files (like validated.tsv)
    #    We will load these with load_data_with_pandas().

    # 3) Load the dataset
    dataset = load_data_with_pandas(str(data_dir))
    train_dataset = dataset["train"]

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
    model = train_model(
        model,
        processor,
        train_dataset,
        epochs=10,
        batch_size=2,  # smaller batch to reduce memory usage
        learning_rate=1e-5,
        save_checkpoint_dir=str(checkpoint_dir)  # must be a string for .save_pretrained()
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