# [main.py]
import sys
import torch
from modules.data_preprocessing_pandas import load_data_with_pandas
from modules.model_definition import get_model_and_processor
from modules.model_training import train_model
from modules.model_inference import predict

def main():
    data_dir = "/Users/water/Documents/Coding/Ai/hugging_face/Hugging_Face/data/"
    dataset = load_data_with_pandas(data_dir)
    train_dataset = dataset["train"]

    print("Number of training samples:", len(train_dataset))
    if len(train_dataset) == 0:
        print("ERROR: There is no data to train on. Exiting...")
        sys.exit(1)

    model, processor = get_model_and_processor("facebook/wav2vec2-base-960h")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # NEW: We specify a directory for saving checkpoints
    checkpoint_dir = "/Users/water/Documents/Coding/Ai/hugging_face/Hugging_Face/checkpoints/"

    model = train_model(
        model,
        processor,
        train_dataset,
        epochs=10,
        batch_size=2,  # smaller batch to reduce memory usage
        learning_rate=1e-5,
        save_checkpoint_dir=checkpoint_dir  # <--- new
    )

    # Final Save
    save_path = "/Users/water/Documents/Coding/Ai/hugging_face/Hugging_Face/saved_model/"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"Model and processor saved to {save_path}")

    sample_audio_paths = [
        f"{data_dir}wav_clips/sample_clip_1.wav",
        f"{data_dir}wav_clips/sample_clip_2.wav"
    ]
    predictions = predict(model, processor, sample_audio_paths, device)
    print("Sample predictions:", predictions)

if __name__ == "__main__":
    main()
