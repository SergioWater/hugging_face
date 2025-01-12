# [[ main.py ]]

# If you want to reduce logs from transformers/huggingface_hub, you can optionally do:
# from transformers import logging
# logging.set_verbosity_error()

from modules.data_preprocessing_pandas import load_data_with_pandas
from modules.model_definition import get_model_and_processor
from modules.model_training import train_model
from modules.model_inference import predict

def main():
    print("========== [DEBUG] ENTER main() ==========")

    # 1. Load the dataset via pandas
    data_dir = "/Users/water/Documents/Coding/Ai/hugging_face/Hugging_Face/data/cleaned"
    print(f"========== [DEBUG] data_dir = {data_dir} ==========")

    dataset = load_data_with_pandas(data_dir)
    print("========== [DEBUG] Dataset loaded successfully via load_data_with_pandas ==========")

    # We'll just split out the "train" portion for training right now
    train_dataset = dataset["train"]
    dev_dataset   = dataset["dev"]
    test_dataset  = dataset["test"]

    print(f"========== [DEBUG] train_dataset size: {len(train_dataset)} ==========")
    print(f"========== [DEBUG] dev_dataset size:   {len(dev_dataset)} ==========")
    print(f"========== [DEBUG] test_dataset size:  {len(test_dataset)} ==========")

    # 2. Get the model + processor
    model, processor = get_model_and_processor("facebook/wav2vec2-base-960h")
    print("========== [DEBUG] model & processor loaded ==========")

    # 3. Train
    print("========== [DEBUG] About to call train_model with epochs=2, batch_size=1 ==========")
    model = train_model(model, processor, train_dataset, epochs=2, batch_size=1)
    print("========== [DEBUG] Training completed ==========")

    # 4. Inference
    sample_audio_paths = [
        f"{data_dir}/clips_data/sample_clip_1.wav",
        f"{data_dir}/clips_data/sample_clip_2.wav"
    ]
    print(f"========== [DEBUG] sample_audio_paths = {sample_audio_paths} ==========")

    predictions = predict(model, processor, sample_audio_paths)
    print("========== [DEBUG] Inference completed. Predictions: ==========")
    print(predictions)

    print("========== [DEBUG] EXIT main() ==========")

if __name__ == "__main__":
    main()
