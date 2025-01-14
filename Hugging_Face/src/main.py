# [HIGHLIGHT: CHANGED LINES] 
# - We no longer load train/dev/test. Instead, we load only validated.tsv as our single dataset.
# - We no longer talk about train/dev/test splits, we just have "train".
# - We renamed the sample inference audio paths to point to "validated_clips" and MP3 files.

from modules.data_preprocessing_pandas import load_data_with_pandas
from modules.model_definition import get_model_and_processor
from modules.model_training import train_model
from modules.model_inference import predict

def main():
    print("========== [DEBUG] ENTER main() ==========")

    # 1. Load the dataset via pandas
    data_dir = "/Users/water/Documents/Coding/Ai/hugging_face/Hugging_Face/data/"
    print(f"========== [DEBUG] data_dir = {data_dir} ==========")

    # [HIGHLIGHT: CHANGED]
    # We only load validated.tsv, which will appear as dataset["train"] from our new code in data_preprocessing_pandas.py
    dataset = load_data_with_pandas(data_dir)
    print("========== [DEBUG] Dataset loaded successfully via load_data_with_pandas ==========")

    # [HIGHLIGHT: CHANGED]
    # Now we only have a "train" dataset (which is validated.tsv). No dev/test.
    train_dataset = dataset["train"]
    print(f"========== [DEBUG] train_dataset size: {len(train_dataset)} ==========")

    # 2. Get the model + processor
    model, processor = get_model_and_processor("facebook/wav2vec2-base-960h")
    print("========== [DEBUG] model & processor loaded ==========")

    # 3. Train
    print("========== [DEBUG] About to call train_model with epochs=2, batch_size=1 ==========")
    model = train_model(model, processor, train_dataset, epochs=2, batch_size=1)
    print("========== [DEBUG] Training completed ==========")

    # 4. Inference
    # [HIGHLIGHT: CHANGED]
    # We renamed these sample paths to mp3 in validated_clips
    sample_audio_paths = [
        f"{data_dir}/validated_clips/sample_clip_1.mp3",
        f"{data_dir}/validated_clips/sample_clip_2.mp3"
    ]
    print(f"========== [DEBUG] sample_audio_paths = {sample_audio_paths} ==========")

    predictions = predict(model, processor, sample_audio_paths)
    print("========== [DEBUG] Inference completed. Predictions: ==========")
    print(predictions)

    print("========== [DEBUG] EXIT main() ==========")

if __name__ == "__main__":
    main()
