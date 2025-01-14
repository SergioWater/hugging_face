import sys
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
    model = train_model(model, processor, train_dataset, epochs=5, batch_size=5)

    sample_audio_paths = [
        f"{data_dir}wav_clips/sample_clip_1.wav",
        f"{data_dir}wav_clips/sample_clip_2.wav"
    ]

    predictions = predict(model, processor, sample_audio_paths)
    # print(predictions)

if __name__ == "__main__":
    main()
