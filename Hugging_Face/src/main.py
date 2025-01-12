# [[ main.py ]]

from modules.data_preprocessing import load_data
from modules.model_definition import get_model_and_processor
from modules.model_training import train_model
from modules.model_inference import predict

def main():
    # 1. Load the dataset
    data_dir = "/Users/water/Documents/Coding/Ai/hugging_face/Hugging_Face/data/cleaned"
    dataset = load_data(data_dir)

    # We'll just split out the "train" portion for training right now
    train_dataset = dataset["train"]
    # Optionally, dev and test can be used for validation/inference
    dev_dataset = dataset["dev"]
    test_dataset = dataset["test"]

    # 2. Get the model + processor
    model, processor = get_model_and_processor("facebook/wav2vec2-base-960h")

    # 3. Train
    model = train_model(model, processor, train_dataset, epochs=1, batch_size=4)

    # 4. Inference
    sample_audio_paths = [
        f"{data_dir}/clips/sample_clip_1.wav",
        f"{data_dir}/clips/sample_clip_2.wav"
    ]
    
    predictions = predict(model, processor, sample_audio_paths)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
