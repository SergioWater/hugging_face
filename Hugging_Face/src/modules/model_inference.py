# [[ model_inference.py ]]

import torch

def predict(model, processor, audio_paths):
    """
    audio_paths: A list of audio file paths.
    """
    model.eval()
    predictions = []

    for path in audio_paths:
        # 1. Load the raw audio array and sample rate
        speech, sr = processor.feature_extractor.read_audio(path, sampling_rate=16000)

        # 2. Process the audio
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)

        # 3. Model forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # 4. Greedy decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        predictions.append(transcription)

    return predictions
