# [model_inference.py]
import torch
import soundfile as sf

def predict(model, processor, audio_paths, device):
    model.eval()
    predictions = []

    for path in audio_paths:
        try:
            speech, sr = sf.read(path)
        except Exception as e:
            predictions.append("[Error Loading Audio]")
            continue

        inputs = processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predicted_ids = torch.argmax(logits, dim=-1)
        # NO CHANGE: Decode
        transcription = processor.decode(predicted_ids[0])
        predictions.append(transcription)

    return predictions
