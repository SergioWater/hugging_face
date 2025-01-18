"""
Performs inference/prediction on a list of audio files using a trained model & processor.
"""

import torch
import soundfile as sf

def predict(model, processor, audio_paths, device):
    """
    Runs inference on a list of audio_paths.
    Returns a list of transcriptions.
    """
    model.eval()
    predictions = []

    for path in audio_paths:
        try:
            speech, sr = sf.read(path)
        except Exception:
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
        transcription = processor.decode(predicted_ids[0])
        predictions.append(transcription)

    return predictions