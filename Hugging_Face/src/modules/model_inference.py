# [[ model_inference.py ]]

import torch
import soundfile as sf

def predict(model, processor, audio_paths):
    print("========== [DEBUG] ENTER predict() ==========")
    model.eval()
    print("========== [DEBUG] Model set to eval() mode ==========")
    predictions = []

    for i, path in enumerate(audio_paths):
        print(f"========== [DEBUG] Attempting to load file index={i} at path={path} ==========")
        try:
            speech, sr = sf.read(path)
            print(f"========== [DEBUG] Loaded audio: sr={sr}, len(speech)={len(speech)} ==========")
        except Exception as e:
            print(f"========== [ERROR] Could not load audio from {path}. Exception: {e} ==========")
            predictions.append("[Error Loading Audio]")
            continue

        print("========== [DEBUG] Running processor on loaded audio ==========")
        inputs = processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        print(f"========== [DEBUG] Forward pass for file index={i} ==========")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        print(f"========== [DEBUG] Transcription for file index={i}: {transcription} ==========")
        predictions.append(transcription)

    print("========== [DEBUG] EXIT predict() ==========")
    return predictions
