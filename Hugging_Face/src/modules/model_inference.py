# [[ model_inference.py ]]

import torch
import soundfile as sf

def predict(model, processor, audio_paths):
    """
    Predict transcriptions for a list of audio file paths using a trained Wav2Vec2 model.
    
    Args:
        model: The Wav2Vec2ForCTC model (already trained or fine-tuned).
        processor: The Wav2Vec2Processor (includes feature_extractor + tokenizer).
        audio_paths: A list of file paths (e.g., .wav or .mp3).

    Returns:
        A list of strings, each containing the transcribed text for that audio file.
    """
    model.eval()
    predictions = []

    for path in audio_paths:
        # 1. Load the raw audio data and sample rate from disk
        speech, sr = sf.read(path)

        # 2. (Optional) Resample if sr != 16000. For now, assume sr is 16000.
        # You can use librosa or torchaudio for resampling if needed.

        # 3. Run the audio through the processor
        inputs = processor(
            speech, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )

        # 4. Forward pass through the model (inference)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # 5. Greedy decode the output
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        predictions.append(transcription)

    return predictions

