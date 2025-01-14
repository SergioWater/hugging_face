# [[ model_definition.py ]]

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def get_model_and_processor(model_name: str = "facebook/wav2vec2-base-960h"):
    print(f"========== [DEBUG] ENTER get_model_and_processor with model_name={model_name} ==========")

    processor = Wav2Vec2Processor.from_pretrained(model_name, force_download=True)
    model = Wav2Vec2ForCTC.from_pretrained(model_name, force_download=True)

    print("========== [DEBUG] Wav2Vec2Processor & Wav2Vec2ForCTC loaded ==========")
    print("========== [DEBUG] EXIT get_model_and_processor ==========")
    return model, processor
