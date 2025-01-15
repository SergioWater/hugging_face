# [[ model_definition.py ]]

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def get_model_and_processor(model_name: str = "facebook/wav2vec2-base-960h"):

    processor = Wav2Vec2Processor.from_pretrained(model_name, force_download=True)
    model = Wav2Vec2ForCTC.from_pretrained(model_name, force_download=True)

    return model, processor
