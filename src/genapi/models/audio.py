from transformers import AutoProcessor, AutoModel
from genapi.schemas import VoicePresets
from typing import TypeVar
import numpy as np

BarkProcessor = TypeVar("BarkProcessor")
BarkModel = TypeVar("BarkModel")


def load_audio_model() -> tuple[BarkProcessor, BarkModel]:
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = AutoModel.from_pretrained("suno/bark-small")
    return processor, model


def generate_audio(
    processor: BarkProcessor,
    model: BarkModel,
    prompt: str,
    preset: VoicePresets,
) -> tuple[np.array, int]:
    inputs = processor(text=[prompt], return_tensors="pt", voice_preset=preset)
    output = model.generate(**inputs, do_sample=True).cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    return output, sample_rate


if __name__ == "__main__":
    processor, model = load_audio_model()
    prompt = "Hello, world!"
    preset = "v2/en_speaker_1"
    output, sample_rate = generate_audio(processor, model, prompt, preset)
    print(output.shape, sample_rate)