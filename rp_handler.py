import runpod
import torch
import torchaudio
import os
import tempfile
import base64
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = None
SAMPLE_RATE = 24000


def initialize_model():
    global model
    if model is not None:
        print("Model already initialized")
        return model
    print("Initializing Chatterbox Multilingual V3 model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    print(f"Model initialized. Sample rate: {model.sr}")


def audio_tensor_to_base64(audio_tensor, sample_rate):
    """Convert audio tensor to base64 encoded WAV data."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            torchaudio.save(tmp_file.name, audio_tensor, sample_rate)
            with open(tmp_file.name, 'rb') as audio_file:
                audio_data = audio_file.read()
            os.unlink(tmp_file.name)
            return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        print(f"Error converting audio to base64: {e}")
        raise


def handler(event):
    input_data = event['input']

    prompt = input_data.get('prompt')
    audio_base64 = input_data.get('audio_base64')
    language_id = input_data.get('language_id', 'es')  # default Spanish
    cfg_weight = input_data.get('cfg_weight', 0.0)      # 0.0 = no accent bleed
    exaggeration = input_data.get('exaggeration', 0.5)   # 0.5 = neutral, higher = more expressive

    print(f"New request. Prompt length: {len(prompt)}, language: {language_id}")

    if not prompt:
        return {"error": "prompt is required"}
    if not audio_base64:
        return {"error": "audio_base64 is required"}

    # Decode reference audio from base64
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        return {"error": f"Invalid base64 audio: {e}"}

    # Save reference audio to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as ref_file:
        ref_file.write(audio_bytes)
        ref_path = ref_file.name

    print(f"Reference audio saved: {ref_path} ({len(audio_bytes)} bytes)")

    try:
        # Generate speech with cloned voice
        audio_tensor = model.generate(
            prompt,
            audio_prompt_path=ref_path,
            language_id=language_id,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
        )

        # Save output
        output_filename = "output.wav"
        torchaudio.save(output_filename, audio_tensor, model.sr)

        # Convert to base64
        audio_base64_out = audio_tensor_to_base64(audio_tensor, model.sr)

        response = {
            "status": "success",
            "audio_base64": audio_base64_out,
            "metadata": {
                "sample_rate": model.sr,
                "audio_shape": list(audio_tensor.shape),
                "language_id": language_id,
                "cfg_weight": cfg_weight,
                "exaggeration": exaggeration,
            }
        }

        # Cleanup
        os.remove(output_filename)
        os.remove(ref_path)

        return response

    except Exception as e:
        print(f"Generation error: {e}")
        # Cleanup on error
        if os.path.exists(ref_path):
            os.remove(ref_path)
        return {"error": str(e)}


if __name__ == '__main__':
    initialize_model()
    runpod.serverless.start({'handler': handler})
