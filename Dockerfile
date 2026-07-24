FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1

# Install chatterbox with multilingual support
RUN pip install --no-cache-dir "chatterbox-tts[multilingual] @ git+https://github.com/resemble-ai/chatterbox.git"

WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY rp_handler.py /

# Pre-download model weights (bakes them into image, no cold-start download)
RUN python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; model = ChatterboxMultilingualTTS.from_pretrained(device='cuda')"

# Start the container
CMD ["python3", "-u", "rp_handler.py"]
