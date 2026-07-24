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

# Start the container (model downloads on first worker boot)

# Start the container
CMD ["python3", "-u", "rp_handler.py"]
