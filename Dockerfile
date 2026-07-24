FROM runpod/pytorch:1.0.7-rc.138-cu1300-torch291-ubuntu2404

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
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt
COPY rp_handler.py /

# Start the container (model downloads on first worker boot)
CMD ["python3", "-u", "rp_handler.py"]
