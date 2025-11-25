FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

  WORKDIR /app

  # Install system dependencies
  RUN apt-get update && apt-get install -y \
      git \
      g++ \
      libgl1-mesa-glx \
      libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/*

  # Clone Point-SAM
  RUN git clone --recursive https://github.com/zyc00/Point-SAM.git /app/point-sam

  WORKDIR /app/point-sam

  # Install Python dependencies
  RUN pip install --no-cache-dir \
      timm>=0.9.0 \
      safetensors \
      einops \
      huggingface_hub \
      runpod

  # Install torkit3d (Point-SAM dependency)
  RUN cd third_party/torkit3d && FORCE_CUDA=1 pip install .

  # Download model weights from HuggingFace
  RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('yuchen0187/Point-SAM', 'point_sam_vit_l.safetensors', local_dir='/app/weights')"

  # Copy handler
  COPY handler.py /app/handler.py

  WORKDIR /app

  CMD ["python", "-u", "handler.py"]
