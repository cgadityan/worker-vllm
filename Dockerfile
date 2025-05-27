FROM nvidia/cuda:12.1.0-base-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y python3-pip \
    && apt-get install -y git-lfs \
    && git lfs install 

RUN ldconfig /usr/local/cuda-12.1/compat/

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

# Install vLLM (switching back to pip installs since issues that required building fork are fixed and space optimization is not as important since caching) and FlashInfer 
RUN python3 -m pip install vllm==0.8.5 && \
    python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3

# Set up the working directory for the worker-vllm application
WORKDIR /worker-vllm

# Copy the application code into the container
COPY . /worker-vllm

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_1_NAME=""
ARG MODEL_2_NAME=""
ARG MODEL_3_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 

ENV PYTHONPATH="/:/vllm-workspace"

# Use the RunPod vLLM base image with CUDA 12.x (for H100 support)
FROM runpod/worker-v1-vllm:v2.5.0stable-cuda12.1.0 AS base

# Preload Model 1: Qwen/QwQ-32B
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='Qwen/QwQ-32B', local_dir='/models/QwQ-32B', local_dir_use_symlinks=False)"

# Preload Model 2: Nvidia Llama-3_1-Nemotron-Ultra-253B-v1
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='nvidia/Llama-3_1-Nemotron-Ultra-253B-v1', local_dir='/models/Nemo-253B', local_dir_use_symlinks=False)"

# Preload Model 3: DeepSeek-R1
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='deepseek-ai/DeepSeek-R1', local_dir='/models/DeepSeek-R1', local_dir_use_symlinks=False)"


COPY src /src
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["python3", "/src/handler.py"]
