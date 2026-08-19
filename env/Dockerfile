# Pinned, reproducible serving environment for the sm_120 (Blackwell) study.
#
# CUDA 13, not the cu128 this skeleton originally assumed: the validated stack
# is torch 2.13.0+cu130 / vLLM 0.27.1, resolved against driver 580.159.03.
# See env/requirements.lock for all 203 pins.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY env/requirements.lock /app/requirements.lock

# requirements.lock is a complete `pip freeze` of the validated environment and
# already pins torch/vllm/flashinfer/transformers exactly. Nothing else to add:
# a second install line would risk resolving a different torch than the one the
# numbers were produced on.
RUN pip install --break-system-packages -r requirements.lock

COPY . /app

# Attention backend is a controlled variable -- pin it, do not let vLLM choose.
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTN
# DeepGEMM crashes loading block-scaled FP8 weights on sm_120; see LANDMINES.md.
ENV VLLM_USE_DEEP_GEMM=0

CMD ["bash"]
