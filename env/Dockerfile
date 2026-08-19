# Pinned, reproducible serving environment. Fill the base image + wheel index
# to match your working cu128 stack. This is a skeleton, not a validated build.
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY env/requirements.lock /app/requirements.lock

# Install torch cu128 + vllm from your validated indices, then the rest.
# RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch==<pin>
# RUN pip install vllm==<pin> flashinfer==<pin>
RUN pip install -r requirements.lock

COPY . /app
CMD ["bash"]
