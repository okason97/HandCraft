FROM ubuntu:22.04

# libraries needed by opencv (keypoint extraction) and git/wget for the dataset scripts
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git wget unzip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv manages Python and the dependencies (see pyproject.toml and uv.lock)
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python

# Create a working directory
WORKDIR /HandCraft

# Install the locked dependencies. The torch wheels bundle the CUDA libraries,
# so the container only needs the NVIDIA runtime (docker run --gpus all).
COPY pyproject.toml uv.lock README.md LICENSE.txt ./
RUN uv sync --frozen --no-dev --all-extras

# Copy the current directory contents into the container
COPY . /HandCraft
ENV PATH="/HandCraft/.venv/bin:$PATH"

# Datasets and outputs are expected to be mounted at /data and /outputs:
# docker run --gpus all -v <datasets>:/data -v <outputs>:/outputs -it handcraft
ENV HANDCRAFT_DATA=/data HANDCRAFT_SAVE=/outputs

# wandb login
