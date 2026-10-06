#!/usr/bin/env bash
# One-time host setup for Ubuntu/Debian-like Linux with an NVIDIA GPU.
# Installs Docker Engine + Compose v2 + NVIDIA Container Toolkit and verifies GPU passthrough.
set -euo pipefail

if ! command -v docker >/dev/null; then
  echo ">> Installing Docker Engine"
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER" && echo "   (log out/in for the docker group to apply)"
fi

if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
  echo ">> Installing NVIDIA Container Toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

echo ">> GPU check inside a container:"
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
