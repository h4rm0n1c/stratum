#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

image="${STRATUM_IMAGE:-stratum:latest}"
gpu="${STRATUM_GPU:-all}"
cuda_visible_devices="${STRATUM_CUDA_VISIBLE_DEVICES:-}"
docker_memory="${STRATUM_DOCKER_MEMORY:-88g}"
docker_memory_swap="${STRATUM_DOCKER_MEMORY_SWAP:-88g}"
cache_dir="${STRATUM_CACHE_DIR:-$repo_root/cache}"
out_dir="${STRATUM_OUT_DIR:-$repo_root/out}"
data_dir="${STRATUM_DATA_DIR:-$repo_root/data}"

mkdir -p \
  "$cache_dir/home" \
  "$cache_dir/xdg" \
  "$cache_dir/huggingface/hub" \
  "$cache_dir/huggingface/transformers" \
  "$cache_dir/datasets" \
  "$cache_dir/torch" \
  "$cache_dir/torch_extensions" \
  "$cache_dir/triton" \
  "$cache_dir/cuda" \
  "$cache_dir/nf4" \
  "$out_dir"

cmd=("$@")
if [ "${#cmd[@]}" -eq 0 ]; then
  cmd=(bash)
fi

docker_args=(
  run --rm
  --gpus "$gpu"
  --ipc=host
  --memory "$docker_memory"
  --memory-swap "$docker_memory_swap"
  --ulimit memlock=-1:-1
  --cap-add IPC_LOCK
  -v "$repo_root:/workspace/stratum"
  -v "$cache_dir:/workspace/cache"
  -v "$out_dir:/workspace/out"
  -e HOME=/workspace/cache/home
  -e XDG_CACHE_HOME=/workspace/cache/xdg
  -e STRATUM_CACHE=/workspace/cache
  -e HF_HOME=/workspace/cache/huggingface
  -e HUGGINGFACE_HUB_CACHE=/workspace/cache/huggingface/hub
  -e TRANSFORMERS_CACHE=/workspace/cache/huggingface/transformers
  -e HF_DATASETS_CACHE=/workspace/cache/datasets
  -e TORCH_HOME=/workspace/cache/torch
  -e TORCH_EXTENSIONS_DIR=/workspace/cache/torch_extensions
  -e TRITON_CACHE_DIR=/workspace/cache/triton
  -e CUDA_CACHE_PATH=/workspace/cache/cuda
  -e CUDA_CACHE_MAXSIZE=2147483648
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID
  -w /workspace/stratum
)

if [ -n "$cuda_visible_devices" ]; then
  docker_args+=(-e CUDA_VISIBLE_DEVICES="$cuda_visible_devices")
fi

if [ -d "$data_dir" ]; then
  docker_args+=(-v "$data_dir:/workspace/data:ro")
fi

echo "image:     $image"
echo "gpu:       $gpu"
echo "cuda vis:  ${cuda_visible_devices:-container default}"
echo "memory:    $docker_memory swap=$docker_memory_swap"
echo "repo:      $repo_root -> /workspace/stratum"
echo "cache:     $cache_dir -> /workspace/cache"
echo "out:       $out_dir -> /workspace/out"
if [ -d "$data_dir" ]; then
  echo "data:      $data_dir -> /workspace/data:ro"
else
  echo "data:      $data_dir not mounted (directory missing)"
fi
echo

exec docker "${docker_args[@]}" "$image" "${cmd[@]}"
