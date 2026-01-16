#!/usr/bin/env bash
set -euo pipefail

# 1) prova a prendere il path da ldconfig (se presente)
CUDA_SO1="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF; exit}' || true)"

# 2) fallback: cerca in percorsi tipici (più veloce di find su /)
if [[ -z "${CUDA_SO1}" ]]; then
  for base in \
    /usr/local/nvidia/lib64 \
    /usr/local/nvidia/lib \
    /usr/lib/x86_64-linux-gnu \
    /usr/lib/wsl/drivers \
    /usr/lib
  do
    if [[ -d "$base" ]]; then
      CUDA_SO1="$(find "$base" -maxdepth 5 -name 'libcuda.so.1' -print -quit 2>/dev/null || true)"
      [[ -n "${CUDA_SO1}" ]] && break
    fi
  done
fi

if [[ -z "${CUDA_SO1}" ]]; then
  echo "ERROR: libcuda.so.1 not found (searched common locations)." >&2
  echo "Debug: listing /usr/local/nvidia and /usr/lib/wsl/drivers:" >&2
  ls -la /usr/local/nvidia 2>/dev/null || true
  ls -la /usr/lib/wsl/drivers 2>/dev/null || true
  exit 1
fi

CUDA_DIR="$(dirname "${CUDA_SO1}")"
ln -sf "${CUDA_SO1}" "${CUDA_DIR}/libcuda.so"

# IMPORTANT: fa sì che gcc/ld trovino -lcuda anche se non c'è -L su quella dir
export LIBRARY_PATH="${CUDA_DIR}:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_DIR}:${LD_LIBRARY_PATH:-}"

echo "Using CUDA driver lib: ${CUDA_SO1}"
ls -la "${CUDA_DIR}" | grep -E 'libcuda\.so(\.1)?' || true

exec vllm serve Qwen/Qwen3-4B-Thinking-2507 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 38592 \
  --dtype auto \
  --api-key token-finto
