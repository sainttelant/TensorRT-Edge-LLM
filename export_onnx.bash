#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash export_onnx.bash [--help]

Export the original R1_10B checkpoint to float16 ONNX on CPU.
The checkpoint is read without modification; GPU devices are hidden.

Defaults (relative to this script):
  MODEL_DIR      R1_10B
  ONNX_DIR       tensorrt-edgellm-workspace/R1_10B/onnx
  LOG_DIR        tensorrt-edgellm-workspace/R1_10B/logs
  KV capacity    4096
  CPU threads    8 (OpenMP, MKL, OpenBLAS)

Paths are always resolved relative to this script, not the current directory.
Existing REPO_DIR, WORKSPACE_DIR, MODEL_NAME, MODEL_DIR, ONNX_DIR, and LOG_DIR
environment variables are ignored. No manual exports are required.
Each run writes a unique log and prints its last 25 lines on completion.
This script exports ONNX only; it does not build engines or run inference.
EOF
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

if (($# > 0)); then
    if (($# == 1)) && [[ "$1" == "--help" || "$1" == "-h" ]]; then
        usage
        exit 0
    fi
    usage >&2
    fail "Unsupported arguments: $*"
fi

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${REPO_DIR}/tensorrt-edgellm-workspace"
MODEL_NAME=R1_10B
MODEL_DIR="${REPO_DIR}/${MODEL_NAME}"
ONNX_DIR="${WORKSPACE_DIR}/${MODEL_NAME}/onnx"
LOG_DIR="${WORKSPACE_DIR}/${MODEL_NAME}/logs"

[[ -s "${MODEL_DIR}/config.json" ]] || fail "Missing or empty checkpoint file: ${MODEL_DIR}/config.json"
[[ -s "${MODEL_DIR}/model.safetensors.index.json" ]] \
    || fail "Missing or empty checkpoint file: ${MODEL_DIR}/model.safetensors.index.json"
command -v tensorrt-edgellm-export >/dev/null 2>&1 \
    || fail "tensorrt-edgellm-export is not in PATH. Activate the environment containing TensorRT Edge-LLM."

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p -- "$LOG_DIR"
LOG_FILE="$(mktemp "${LOG_DIR}/export-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")"
printf 'Checkpoint: %s\nONNX output: %s\nLog file: %s\n' "$MODEL_DIR" "$ONNX_DIR" "$LOG_FILE"
printf 'Exporting on CPU (8 threads); this may take a while.\n'

TIMEFORMAT=$'Elapsed: %3R seconds\nUser CPU: %3U seconds\nSystem CPU: %3S seconds'
status=0
{
    time tensorrt-edgellm-export \
        "$MODEL_DIR" \
        "$ONNX_DIR" \
        --dtype float16 \
        --max-kv-cache-capacity 4096
} > "$LOG_FILE" 2>&1 || status=$?

tail -n 25 -- "$LOG_FILE"
printf 'Exporter exit status: %s\n' "$status"
if ((status != 0)); then
    printf 'ERROR: ONNX export failed. Full log: %s\n' "$LOG_FILE" >&2
fi
exit "$status"