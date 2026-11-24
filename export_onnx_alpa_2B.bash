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
Usage: bash export_onnx_alpa_2B.bash [--help]

Export the local Alpamayo2-Edge checkpoint for Jetson Orin (SM87). The export
uses FP16 model compute and KV cache, the full vocabulary and greedy decoding.
The action graph retains FP32 operations where the recipe requires them.
No calibration data, FP8 quantization, or DFlash draft is required.

First obtain the public NVlabs/alpamayo-recipes repository, then set RECIPE_DIR
to its recipes/alpamayo2edge_edgellm directory. Install the dependencies of
this TensorRT Edge-LLM checkout in the active Python environment.

Environment variables:
  RECIPE_DIR   Recipe directory (default: ./alpamayo-recipes/recipes/
               alpamayo2edge_edgellm, relative to this script)
  MODEL_DIR    Release checkpoint (default: ./R1_2B, relative to this script)
  OUTPUT_ROOT  Output parent (default: ./tensorrt-edgellm-workspace/R1_2B)
  PYTHON       Python executable in the Edge-LLM environment (default: python3)
  CPU_THREADS  Export CPU threads (default: 8)

Relative paths are resolved relative to this script.

Each run creates a fresh directory under OUTPUT_ROOT containing shim/, onnx/
(llm/, visual/, action/) and export.log. The source checkpoint is not changed.
The action graph uses five Euler steps, six candidates and KV capacity 3584.
This only exports ONNX; it does not build engines or validate inference.
The upstream recipe was tested on DRIVE Thor, not Jetson Orin.
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
RECIPE_DIR="${RECIPE_DIR:-${REPO_DIR}/alpamayo-recipes/recipes/alpamayo2edge_edgellm}"
MODEL_DIR="${MODEL_DIR:-${REPO_DIR}/R1_2B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/tensorrt-edgellm-workspace/R1_2B}"
PYTHON="${PYTHON:-python3}"
CPU_THREADS="${CPU_THREADS:-8}"

cd -- "$REPO_DIR"
[[ "$CPU_THREADS" =~ ^[1-9][0-9]*$ ]] || fail "CPU_THREADS must be a positive integer."
command -v "$PYTHON" >/dev/null 2>&1 || fail "Python executable not found: $PYTHON"
for file in shim.py action_graph.py onnx_graph.py; do
    [[ -s "${RECIPE_DIR}/host/${file}" ]] || fail "Missing recipe file: ${RECIPE_DIR}/host/${file}"
done
[[ -s "${MODEL_DIR}/config.json" ]] || fail "Missing checkpoint config: ${MODEL_DIR}/config.json"
[[ -s "${MODEL_DIR}/model.safetensors.index.json" ]] \
    || fail "Missing checkpoint index: ${MODEL_DIR}/model.safetensors.index.json"

"$PYTHON" - "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

model = Path(sys.argv[1])
with (model / "config.json").open() as f:
    config = json.load(f)
if config.get("model_type") != "alpamayo2_edge":
    sys.exit("ERROR: checkpoint model_type must be alpamayo2_edge.")
with (model / "model.safetensors.index.json").open() as f:
    index = json.load(f)
shards = set(index["weight_map"].values())
if not shards:
    sys.exit("ERROR: checkpoint index contains no weight shards.")
for shard in sorted(shards):
    path = model / shard
    if not path.is_file() or path.stat().st_size == 0:
        sys.exit(f"ERROR: missing or empty checkpoint shard: {path}")
PY
"$PYTHON" -c 'import numpy, onnx, safetensors, torch; from tensorrt_edgellm.onnx.cli import main' \
    || fail "Install the Edge-LLM ONNX export dependencies in the selected Python environment."

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p -- "$OUTPUT_ROOT"
RUN_DIR="$(mktemp -d "${OUTPUT_ROOT}/export-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
LOG_FILE="${RUN_DIR}/export.log"
SHIM_DIR="${RUN_DIR}/shim"
ONNX_DIR="${RUN_DIR}/onnx"

printf 'Checkpoint: %s\nRecipe: %s\nOutput: %s\nLog: %s\n' \
    "$MODEL_DIR" "$RECIPE_DIR" "$RUN_DIR" "$LOG_FILE" | tee -a "$LOG_FILE"
printf '\n[1/3] Converting checkpoint layout\n' | tee -a "$LOG_FILE"
"$PYTHON" "${RECIPE_DIR}/host/shim.py" "$MODEL_DIR" "$SHIM_DIR" 2>&1 | tee -a "$LOG_FILE"

printf '\n[2/3] Exporting FP16 language and vision graphs for SM87\n' | tee -a "$LOG_FILE"
"$PYTHON" -m tensorrt_edgellm.onnx.cli "$SHIM_DIR" "$ONNX_DIR" \
    --task reasoning --dtype float16 --target-sm 87 2>&1 | tee -a "$LOG_FILE"

printf '\n[3/3] Exporting FP16 action graph and FP16 KV cache\n' | tee -a "$LOG_FILE"
"$PYTHON" "${RECIPE_DIR}/host/action_graph.py" "$SHIM_DIR" "${ONNX_DIR}/action" 5 3584 \
    2>&1 | tee -a "$LOG_FILE"

for component in llm visual action; do
    [[ -s "${ONNX_DIR}/${component}/model.onnx" ]] \
        || fail "Missing ${component} ONNX graph; see ${LOG_FILE}"
done
printf '\nONNX export complete: %s\n' "$ONNX_DIR" | tee -a "$LOG_FILE"
printf 'Engine build and inference on Jetson Orin are still required.\n' | tee -a "$LOG_FILE"
