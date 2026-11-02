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

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_DIR}/build-orin}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_DIR}/tensorrt-edgellm-workspace/R1_10B}"
ONNX_DIR="${ONNX_DIR:-${MODEL_ROOT}/onnx}"
ENGINE_DIR="${ENGINE_DIR:-${MODEL_ROOT}/engines}"
MODEL_DIR="${MODEL_DIR:-${MODEL_ROOT}/checkpoint}"
TRT_PACKAGE_DIR="${TRT_PACKAGE_DIR:-/usr}"
CUDA_DIR="${CUDA_DIR:-$(readlink -f /usr/local/cuda)}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
INPUT_FILE="${INPUT_FILE:-${MODEL_ROOT}/input_action.json}"
OUTPUT_FILE="${OUTPUT_FILE:-${MODEL_ROOT}/output_action.json}"
RUN_INFERENCE=0
CHECK_ONLY=0

usage() {
    cat <<'EOF'
Usage: build_orin.bash [--check-only] [--inference] [--help]

Configure and compile the Orin SM87 builders, then build LLM, visual, and
action TensorRT engines. Set MODEL_ROOT, ONNX_DIR, ENGINE_DIR, BUILD_DIR,
TRT_PACKAGE_DIR, CUDA_DIR, or BUILD_JOBS to override the defaults.

Options:
  --check-only  Check platform, tools, and ONNX inputs without building.
  --inference   Also run action inference after engine builds. Requires
                MODEL_DIR/config.json and INPUT_FILE (defaults to
                MODEL_ROOT/input_action.json).
  --help        Show this help.
EOF
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

while (($# > 0)); do
    case "$1" in
        --check-only)
            CHECK_ONLY=1
            ;;
        --inference)
            RUN_INFERENCE=1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            usage >&2
            fail "Unknown option: $1"
            ;;
    esac
    shift
done

[[ "$(uname -m)" == "aarch64" ]] || fail "This script must run on an Orin (aarch64) device."
[[ -d "$CUDA_DIR" ]] || fail "CUDA toolkit directory not found: $CUDA_DIR"
[[ -x "${CUDA_DIR}/bin/nvcc" ]] || fail "nvcc not found under CUDA_DIR: $CUDA_DIR"
CUDA_CTK_VERSION="$("${CUDA_DIR}/bin/nvcc" --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p')"
[[ -n "$CUDA_CTK_VERSION" ]] || fail "Could not determine CUDA toolkit version from nvcc."
command -v cmake >/dev/null 2>&1 || fail "cmake is required."
command -v tee >/dev/null 2>&1 || fail "tee is required."
[[ -f "${TRT_PACKAGE_DIR}/include/aarch64-linux-gnu/NvInfer.h" \
    || -f "${TRT_PACKAGE_DIR}/include/NvInfer.h" ]] \
    || fail "TensorRT headers not found under TRT_PACKAGE_DIR=$TRT_PACKAGE_DIR."

for component in llm visual action; do
    component_dir="${ONNX_DIR}/${component}"
    [[ -s "${component_dir}/model.onnx" ]] || fail "Missing ONNX graph: ${component_dir}/model.onnx"
    [[ -s "${component_dir}/model.onnx.data" ]] || fail "Missing ONNX external weights: ${component_dir}/model.onnx.data"
    [[ -s "${component_dir}/config.json" ]] || fail "Missing component config: ${component_dir}/config.json"
done

if [[ ! -f "${REPO_DIR}/3rdParty/nlohmannJson/include/nlohmann/json.hpp" \
    || ! -f "${REPO_DIR}/3rdParty/xgrammar/CMakeLists.txt" \
    || ! -f "${REPO_DIR}/3rdParty/xgrammar/3rdparty/dlpack/include/dlpack/dlpack.h" ]]; then
    fail "Required git submodules are missing. Run: git submodule update --init --recursive"
fi

CUTE_DSL_ARCHIVE="${REPO_DIR}/kernelSrcs/cuteDSLPrebuilt/cutedsl_aarch64_sm_87_cuda${CUDA_CTK_VERSION%%.*}.tar.gz"
CUTE_DSL_LIBRARY="${REPO_DIR}/cpp/kernels/cuteDSLArtifact/aarch64/sm_87/libcutedsl_aarch64.a"
if [[ ! -s "$CUTE_DSL_ARCHIVE" && ! -s "$CUTE_DSL_LIBRARY" ]]; then
    fail "Missing SM87 CuTe DSL artifact for CUDA ${CUDA_CTK_VERSION}. Generate it with kernelSrcs/build_cutedsl.py."
fi

if ((RUN_INFERENCE)); then
    [[ -s "${MODEL_DIR}/config.json" ]] || fail "Inference requires checkpoint config: ${MODEL_DIR}/config.json"
    [[ -s "$INPUT_FILE" ]] || fail "Inference input JSON not found: $INPUT_FILE"
fi

printf 'Platform: %s\nCUDA: %s (%s)\nTensorRT root: %s\n' \
    "$(uname -m)" "$CUDA_DIR" "$CUDA_CTK_VERSION" "$TRT_PACKAGE_DIR"
printf 'Build directory: %s\nModel root: %s\nEngine directory: %s\n' \
    "$BUILD_DIR" "$MODEL_ROOT" "$ENGINE_DIR"
printf 'CUDA architecture: SM87; build jobs: %s\n' "$BUILD_JOBS"

if ((CHECK_ONLY)); then
    printf 'Preflight checks passed.\n'
    exit 0
fi

export LD_LIBRARY_PATH="${TRT_PACKAGE_DIR}/lib:${TRT_PACKAGE_DIR}/lib64:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset CUDA_VISIBLE_DEVICES
mkdir -p "${MODEL_ROOT}/logs"

cmake -S "$REPO_DIR" -B "$BUILD_DIR" \
    -DTRT_PACKAGE_DIR="$TRT_PACKAGE_DIR" \
    -DCUDA_DIR="$CUDA_DIR" \
    -DCUDA_CTK_VERSION="$CUDA_CTK_VERSION" \
    -DEMBEDDED_TARGET=jetson-orin \
    -DCMAKE_CUDA_COMPILER="${CUDA_DIR}/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES=87

BUILD_TARGETS=(NvInfer_edgellm_plugin llm_build visual_build action_build)
if ((RUN_INFERENCE)); then
    BUILD_TARGETS+=(action_inference)
fi

cmake --build "$BUILD_DIR" \
    --target "${BUILD_TARGETS[@]}" \
    --parallel "$BUILD_JOBS"

export EDGELLM_PLUGIN_PATH="${BUILD_DIR}/libNvInfer_edgellm_plugin.so"

"${BUILD_DIR}/examples/llm/llm_build" \
    --onnxDir "${ONNX_DIR}/llm" \
    --engineDir "${ENGINE_DIR}/llm" \
    --maxInputLen 3424 \
    --maxKVCacheCapacity 4096 \
    --maxBatchSize 1 \
    2>&1 | tee "${MODEL_ROOT}/logs/build-llm.log"
[[ -s "${ENGINE_DIR}/llm/llm.engine" ]] || fail "LLM builder did not produce llm.engine."

"${BUILD_DIR}/examples/multimodal/visual_build" \
    --onnxDir "${ONNX_DIR}/visual" \
    --engineDir "$ENGINE_DIR" \
    --minImageTokens 160 \
    --maxImageTokens 18432 \
    --maxImageTokensPerImage 192 \
    2>&1 | tee "${MODEL_ROOT}/logs/build-visual.log"
[[ -s "${ENGINE_DIR}/visual/visual.engine" ]] || fail "Visual builder did not produce visual.engine."

"${BUILD_DIR}/examples/multimodal/action_build" \
    --onnxDir "${ONNX_DIR}/action" \
    --engineDir "$ENGINE_DIR" \
    --maxBatchSize 1 \
    2>&1 | tee "${MODEL_ROOT}/logs/build-action.log"
[[ -s "${ENGINE_DIR}/action/action.engine" ]] || fail "Action builder did not produce action.engine."

if ((RUN_INFERENCE)); then
    "${BUILD_DIR}/examples/multimodal/action_inference" \
        --engineDir "${ENGINE_DIR}/llm" \
        --multimodalEngineDir "$ENGINE_DIR" \
        --checkpointDir "$MODEL_DIR" \
        --inputFile "$INPUT_FILE" \
        --outputFile "$OUTPUT_FILE" \
        --batchSize 1 \
        --noiseSeed 5 \
        --dumpProfile \
        --profileOutputFile "${MODEL_ROOT}/profile_action.json" \
        2>&1 | tee "${MODEL_ROOT}/logs/inference-action.log"
    [[ -s "$OUTPUT_FILE" ]] || fail "Inference completed without a non-empty output: $OUTPUT_FILE"
fi

printf 'Orin engine build completed successfully.\n'
