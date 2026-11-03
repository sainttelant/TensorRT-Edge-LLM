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
BUILD_ONLY=0
CHECK_ONLY=0

usage() {
    cat <<'EOF'
Usage: build_orin.bash [--check-only] [--buildonly | --inference] [--help]

Reuse existing engines and build only missing components.
With --inference, run action inference after preparing missing artifacts.
Set MODEL_ROOT, ONNX_DIR, ENGINE_DIR, BUILD_DIR,
TRT_PACKAGE_DIR, CUDA_DIR, or BUILD_JOBS to override the defaults.

Options:
  --check-only  Check prerequisites for the selected mode without running.
  --buildonly   Prepare engines without inference (the default).
                Reuse existing engines and build only missing components.
  --inference   Reuse existing engines, build missing ones, then infer. Requires
                INPUT_FILE (defaults to MODEL_ROOT/input_action.json).
                MODEL_DIR is required only for checkpoint-backed weights.
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
        --buildonly)
            BUILD_ONLY=1
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

((BUILD_ONLY && RUN_INFERENCE)) && fail "--buildonly and --inference cannot be used together."

[[ "$(uname -m)" == "aarch64" ]] || fail "This script must run on an Orin (aarch64) device."
command -v tee >/dev/null 2>&1 || fail "tee is required."

MISSING_COMPONENTS=()
BUILD_TARGETS=()
if [[ ! -s "${BUILD_DIR}/libNvInfer_edgellm_plugin.so" ]]; then
    BUILD_TARGETS+=(NvInfer_edgellm_plugin)
fi
if [[ ! -x "${BUILD_DIR}/examples/multimodal/action_inference" ]]; then
    BUILD_TARGETS+=(action_inference)
fi
for component in llm visual action; do
    if [[ -s "${ENGINE_DIR}/${component}/${component}.engine" \
        && -s "${ENGINE_DIR}/${component}/config.json" ]]; then
        printf 'Reusing existing %s engine.\n' "$component"
        continue
    fi
    printf 'Missing %s engine/config; scheduling component build.\n' "$component"
    MISSING_COMPONENTS+=("$component")
    component_dir="${ONNX_DIR}/${component}"
    [[ -s "${component_dir}/model.onnx" ]] || fail "Missing ONNX graph: ${component_dir}/model.onnx"
    [[ -s "${component_dir}/model.onnx.data" ]] || fail "Missing ONNX external weights: ${component_dir}/model.onnx.data"
    [[ -s "${component_dir}/config.json" ]] || fail "Missing component config: ${component_dir}/config.json"
    if [[ "$component" == "llm" ]]; then
        builder="${BUILD_DIR}/examples/llm/llm_build"
    else
        builder="${BUILD_DIR}/examples/multimodal/${component}_build"
    fi
    if [[ ! -x "$builder" ]]; then
        BUILD_TARGETS+=("${component}_build")
    fi
done

if ((${#BUILD_TARGETS[@]})); then
    [[ -d "$CUDA_DIR" ]] || fail "CUDA toolkit directory not found: $CUDA_DIR"
    [[ -x "${CUDA_DIR}/bin/nvcc" ]] || fail "nvcc not found under CUDA_DIR: $CUDA_DIR"
    CUDA_CTK_VERSION="$("${CUDA_DIR}/bin/nvcc" --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p')"
    [[ -n "$CUDA_CTK_VERSION" ]] || fail "Could not determine CUDA toolkit version from nvcc."
    command -v cmake >/dev/null 2>&1 || fail "cmake is required."
    [[ -f "${TRT_PACKAGE_DIR}/include/aarch64-linux-gnu/NvInfer.h" \
        || -f "${TRT_PACKAGE_DIR}/include/NvInfer.h" ]] \
        || fail "TensorRT headers not found under TRT_PACKAGE_DIR=$TRT_PACKAGE_DIR."

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
fi

if ((RUN_INFERENCE)); then
    [[ -s "$INPUT_FILE" ]] || fail "Inference input JSON not found: $INPUT_FILE"
    python3 - "$ENGINE_DIR" "$ONNX_DIR" "$MODEL_DIR" "${MISSING_COMPONENTS[@]}" <<'PY'
import json
import pathlib
import sys

engine_dir = pathlib.Path(sys.argv[1])
onnx_dir = pathlib.Path(sys.argv[2])
checkpoint_dir = pathlib.Path(sys.argv[3])
missing_components = sys.argv[4:]
for component in ("llm", "visual", "action"):
    config_root = onnx_dir if component in missing_components else engine_dir
    config_path = config_root / component / "config.json"
    with config_path.open() as stream:
        config = json.load(stream)
    bindings = config.get("checkpoint_weight_bindings", [])
    if not isinstance(bindings, list):
        sys.exit(f"ERROR: checkpoint_weight_bindings must be an array: {config_path}")
    if bindings and not (checkpoint_dir / "config.json").is_file():
        sys.exit(
            f"ERROR: {component} requires checkpoint-backed weights. "
            f"Set MODEL_DIR to the original checkpoint directory: {checkpoint_dir}"
        )
    print(f"{component}: {'checkpoint-backed' if bindings else 'embedded/sidecar'} weights")
PY
fi

printf 'Platform: %s\nTensorRT root: %s\n' "$(uname -m)" "$TRT_PACKAGE_DIR"
printf 'Build directory: %s\nModel root: %s\nEngine directory: %s\n' \
    "$BUILD_DIR" "$MODEL_ROOT" "$ENGINE_DIR"
if ((${#BUILD_TARGETS[@]})); then
    printf 'CUDA: %s (%s); SM87; build jobs: %s\n' "$CUDA_DIR" "$CUDA_CTK_VERSION" "$BUILD_JOBS"
fi

if ((CHECK_ONLY)); then
    printf 'Preflight checks passed.\n'
    exit 0
fi

export LD_LIBRARY_PATH="${TRT_PACKAGE_DIR}/lib:${TRT_PACKAGE_DIR}/lib64:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset CUDA_VISIBLE_DEVICES
mkdir -p "${MODEL_ROOT}/logs"
export EDGELLM_PLUGIN_PATH="${BUILD_DIR}/libNvInfer_edgellm_plugin.so"

run_inference() {
    printf 'Running inference using prepared engines.\n'
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
    printf 'Orin inference completed successfully.\n'
}

if ((${#BUILD_TARGETS[@]})); then
    printf 'Compiling missing program targets: %s\n' "${BUILD_TARGETS[*]}"
    cmake -S "$REPO_DIR" -B "$BUILD_DIR" \
        -DTRT_PACKAGE_DIR="$TRT_PACKAGE_DIR" \
        -DCUDA_DIR="$CUDA_DIR" \
        -DCUDA_CTK_VERSION="$CUDA_CTK_VERSION" \
        -DEMBEDDED_TARGET=jetson-orin \
        -DCMAKE_CUDA_COMPILER="${CUDA_DIR}/bin/nvcc" \
        -DCMAKE_CUDA_ARCHITECTURES=87

    cmake --build "$BUILD_DIR" \
        --target "${BUILD_TARGETS[@]}" \
        --parallel "$BUILD_JOBS"
fi

for component in "${MISSING_COMPONENTS[@]}"; do
    case "$component" in
        llm)
            "${BUILD_DIR}/examples/llm/llm_build" \
                --onnxDir "${ONNX_DIR}/llm" \
                --engineDir "${ENGINE_DIR}/llm" \
                --maxInputLen 3424 \
                --maxKVCacheCapacity 4096 \
                --maxBatchSize 1 \
                2>&1 | tee "${MODEL_ROOT}/logs/build-llm.log"
            ;;
        visual)
            "${BUILD_DIR}/examples/multimodal/visual_build" \
                --onnxDir "${ONNX_DIR}/visual" \
                --engineDir "$ENGINE_DIR" \
                --minImageTokens 160 \
                --maxImageTokens 18432 \
                --maxImageTokensPerImage 192 \
                2>&1 | tee "${MODEL_ROOT}/logs/build-visual.log"
            ;;
        action)
            "${BUILD_DIR}/examples/multimodal/action_build" \
                --onnxDir "${ONNX_DIR}/action" \
                --engineDir "$ENGINE_DIR" \
                --maxBatchSize 1 \
                2>&1 | tee "${MODEL_ROOT}/logs/build-action.log"
            ;;
    esac
    [[ -s "${ENGINE_DIR}/${component}/${component}.engine" ]] || fail "Builder did not produce ${component} engine."
    [[ -s "${ENGINE_DIR}/${component}/config.json" ]] || fail "Builder did not produce ${component} config.json."
done

if ((RUN_INFERENCE)); then
    run_inference
else
    printf 'Orin engines are ready (existing components reused).\n'
fi
