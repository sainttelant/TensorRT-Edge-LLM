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
LLM_ENGINE_DIR="${LLM_ENGINE_DIR:-${MODEL_ROOT}/engines/llm}"
MULTIMODAL_ENGINE_DIR="${MULTIMODAL_ENGINE_DIR:-${MODEL_ROOT}/engines}"
CAMERA_CONFIG="${CAMERA_CONFIG:-${REPO_DIR}/examples/configs/camera_video_5e9a00f1.json}"
TRAJECTORY_TEMPLATE="${TRAJECTORY_TEMPLATE:-${REPO_DIR}/VLA_tests/selected/26f8dc7e-a9a2-4302-b847-af2009abf677/input_action.json}"
TRT_PACKAGE_DIR="${TRT_PACKAGE_DIR:-/usr}"
MAX_WINDOWS=1
SHOW_ARGS=()
PREVIEW_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./run_alpaway_smoke.bash [--show] [--max-windows N]
       [--preview-speed-mps VALUE --preview-dt SECONDS] [--preview-calibration PATH] [--help]

Run local video smoke inference with an automatically refreshed EXAMPLE history.
This script is NOT for live cameras or vehicle control.
Defaults: clip 5e9a00f1-f575-43c8-a4e2-62e52a651dbb, FP16 LLM, one window.

Environment overrides:
  CAMERA_CONFIG           Four file-based video sources (paths relative to repo).
  TRAJECTORY_TEMPLATE     Offline input JSON containing a 16-point history.
  MODEL_ROOT              Model workspace root.
  LLM_ENGINE_DIR          FP16 LLM engine directory.
  MULTIMODAL_ENGINE_DIR   Directory containing visual/ and action/ engines.
  BUILD_DIR               Build directory with bin/alpaway_real and lib/libNvInfer_edgellm_plugin.so.
  TRT_PACKAGE_DIR         TensorRT package root (default: /usr).
  EDGELLM_PLUGIN_PATH      Override the plugin library path.
EOF
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --show) SHOW_ARGS=(--show) ;;
        --preview-speed-mps|--preview-dt|--preview-calibration)
            (($# >= 2)) || fail "$1 requires a value."
            PREVIEW_ARGS+=("$1" "$2")
            shift
            ;;
        --max-windows)
            (($# >= 2)) || fail "--max-windows requires a positive integer."
            [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "--max-windows requires a positive integer."
            MAX_WINDOWS="$2"
            shift
            ;;
        *) fail "Unknown option: $1 (use --help)." ;;
    esac
    shift
done

cd "$REPO_DIR"
command -v python3 >/dev/null || fail "python3 is required."
ALPAWAY_BIN="${BUILD_DIR}/bin/alpaway_real"
[[ -x "$ALPAWAY_BIN" ]] || fail "Build alpaway_real first: $ALPAWAY_BIN"
export EDGELLM_PLUGIN_PATH="${EDGELLM_PLUGIN_PATH:-${BUILD_DIR}/lib/libNvInfer_edgellm_plugin.so}"
for file in "$CAMERA_CONFIG" "$TRAJECTORY_TEMPLATE" "$EDGELLM_PLUGIN_PATH" \
    "${LLM_ENGINE_DIR}/llm.engine" "${LLM_ENGINE_DIR}/config.json" \
    "${MULTIMODAL_ENGINE_DIR}/visual/visual.engine" "${MULTIMODAL_ENGINE_DIR}/visual/config.json" \
    "${MULTIMODAL_ENGINE_DIR}/action/action.engine" "${MULTIMODAL_ENGINE_DIR}/action/config.json"; do
    [[ -s "$file" ]] || fail "Missing or empty file: $file"
done
export LD_LIBRARY_PATH="${TRT_PACKAGE_DIR}/lib:${TRT_PACKAGE_DIR}/lib64:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

TEMP_DIR="$(mktemp -d)"
PRODUCER_PID=""
INFERENCE_PID=""
cleanup() {
    for pid in "$INFERENCE_PID" "$PRODUCER_PID"; do
        if [[ -n "$pid" ]]; then
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
            wait "$pid" 2>/dev/null || true
        fi
    done
    rm -f -- "${TEMP_DIR}/trajectory.json" "${TEMP_DIR}/trajectory.tmp"
    rmdir -- "$TEMP_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'WARNING: Replaying EXAMPLE trajectory, not measured video motion. Smoke test only.\n' >&2
printf 'Camera config: %s\nLLM engine: %s\n' "$CAMERA_CONFIG" "$LLM_ENGINE_DIR" >&2
python3 - "$TRAJECTORY_TEMPLATE" "$TEMP_DIR" "$CAMERA_CONFIG" <<'PY' &
import json
import math
import os
import pathlib
import sys
import time

with open(sys.argv[3]) as stream:
    cameras = json.load(stream)["stream_cameras"]
roles = ["front_wide", "front_tele", "cross_left", "cross_right"]
if [camera["role"] for camera in cameras] != roles:
    raise ValueError("Camera config must contain four cameras in model order")
for camera in cameras:
    if camera["type"] != "file":
        raise ValueError("Smoke trajectory replay is allowed only with video files")
    if not pathlib.Path(camera["source"]).is_file():
        raise FileNotFoundError(camera["source"])
with open(sys.argv[1]) as stream:
    request = json.load(stream)["requests"][0]
histories = [
    item["trajectory"]
    for message in request["messages"]
    if isinstance(message["content"], list)
    for item in message["content"]
    if item["type"] == "trajectory"
]
if len(histories) != 1 or len(histories[0]) != 16:
    raise ValueError("Template must contain exactly one 16-point trajectory")
points = histories[0]
for point in points:
    if not isinstance(point, list) or len(point) != 3:
        raise ValueError("Each trajectory point must be [x, y, z]")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in point):
        raise ValueError("Trajectory coordinates must be finite numbers")
directory = pathlib.Path(sys.argv[2])
while True:
    with (directory / "trajectory.tmp").open("w") as stream:
        json.dump({"timestamp_ms": time.time_ns() // 1_000_000, "trajectory": points}, stream)
    os.replace(directory / "trajectory.tmp", directory / "trajectory.json")
    time.sleep(0.1)
PY
PRODUCER_PID=$!

for ((attempt = 0; attempt < 100; ++attempt)); do
    kill -0 "$PRODUCER_PID" 2>/dev/null || fail "Example trajectory producer failed."
    [[ -s "${TEMP_DIR}/trajectory.json" ]] && break
    sleep 0.05
done
[[ -s "${TEMP_DIR}/trajectory.json" ]] || fail "Timed out waiting for the example trajectory."

"$ALPAWAY_BIN" \
    --camera-config "$CAMERA_CONFIG" \
    --trajectory-file "${TEMP_DIR}/trajectory.json" \
    --engine-dir "$LLM_ENGINE_DIR" \
    --multimodal-engine-dir "$MULTIMODAL_ENGINE_DIR" \
    --max-windows "$MAX_WINDOWS" \
    "${SHOW_ARGS[@]}" "${PREVIEW_ARGS[@]}" &
INFERENCE_PID=$!
while kill -0 "$INFERENCE_PID" 2>/dev/null; do
    kill -0 "$PRODUCER_PID" 2>/dev/null || fail "Example trajectory producer stopped during inference."
    sleep 0.1
done
wait "$INFERENCE_PID"
