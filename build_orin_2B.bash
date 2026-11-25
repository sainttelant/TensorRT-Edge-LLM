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
export MODEL_ROOT="${MODEL_ROOT:-${REPO_DIR}/tensorrt-edgellm-workspace/R1_2B}"
export ACTION_WEIGHTS_FILE="${ACTION_WEIGHTS_FILE:-weights.bin}"

for arg in "$@"; do
    if [[ "$arg" == "--inference" ]]; then
        printf 'ERROR: Alpamayo2-Edge action inference requires its tree-action runtime; action_inference is incompatible.\n' >&2
        exit 1
    fi
done

# Share build options and inference behavior with the original Orin entry point.
exec bash "${REPO_DIR}/build_orin.bash" "$@"
