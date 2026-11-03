#!/usr/bin/env python3
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

"""Render VLA camera frames, action curves, and model CoC text as PNGs."""

import argparse
import json
import math
import pathlib
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.image
import matplotlib.pyplot as plt

def image_paths(messages):
    """Return image paths in the request's parser order."""
    return [
        item["image"] for message in messages
        if isinstance(message["content"], list)
        for item in message["content"] if item["type"] == "image"
    ]


def main():
    """Validate response/window correspondence and render every response."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--mapping-file", type=pathlib.Path,
                        help="Default: window_mapping.json beside --output-file.")
    parser.add_argument("--output-dir", type=pathlib.Path,
                        help="Default: visualizations/ beside --output-file.")
    args = parser.parse_args()
    args.output_file = args.output_file.resolve()
    if args.mapping_file is None:
        args.mapping_file = args.output_file.with_name("window_mapping.json")
    if args.output_dir is None:
        args.output_dir = args.output_file.parent / "visualizations"
    if not args.mapping_file.is_file():
        parser.error(f"Window mapping not found: {args.mapping_file}; "
                     "use --mapping-file to specify the matching mapping")
    data = json.loads(args.output_file.read_text(encoding="utf-8"))
    mapping = json.loads(args.mapping_file.read_text(encoding="utf-8"))
    windows = mapping["windows"]
    responses = data["responses"]
    if len(responses) != len(windows):
        parser.error(f"Output has {len(responses)} responses "
                     f"({args.output_file}), but mapping has {len(windows)} "
                     f"windows ({args.mapping_file}); "
                     "use the mapping for this inference input")
    by_index = {response["request_idx"]: response for response in responses}
    if len(by_index) != len(responses):
        parser.error("Duplicate request_idx in inference output")
    for window in windows:
        index = window["request_idx"]
        if index not in by_index:
            parser.error(f"Missing output for request {index}")
        response = by_index[index]
        if image_paths(response["messages"]) != [
                frame["png"] for frame in window["frames"]]:
            parser.error(f"Output images do not match window {index}")
        trajectory = response.get("output_trajectory", [])
        if (len(trajectory) != 64
                or any(len(point) != 2 or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(value) for value in point)
                       for point in trajectory)):
            parser.error(f"Request {index}: expected 64 finite action pairs")
        if not response.get("output_text", "").strip():
            parser.error(f"Request {index}: missing model CoC/output text")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for window in windows:
        index = window["request_idx"]
        response = by_index[index]
        lines = []
        for paragraph in response["output_text"].splitlines():
            lines.extend(textwrap.wrap(paragraph, width=135) or [""])
        text_height = max(1.4, len(lines) * 0.19 + 0.6)
        figure = plt.figure(figsize=(16, 13 + text_height))
        grid = figure.add_gridspec(
            6, 4, height_ratios=[2, 2, 2, 2, 2.3, text_height],
            hspace=0.45, wspace=0.22)
        figure.suptitle(
            f"Clip {window['clip_uuid']} | window {window['window_index']:02d}"
            f" | request {index}\n"
            "OFFLINE SMOKE TEST: template trajectory, not matched to video; "
            "not for vehicle control",
            fontsize=12)
        for slot, frame in enumerate(window["frames"]):
            axis = figure.add_subplot(grid[slot // 4, slot % 4])
            axis.imshow(matplotlib.image.imread(frame["png"]))
            axis.set_title(
                f"{frame['nominal_video_time_seconds']:.1f}s | "
                f"{frame['camera'][len('camera_'):]}", fontsize=9)
            axis.axis("off")
        trajectory = response["output_trajectory"]
        for column, (label, unit) in enumerate((
                ("Acceleration", "accel (model output)"),
                ("Curvature", "kappa (model output)"))):
            axis = figure.add_subplot(grid[4, column * 2:(column + 1) * 2])
            axis.plot(range(len(trajectory)),
                      [point[column] for point in trajectory])
            axis.set_title(f"{label}: raw action sequence, not an XY path")
            axis.set_xlabel("Predicted step index (time step unspecified)")
            axis.set_ylabel(unit)
            axis.grid(alpha=0.3)
        axis = figure.add_subplot(grid[5, :])
        axis.axis("off")
        axis.text(0, 1, "CoC / model output text (unverified)\n\n"
                  + "\n".join(lines), va="top", fontsize=10)
        figure.subplots_adjust(top=0.93, bottom=0.025, left=0.06, right=0.98)
        output = args.output_dir / (
            f"request_{index:03d}_{window['clip_uuid']}"
            f"_window_{window['window_index']:02d}.png")
        figure.savefig(output, dpi=110)
        plt.close(figure)
        artifacts.append({"request_idx": index, "png": str(output.resolve())})
        print(f"PASS request {index}: camera grid + action curves + CoC: "
              f"{output}", flush=True)
    (args.output_dir / "index.json").write_text(
        json.dumps({"source_output": str(args.output_file.resolve()),
                    "images": artifacts}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
