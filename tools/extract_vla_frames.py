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

"""Extract time-major four-camera PNG inputs for offline VLA smoke tests."""

import argparse
import copy
import json
import math
import pathlib

import cv2

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CAMERAS = (
    "camera_front_wide_120_fov",
    "camera_front_tele_30_fov",
    "camera_cross_left_120_fov",
    "camera_cross_right_120_fov",
)


def image_items(request):
    """Return image content items in parser order."""
    return [
        item for message in request["messages"]
        if isinstance(message["content"], list)
        for item in message["content"] if item["type"] == "image"
    ]


def extract_clip(videos, destination, timestamps):
    """Write 16 PNGs ordered by timestamp, then camera, and return mappings."""
    frames = [None] * (len(CAMERAS) * len(timestamps))
    for camera_index, camera in enumerate(CAMERAS):
        source = videos[camera]
        capture = cv2.VideoCapture(str(source))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video: {source}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if abs(fps - 30.0) > 0.001:
                raise ValueError(f"Expected 30 FPS, got {fps}: {source}")
            indices = [round(timestamp * fps) for timestamp in timestamps]
            if len(set(indices)) != len(indices):
                raise ValueError(f"Timestamps select duplicate frames: {source}")
            if indices[-1] >= capture.get(cv2.CAP_PROP_FRAME_COUNT):
                raise ValueError(f"Requested window exceeds video duration: {source}")
            for frame_index in range(indices[-1] + 1):
                success, image = capture.read()
                if not success:
                    raise RuntimeError(
                        f"Cannot decode frame {frame_index}: {source}")
                if frame_index not in indices:
                    continue
                time_index = indices.index(frame_index)
                slot = time_index * len(CAMERAS) + camera_index
                output = destination / f"frame_{slot:02d}.png"
                if not cv2.imwrite(str(output), image):
                    raise RuntimeError(f"Cannot write PNG: {output}")
                decoded = cv2.imread(str(output))
                if (decoded is None or decoded.shape != image.shape
                        or not (decoded == image).all()):
                    raise RuntimeError(f"PNG round-trip mismatch: {output}")
                frames[slot] = {
                    "input_image_index": slot,
                    "timestamp_index": time_index,
                    "camera_index": camera_index,
                    "png": str(output),
                    "camera": camera,
                    "source_video": str(source),
                    "source_frame_index_zero_based": frame_index,
                    "nominal_video_time_seconds": frame_index / fps,
                    "width": image.shape[1],
                    "height": image.shape[0],
                }
        finally:
            capture.release()
    if any(frame is None for frame in frames):
        raise RuntimeError(f"Incomplete frame extraction: {destination}")
    return frames


def write_json(path, data):
    """Write readable UTF-8 JSON with a trailing newline."""
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def default_template():
    """Return the documented offline example, not measured video egomotion."""
    trajectory = [
        [-13.570470809936523, 0.06137955188751221, -0.021795138716697693],
        [-12.617349624633789, 0.05502257123589516, -0.021485785022377968],
        [-11.672354698181152, 0.04983367770910263, -0.019761288538575172],
        [-10.733928680419922, 0.04469458386301994, -0.01891801320016384],
        [-9.802403450012207, 0.03998061269521713, -0.019592544063925743],
        [-8.87903118133545, 0.034502264112234116, -0.016824787482619286],
        [-7.963467121124268, 0.03007129393517971, -0.011910118162631989],
        [-7.054953098297119, 0.02519317716360092, -0.008388019166886806],
        [-6.15440559387207, 0.020703254267573357, -0.003812047652900219],
        [-5.261260032653809, 0.016889382153749466, 0.0007052596774883568],
        [-4.374241828918457, 0.012481293641030788, 0.004680476617068052],
        [-3.4911155700683594, 0.00914590060710907, 0.0029235705733299255],
        [-2.611931085586548, 0.005678099580109119, 7.79956899350509e-05],
        [-1.736931562423706, 0.003552841255441308, -0.0018224255181849003],
        [-0.8664319515228271, 0.001968142343685031, -0.001738768769428134],
        [0.0, 0.0, 0.0],
    ]
    return {
        "batch_size": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "max_generate_length": 128,
        "requests": [{
            "messages": [{
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": "You are a driving assistant that generates safe "
                    "and accurate actions.",
                }],
            }, {
                "role": "user",
                "content": [{"type": "image", "image": ""} for _ in range(16)]
                + [{"type": "trajectory", "trajectory": trajectory}, {
                    "type": "text",
                    "text": "output the chain-of-thought reasoning of the driving "
                    "process, then output the future trajectory.",
                }],
            }],
        }],
    }


def main():
    """Generate per-clip inputs and a combined batch-size-one request file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=pathlib.Path,
                        default=REPO_ROOT / "datasets")
    parser.add_argument("--input-file", type=pathlib.Path,
                        help="Combined output; an existing file is also the "
                        "template. Missing files use the built-in smoke-test "
                        "example. Default: VLA_tests/input_action.json, or "
                        "VLA_tests/selected/UUID/input_action.json with --uuid.")
    parser.add_argument("--frames-dir", type=pathlib.Path,
                        default=REPO_ROOT / "VLA_tests/frames")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--clips", type=int, default=3)
    selection.add_argument("--uuid",
                           help="Select one exact clip UUID with all four cameras; "
                           "mutually exclusive with --clips.")
    parser.add_argument("--windows", type=int, default=8,
                        help="Number of requests per clip.")
    parser.add_argument("--start", type=float, default=2.0,
                        help="First window start time in seconds.")
    parser.add_argument("--window-stride", type=float, default=2.0,
                        help="Seconds between window start times.")
    parser.add_argument("--frame-interval", type=float, default=0.5,
                        help="Seconds between the four timestamps in a window.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace generated PNGs and per-clip JSON files.")
    args = parser.parse_args()
    if args.uuid is not None:
        args.clips = 1
    if args.clips <= 0 or args.windows <= 0:
        parser.error("--clips and --windows must be positive")
    if (not all(math.isfinite(value) for value in
                (args.start, args.window_stride, args.frame_interval))
            or args.start < 0 or args.window_stride <= 0
            or args.frame_interval <= 0):
        parser.error("Times must be finite; start >= 0 and intervals > 0")
    dataset_dir = args.dataset_dir.resolve()
    if args.input_file is None:
        output_dir = REPO_ROOT / "VLA_tests"
        if args.uuid is not None:
            output_dir = output_dir / "selected" / args.uuid
        args.input_file = output_dir / "input_action.json"
    input_file = args.input_file.resolve()
    frames_dir = args.frames_dir.resolve()
    original_text = (input_file.read_text(encoding="utf-8")
                     if input_file.exists() else None)
    if original_text is None:
        template = default_template()
        print("WARNING: Using built-in smoke-test text and 16-point example "
              "trajectory; not measured from this video.")
    else:
        template = json.loads(original_text)
    requests = template["requests"]
    if not requests or template["batch_size"] != 1:
        parser.error("Template requires nonempty requests and batch_size=1")
    if len(requests) not in (1, args.clips, args.clips * args.windows):
        parser.error("Template must have one request, one per clip, "
                     "or one per output window")
    if any(len(image_items(request)) != 16 for request in requests):
        parser.error("Each template request must contain exactly 16 images")
    samples = json.loads(
        (dataset_dir / "samples.json").read_text(encoding="utf-8"))["samples"]
    groups = {}
    for sample in samples:
        groups.setdefault(sample["uuid"], {})[sample["group"]["name"]] = (
            dataset_dir / sample["filepath"])
    if args.uuid is not None:
        if args.uuid not in groups:
            parser.error(f"Unknown clip UUID: {args.uuid}")
        paths = groups[args.uuid]
        missing = [
            camera for camera in CAMERAS
            if camera not in paths or not paths[camera].is_file()
        ]
        if missing:
            parser.error(f"Clip {args.uuid} is missing camera videos: "
                         + ", ".join(missing))
        clips = [(args.uuid, paths)]
    else:
        clips = [
            (uuid, paths) for uuid, paths in sorted(groups.items())
            if all(camera in paths and paths[camera].is_file()
                   for camera in CAMERAS)
        ][:args.clips]
    if len(clips) != args.clips:
        parser.error(f"Need {args.clips} clips with all four cameras")
    destinations = [
        frames_dir / f"clip_{index:02d}_{uuid}"
        for index, (uuid, _) in enumerate(clips, start=1)
    ]
    for destination in destinations:
        if destination.exists() and not args.overwrite:
            parser.error(f"Output exists: {destination}; use --overwrite")
    combined = copy.deepcopy(template)
    combined["requests"] = []
    window_mapping = []
    for index, ((uuid, videos), destination) in enumerate(
            zip(clips, destinations)):
        destination.mkdir(parents=True, exist_ok=args.overwrite)
        clip_requests = []
        for window_index in range(args.windows):
            window_dir = destination / f"window_{window_index:02d}"
            window_dir.mkdir(exist_ok=args.overwrite)
            timestamps = [
                args.start + window_index * args.window_stride
                + step * args.frame_interval for step in range(4)
            ]
            frames = extract_clip(videos, window_dir, timestamps)
            request_index = len(combined["requests"])
            template_index = 0
            if len(requests) == args.clips * args.windows:
                template_index = request_index
            elif len(requests) == args.clips:
                template_index = index
            request = copy.deepcopy(requests[template_index])
            for item, frame in zip(image_items(request), frames):
                item["image"] = frame["png"]
            single = copy.deepcopy(template)
            single["requests"] = [request]
            write_json(window_dir / "input_action.json", single)
            mapping = {
                "request_idx": request_index,
                "clip_uuid": uuid,
                "window_index": window_index,
                "purpose": "Offline smoke test only, not action accuracy "
                "evaluation or vehicle control.",
                "trajectory_provenance": (
                    "Built-in example trajectory; not measured from this video."
                    if original_text is None else
                    "Preserved from input template; "
                    "not verified against this video clip."),
                "ordering": "Time-major: four cameras per timestamp.",
                "cameras": list(CAMERAS),
                "timestamps_seconds": timestamps,
                "camera_layout_status": "Explicit test camera selection; "
                "not verified against the model training layout.",
                "time_alignment_status": "Matching video-relative frame times; "
                "sensor timestamp synchronization is unverified.",
                "frames": frames,
            }
            write_json(window_dir / "frame_mapping.json", mapping)
            window_mapping.append(mapping)
            combined["requests"].append(request)
            clip_requests.append(request)
            print(f"PASS {destination.name}/{window_dir.name}: "
                  f"{timestamps[0]:.1f}-{timestamps[-1]:.1f}s, "
                  "16 images + template trajectory", flush=True)
        single = copy.deepcopy(template)
        single["requests"] = clip_requests
        write_json(destination / "input_action.json", single)
        write_json(destination / "frame_mapping.json", {
            "clip_uuid": uuid,
            "windows": window_mapping[-args.windows:],
        })
    current_text = (input_file.read_text(encoding="utf-8")
                    if input_file.exists() else None)
    if current_text != original_text:
        raise RuntimeError("Input file changed during extraction")
    input_file.parent.mkdir(parents=True, exist_ok=True)
    write_json(input_file, combined)
    write_json(input_file.with_name("window_mapping.json"), {
        "input_file": str(input_file),
        "windows": window_mapping,
    })
    print(f"Written {input_file}; template text and trajectories preserved.")


if __name__ == "__main__":
    main()
