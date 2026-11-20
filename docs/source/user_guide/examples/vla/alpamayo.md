# Alpamayo-R1-10B (Vision-Language-Action)

Complete workflow for running [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B), a Vision-Language-Action (VLA) model that combines a VLM backbone with an action expert.

> **Prerequisites:** Complete the
> [Installation Guide](../../getting_started/installation.md) for both x86
> host and edge device. The checkpoint may require Hugging Face login and
> license access before downloading.

---

## Architecture Overview

Alpamayo-R1-10B runs as a chained VLM + action pipeline:

1. **VLM backbone** — exported through `tensorrt_edgellm` as `onnx/llm` and `onnx/visual`; it processes image, text, and trajectory-history tokens and generates language output.
2. **Action expert** — exported through `tensorrt_edgellm` as `onnx/action`; it consumes the VLM KV cache and produces future action waypoints.

Both stages are run with a single `action_inference` invocation. The engines are built separately and loaded together at runtime.

For ONNX export, `--tp-size` selects the LLM's TP size and the compatible
action model: TP=1 uses paged prefix reuse, and TP>1 uses dense KV inputs.
The action engine runs on one device in both cases.

For the [direct engine builder](../../getting_started/direct-engine-builder.md),
Alpamayo action builds default to `--reuse-llm-kv-cache=true` for an LLM
using TP=1. For an LLM using TP>1, build action separately with
`--components action --reuse-llm-kv-cache=false`.

The current TensorRT Edge-LLM input parser accepts image files, text, and
trajectory history points as `[x, y, z]`. The action output is written as
`output_trajectory`, a list of `(accel, kappa)` acceleration and curvature pairs.

---

## Step 1: Export (x86 Host)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Alpamayo-R1-10B
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Download Alpamayo-R1-10B from HuggingFace
hf auth login
hf download nvidia/Alpamayo-R1-10B --local-dir $MODEL_NAME

# Export language model backbone, visual encoder, and action expert
trtedge export \
  $MODEL_NAME \
  $MODEL_NAME/onnx \
  --max-kv-cache-capacity 4096
```

This creates `onnx/llm`, `onnx/visual`, and `onnx/action`.

> **Note:** The export also reads the public
> [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
> configuration from Hugging Face.
>
> This guide covers the FP16 flow. Quantized export of the LLM is not covered
> in 0.12.0; see [Limitations and Known Issues](../../getting_started/limitations.md).
>
> The action expert uses FP16 weights and action KV. The `--max-kv-cache-capacity`
> value must match the `--max-kv-cache-capacity` used when building the LLM engine
> in Step 3.

An opt-in, experimental text-calibrated INT8 SmoothQuant language-backbone
workflow is documented in
[the R1_10B quantization walkthrough](../../../../../README_quantize_R1_10B.md).
It keeps the visual encoder, action expert, LM head, and KV cache unquantized.
It is not a qualified Alpamayo or Orin deployment path.

## Step 2: Transfer to Device

```bash
scp -r $MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

## Step 3: Build Engines (Thor Device)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Alpamayo-R1-10B
cd /path/to/TensorRT-Edge-LLM

# Build language model engine
./build/bin/llm_build \
  --onnx-dir $WORKSPACE_DIR/$MODEL_NAME/onnx/llm \
  --engine-dir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --max-input-len 3424 \
  --max-kv-cache-capacity 4096 \
  --max-batch-size 1

# Build visual encoder engine
./build/bin/visual_build \
  --onnx-dir $WORKSPACE_DIR/$MODEL_NAME/onnx/visual \
  --engine-dir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --min-image-tokens 160 \
  --max-image-tokens 18432 \
  --max-image-tokens-per-image 192

# Build action expert engine
./build/bin/action_build \
  --onnx-dir $WORKSPACE_DIR/$MODEL_NAME/onnx/action \
  --engine-dir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --max-batch-size 1
```

Use `--max-batch-size 1` for single-scene deployment, and the same value for
the language model and action engines.

`visual_build` writes `$WORKSPACE_DIR/$MODEL_NAME/engines/visual/`, and
`action_build` writes `$WORKSPACE_DIR/$MODEL_NAME/engines/action/`. Both
subdirectories are loaded through `--multimodal-engine-dir` during inference.

## Step 4: Prepare an Input

Alpamayo takes 4 cameras at 4 timesteps (16 images) and 16 egomotion history
points. Build requests from the
[NVIDIA PhysicalAI Autonomous Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
dataset with the dataset preparation script. Accept the dataset license and run
`hf auth login` first. Only the selected clips are downloaded:

```bash
pip install -r examples/accuracy/requirements-physical-ai-av.txt
python examples/accuracy/scripts/prepare_dataset.py \
  --dataset PhysicalAIAV \
  --max_samples 1 \
  --output_dir $WORKSPACE_DIR/physical_ai_av
```

Copy the output directory to the device. Each request has this format, and you
can also write one by hand. The `trajectory` item provides past egomotion
history as `[x, y, z]` positions:

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "max_generate_length": 128,
    "requests": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a driving assistant that generates safe and accurate actions."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "/path/to/frame_00.png"},
                        {"type": "image", "image": "/path/to/frame_01.png"},
                        {"type": "image", "image": "/path/to/frame_02.png"},
                        {"type": "image", "image": "/path/to/frame_03.png"},
                        {"type": "image", "image": "/path/to/frame_04.png"},
                        {"type": "image", "image": "/path/to/frame_05.png"},
                        {"type": "image", "image": "/path/to/frame_06.png"},
                        {"type": "image", "image": "/path/to/frame_07.png"},
                        {"type": "image", "image": "/path/to/frame_08.png"},
                        {"type": "image", "image": "/path/to/frame_09.png"},
                        {"type": "image", "image": "/path/to/frame_10.png"},
                        {"type": "image", "image": "/path/to/frame_11.png"},
                        {"type": "image", "image": "/path/to/frame_12.png"},
                        {"type": "image", "image": "/path/to/frame_13.png"},
                        {"type": "image", "image": "/path/to/frame_14.png"},
                        {"type": "image", "image": "/path/to/frame_15.png"},
                        {
                            "type": "trajectory",
                            "trajectory": [
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
                                [0.0, 0.0, 0.0]
                            ]
                        },
                        {
                            "type": "text",
                            "text": "output the chain-of-thought reasoning of the driving process, then output the future trajectory."
                        }
                    ]
                }
            ]
        }
    ]
}
```

## Step 5: Run Inference (Thor Device)

```bash
cd /path/to/TensorRT-Edge-LLM

./build/bin/action_inference \
  --engine-dir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --multimodal-engine-dir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --input-file $WORKSPACE_DIR/physical_ai_av/physical_ai_av_dataset.json \
  --output-file $WORKSPACE_DIR/output_action.json
```

Check `output_action.json` for the response. Each entry includes `output_text` (VLM language output) and `output_trajectory` (64 predicted (accel, kappa) acceleration and curvature pairs).

## Live camera example (experimental)

`alpaway_real` runs the same VLM and action engines with four camera streams.
This is an experimental inference example, **not a vehicle controller or a
safety-validated driving system**. Do not send its output to actuators.

```bash
export TRT_PACKAGE_DIR=/usr
cmake -S . -B build-orin -DTRT_PACKAGE_DIR="$TRT_PACKAGE_DIR" -DBUILD_ALPAWAY_REAL=ON
cmake --build build-orin --target alpaway_real -j2
export LD_LIBRARY_PATH="$TRT_PACKAGE_DIR/lib:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"
export EDGELLM_PLUGIN_PATH="$PWD/build-orin/lib/libNvInfer_edgellm_plugin.so"
./build-orin/bin/alpaway_real \
  --camera-config examples/configs/camera_video_smoke.json \
  --trajectory-file /path/to/trajectory.json \
  --engine-dir "$WORKSPACE_DIR/$MODEL_NAME/engines/llm" \
  --multimodal-engine-dir "$WORKSPACE_DIR/$MODEL_NAME/engines" \
  --max-windows 1
```

The optional OpenCV dependency requires its core, imgproc, videoio and highgui
development components. `camera_video_smoke.json` references the local sample
dataset, which is not distributed with the repository; replace the paths with
four matching videos. File paths are relative to the process working directory.
For camera streams, pass a different config with `--camera-config` and restart
the program. There is no automatic failover or live source switching.

For the local `R1_10B` workspace and clip
`5e9a00f1-f575-43c8-a4e2-62e52a651dbb`, run from the repository root:

```bash
./run_alpaway_smoke.bash
# On a device with a working graphical display:
./run_alpaway_smoke.bash --show
# Schematic path with EXPLICIT visualization assumptions, not measured speed:
./run_alpaway_smoke.bash --show --preview-speed-mps 2.0 --preview-dt 0.1
```

This wrapper starts and monitors the example trajectory producer, waits for
its first update, runs the FP16 video example, then cleans up the producer and
its private temporary files. No second terminal is needed. It only permits
file-based camera sources and explicitly replays synthetic example history,
not measured motion. Use `--max-windows N` to change the default single window.
`--help` lists environment overrides for video config, history template and
engine/build directories. Relative paths are resolved from the repository
root. The local dataset, template and engines must already exist; the wrapper
does not export models or build engines.

The `--show` dashboard places the matched front-wide input on the left and
large, wrapped model explanation text on the right, with acceleration and
curvature plots below the camera image. Control/discrete tokens are omitted
from display text only; the raw JSON output is unchanged. Token-only output is
explicitly labeled as lacking readable explanation. Long explanations are
visibly truncated in the display; use raw JSON for the complete text. The
OpenCV Hershey font is intended for the model's English/ASCII output.

Without both `--preview-speed-mps` and `--preview-dt`, no XY path is invented:
the dashboard shows the raw action curves and a missing-assumptions notice.
With both supplied, the renderer holds each `(accel, curvature)` pair for one
timestep, integrates speed and heading, and draws a **schematic bird's-eye
path**, not a projection onto the camera. Green segments join circular
waypoints; yellow triangles indicate direction and a magenta triangle marks
the initial ego position. Forward is up and left is left, with equal metric
scale on both axes. Positive curvature turns left. Speed is clamped at zero
when braking, so reversing is not modeled. Labels show the supplied assumptions;
they are not inferred from the example trajectory or measured from the video.
These visualization arguments are also accepted by `alpaway_real` directly.

ESC stops the display between inference calls. Inference itself is
synchronous, so the GUI may pause while a request runs. For a finite
`--max-windows` run with `--show`, the final result stays visible until a key
is pressed. Headless runs without `--show` do not open a window or wait for
keyboard input.

For manual execution with a separately running trajectory producer:

```bash
export ENGINE_ROOT="$PWD/tensorrt-edgellm-workspace/R1_10B/engines"
export LLM_ENGINE_DIR="$PWD/tensorrt-edgellm-workspace/R1_10B/engines-fp16-diagnostic/llm"
export EDGELLM_PLUGIN_PATH="$PWD/build-orin/lib/libNvInfer_edgellm_plugin.so"
export TRT_PACKAGE_DIR=/usr
export LD_LIBRARY_PATH="$TRT_PACKAGE_DIR/lib:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"
./build-orin/bin/alpaway_real \
  --camera-config examples/configs/camera_video_5e9a00f1.json \
  --trajectory-file /tmp/alpaway-smoke-trajectory.json \
  --engine-dir "$LLM_ENGINE_DIR" \
  --multimodal-engine-dir "$ENGINE_ROOT" \
  --max-windows 1
```

Start the smoke trajectory producer below before this command. On this local
workspace, `VLA_tests/selected/26f8dc7e-a9a2-4302-b847-af2009abf677/input_action.json`
is an existing offline template for that producer; its example history is not
measured motion from the selected clip. Those local dataset/template files are
not distributed with the repository. Do not pass literal `/path/to/...`
placeholders. In the generic command, unset `WORKSPACE_DIR` and `MODEL_NAME`
expand to `//engines/llm`; set them first or use `ENGINE_ROOT` as above.
`LLM_ENGINE_DIR` selects the separately built FP16 baseline described below;
it must exist before running this command. Visual/action engines are still
loaded from `ENGINE_ROOT`. Do not replace only a config or tokenizer file to
change an engine's precision.

Each `stream_cameras` entry must appear in this order: `front_wide`,
`front_tele`, `cross_left`, `cross_right`. `type` is `usb` (integer `device`),
`rtsp` (URL `source`), `gstreamer` (pipeline `source`), or `file` (video path
`source`). OpenCV must have the corresponding backend installed. Do not commit
URLs containing credentials. The ShengTai example contains calibration data
and empty/invalid source placeholders: supply actual sources and verify their
viewpoints. The original `front/back/left/right` calibration is not a mapping
to the four model roles. Calibration is retained for integration but is not
applied by this example; no undistortion or extrinsic transformation is done.

The program keeps capturing while inference runs, samples four time steps at
approximately 0.5-second intervals, and passes 16 RGB images directly as
in-memory tensors in time-major camera order. It drops a window when sampling
is delayed and fails if fresh frames are unavailable. Host receipt times must
be at most 250 ms old and 100 ms apart; these checks **do not establish exposure
synchronization**. Video files are independently paced and are only smoke-test
inputs. For real cameras, confirm source latency, exposure synchronization,
viewpoints and calibration separately. Capture backends may buffer frames or
block on reads; RTSP requests five-second open/read timeouts, while other
backends depend on their own timeout configuration.

The trajectory producer must atomically replace a JSON file containing 16
time-ordered measured positions in the model's expected ego coordinate frame,
with the newest measurement time in Unix milliseconds:

```json
{"timestamp_ms": 1730000000000, "trajectory": [[0, 0, 0], "... 15 more [x,y,z] points ..."]}
```

This abbreviated JSON is illustrative, not valid input. The timestamp must be
within one second of the latest host-received image; this does not prove that
the whole history matches the image window. No stale, fabricated or static
trajectory is substituted. For a smoke test only, an external producer may
replay the example history from the offline input with a refreshed timestamp;
that history is not measured vehicle motion and must not be used for real
driving. `vehicle::TrajectoryHistory::updateTrajectory()` accepts a complete
history directly, `saveJson()` atomically publishes it, and `loadJson()` reads
it. An in-process integration can instead supply `points()` to the request.

For local smoke testing only, the following producer replays the trajectory in
an existing offline input JSON. Run it in another terminal, pass the output
path as `--trajectory-file`, and stop it with Ctrl-C after the test:

```bash
python3 - /path/to/input_action.json /tmp/alpaway-smoke-trajectory.json <<'PY'
import json, os, sys, time

with open(sys.argv[1]) as f:
    request = json.load(f)["requests"][0]
points = next(item["trajectory"] for message in request["messages"]
              for item in message["content"] if item["type"] == "trajectory")
while True:
    temporary = sys.argv[2] + ".tmp"
    with open(temporary, "w") as f:
        json.dump({"timestamp_ms": time.time_ns() // 1_000_000,
                   "trajectory": points}, f)
    os.replace(temporary, sys.argv[2])
    time.sleep(0.1)
PY
```

For an SDK callback, `cameras::FrameBuffer::publish(cameraIndex, bgr,
steady_clock_timestamp)` copies a BGR frame into one of the four slots.
`FrameBuffer` implements the same `FrameSource` interface as `CameraRig`:
replace the constructed `CameraRig`/`frameSource` in the example with the
SDK-backed buffer and start the SDK producer before sampling. The timestamp
must use the host steady clock, strictly increase per camera, and describe host
receipt time; preserve and validate sensor exposure timestamps separately.
Published buffers are copied, and snapshots retain ownership until inference
finishes. The example prints one JSON result per completed window; logs may
also appear on stdout. `--show` displays the matched-frame dashboard and ESC exits.
`--max-windows` bounds runs; absent that option, the example continues until
interrupted, a stream ends, or an error occurs.

### Rebuild Orin engines after a plugin upgrade

If `alpaway_real` fails while loading `llm.engine` with `XQA JIT blob does
not start with the expected magic word`, the serialized engine was built with
an incompatible Edge-LLM plugin. This was observed when loading the old Orin
FP16 engine with the 0.12.0 runtime. The failure happens before processing the
camera inputs; changing the video configuration or editing `config.json` does
not fix it.

On the target Orin, build the `llm_build`, `visual_build`, `action_build`, and
`alpaway_real` targets from the same checkout configured with
`BUILD_ALPAWAY_REAL=ON`. Use the same `NvInfer_edgellm_plugin` library for
engine build and inference.

An engine can build successfully from an older ONNX export but still fail
runtime initialization. In the 0.12.0 Orin smoke test, rebuilding from the
old export cleared the XQA error and initialized the visual runner, but the
old action graph failed with `Action engine KV contract does not match LLM
tensor parallelism`. For TP=1, use a current exporter that emits the paged
prefix KV inputs. Rebuild the action engine from that export; do not edit the
engine metadata or bypass the runtime check.

On the export host, use the matching checkout and an existing Alpamayo
checkpoint (see [Step 1](#step-1-export-x86-host)):

```bash
trtedge export "$CHECKPOINT_DIR" "$WORKSPACE_DIR/$MODEL_NAME/onnx-fp16-v0.12.0" \
  --tp-size 1 --max-kv-cache-capacity 4096
```

Transfer `onnx-fp16-v0.12.0/` to the Orin, including the external weight
files and other export sidecars. Build into a fresh directory so the
previous engines remain available:

```bash
export MODEL_ROOT="$PWD/tensorrt-edgellm-workspace/R1_10B"
export TRT_PACKAGE_DIR=/usr
export LD_LIBRARY_PATH="$TRT_PACKAGE_DIR/lib:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export EDGELLM_PLUGIN_PATH="$PWD/build-orin/lib/libNvInfer_edgellm_plugin.so"
ONNX_ROOT="$MODEL_ROOT/onnx-fp16-v0.12.0"
ENGINE_ROOT="$MODEL_ROOT/engines-fp16-v0.12.0-tp1"

mkdir "$ENGINE_ROOT" && \
./build-orin/bin/llm_build \
  --onnx-dir "$ONNX_ROOT/llm" \
  --engine-dir "$ENGINE_ROOT/llm" \
  --max-input-len 3424 --max-kv-cache-capacity 4096 --max-batch-size 1 \
  --profiling-detailed && \
./build-orin/bin/visual_build \
  --onnx-dir "$ONNX_ROOT/visual" \
  --engine-dir "$ENGINE_ROOT" \
  --min-image-tokens 160 --max-image-tokens 18432 --max-image-tokens-per-image 192 && \
./build-orin/bin/action_build \
  --onnx-dir "$ONNX_ROOT/action" \
  --engine-dir "$ENGINE_ROOT" \
  --max-batch-size 1
```

After all three builds succeed, point both engine overrides at the rebuilt
set. Run headless first; add `--show` for the dashboard on a device with a
working display.

```bash
LLM_ENGINE_DIR="$ENGINE_ROOT/llm" MULTIMODAL_ENGINE_DIR="$ENGINE_ROOT" \
  ./run_alpaway_smoke.bash --max-windows 1
```

### Repeated discrete tokens instead of language

`<i1495>` is a model discrete token (ID 153164 in this local tokenizer), not a
character-encoding error. A nonempty string consisting only of repeated
discrete tokens is not a successful language-generation result. Likewise,
64 finite action pairs establish an output shape, not driving accuracy.
Do not hide the failure by stripping tokens, banning that particular token,
or changing temperature.

Local diagnosis on 2026-11-19 reproduced this behavior with the existing
approximately 7.8 GiB LLM engine in both offline VLA and text-only requests.
The captured pre-sampling logits were finite and ranked those discrete tokens
highest. With the same runtime and visual/action engines, a separately built
FP16 LLM restored natural-language output for both an offline VLA request and
the four videos from clip `5e9a00f1-f575-43c8-a4e2-62e52a651dbb`; both VLA
requests also returned 64 finite action pairs. This isolates the regression
to the old LLM engine/artifact path, but does not distinguish quantization,
export, and build defects within that path. The experimental text-only INT8
calibration is not a qualified substitute for a multimodal FP16 baseline.

The local FP16 baseline was built from the existing FP16 ONNX export without
overwriting the old engine:

```bash
export MODEL_ROOT="$PWD/tensorrt-edgellm-workspace/R1_10B"
export TRT_PACKAGE_DIR=/usr
export LD_LIBRARY_PATH="$TRT_PACKAGE_DIR/lib:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"
export EDGELLM_PLUGIN_PATH="$PWD/build-orin/lib/libNvInfer_edgellm_plugin.so"
# Use a new directory if this output already exists.
./build-orin/bin/llm_build \
  --onnx-dir "$MODEL_ROOT/onnx/llm" \
  --engine-dir "$MODEL_ROOT/engines-fp16-diagnostic/llm" \
  --max-input-len 3424 \
  --max-kv-cache-capacity 4096 \
  --max-batch-size 1 \
  --profiling-detailed
```

This local build took approximately 7.6 minutes and produced an approximately
26 GiB LLM engine. It has substantial shared-memory requirements on Orin.
The successful test only establishes recovery of language generation and the
action output shape, not semantic or numerical equivalence to Hugging Face.
The original export still reports model version 0.11.0 against runtime 0.11.1,
and the reused visual/action engines still report device-compatibility
warnings. For qualification, re-export with the matching software version,
rebuild all engines on the target device, and compare against a reference
using aligned measured inputs. Do not edit the version string to suppress
the warning.
