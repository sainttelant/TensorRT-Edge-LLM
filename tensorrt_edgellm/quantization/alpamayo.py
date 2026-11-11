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
"""Experimental text-calibrated INT8 backbone quantization for Alpamayo-R1."""

import gc
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def _rewrite_shards(source: Path, destination: Path,
                    key_map: Callable[[str], Optional[str]],
                    prefix: str) -> tuple[dict[str, str], int]:
    index = source / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        filenames = sorted(set(weight_map.values()))
    else:
        filenames = ["model.safetensors"]
    result: dict[str, str] = {}
    size = 0
    for i, filename in enumerate(filenames):
        tensors = {}
        with safe_open(source / filename, framework="pt", device="cpu") as f:
            for key in f.keys():
                mapped = key_map(key)
                if mapped is None:
                    continue
                if mapped in result or mapped in tensors:
                    raise ValueError(f"Duplicate checkpoint tensor: {mapped}")
                tensor = f.get_tensor(key)
                if tensor.is_floating_point(
                ) and tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float16)
                tensors[mapped] = tensor
                size += tensor.numel() * tensor.element_size()
        if tensors:
            name = f"{prefix}-{i + 1:05d}.safetensors"
            save_file(tensors, destination / name, metadata={"format": "pt"})
            result.update({key: name for key in tensors})
        del tensors
    return result, size


def _write_index(directory: Path, weight_map: dict[str, str],
                 size: int) -> None:
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {
                "total_size": size
            },
            "weight_map": weight_map
        },
                   indent=2) + "\n")


def _restore_checkpoint(source: Path, quantized: Path, output: Path,
                        root: dict) -> None:
    quant_config = json.loads((quantized / "hf_quant_config.json").read_text())
    quant = quant_config["quantization"]
    if (quant.get("quant_algo") != "W8A8_SQ_PER_CHANNEL"
            or quant.get("kv_cache_quant_algo") is not None):
        raise ValueError(
            "Expected INT8 SmoothQuant with an unquantized KV cache.")

    weight_map, size = _rewrite_shards(quantized, output,
                                       lambda key: "vlm." + key, "vlm")
    extra_map, extra_size = _rewrite_shards(
        source, output, lambda key: None
        if key.startswith("vlm.") else key, "action")
    if not extra_map:
        raise ValueError("Alpamayo checkpoint contains no action weights.")
    weight_map.update(extra_map)
    _write_index(output, weight_map, size + extra_size)
    (output / "config.json").write_text(json.dumps(root, indent=2) + "\n")
    # Quantization exclusions use the exporter's logical VLM module names,
    # not the outer Alpamayo checkpoint's vlm.* storage prefix.
    (output / "hf_quant_config.json"
     ).write_text(json.dumps(quant_config, indent=2) + "\n")
    for filename in ("tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json", "preprocessor_config.json",
                     "processor_config.json", "video_preprocessor_config.json",
                     "chat_template.jinja", "chat_template.json"):
        path = quantized / filename
        if path.is_file():
            shutil.copy2(path, output / filename)


def quantize_alpamayo_int8(model_dir: str,
                           output_dir: str,
                           *,
                           device: str = "cuda",
                           text_dataset: str = "cnn_dailymail",
                           num_samples: int = 512) -> str:
    """Quantize only the language backbone; preserve the full VLA checkpoint.

    Calibration is text-only and does not qualify trajectory accuracy or Orin
    deployment. The original checkpoint is never modified.
    """
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import (AutoConfig, AutoModelForImageTextToText,
                              AutoProcessor, AutoTokenizer)

    from ..checkpoint.checkpoint_utils import _build_alpamayo_tokenizer
    from .datasets import resolve_dataset
    from .quantization_configs import build_quant_config
    from .quantize import (_calibrate,
                           _fix_generation_config_for_strict_validate,
                           _normalize_tied_weights_keys,
                           _text_calib_dataloader)

    source = Path(model_dir).resolve()
    output = Path(output_dir).resolve()
    root = json.loads((source / "config.json").read_text())
    if root.get("model_type") != "alpamayo_r1":
        raise ValueError(
            "Experimental Alpamayo INT8 requires model_type=alpamayo_r1.")
    if root.get("vlm_backend") != "qwenvl3":
        raise ValueError(
            "Experimental Alpamayo INT8 requires vlm_backend=qwenvl3.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if source == output or source in output.parents:
        raise ValueError("The output must be outside the original checkpoint.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite nonempty checkpoint: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    logger.warning(
        "Experimental Alpamayo INT8: text-only calibration; visual, "
        "action, LM head and KV cache stay FP16. Not Orin-qualified.")
    vlm_name = root.get("vlm_name_or_path", "Qwen/Qwen3-VL-8B-Instruct")
    config = AutoConfig.from_pretrained(vlm_name)
    if config.model_type != "qwen3_vl":
        raise ValueError(
            f"Expected a Qwen3-VL config, got {config.model_type}.")
    config.text_config.vocab_size = root["vocab_size"]
    config.text_config.use_cache = False
    config.architectures = ["Qwen3VLForConditionalGeneration"]

    with tempfile.TemporaryDirectory(prefix=".alpamayo-int8-",
                                     dir=output.parent) as temp:
        stage = Path(temp)
        vlm = stage / "vlm"
        quantized = stage / "quantized"
        complete = stage / "complete"
        vlm.mkdir()
        complete.mkdir()
        config.save_pretrained(vlm)
        AutoProcessor.from_pretrained(vlm_name).save_pretrained(vlm)
        _build_alpamayo_tokenizer(root, str(vlm))
        if not (vlm / "tokenizer.json").is_file():
            raise FileNotFoundError("Failed to prepare Alpamayo tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(vlm)
        if len(tokenizer) != root["vocab_size"]:
            raise ValueError(
                "Alpamayo tokenizer vocabulary does not match checkpoint.")
        for key, token in (("history", "<|traj_history|>"),
                           ("future", "<|traj_future|>")):
            if tokenizer.convert_tokens_to_ids(
                    token) != root["traj_token_ids"][key]:
                raise ValueError(
                    f"Alpamayo trajectory token ID mismatch: {token}")
        logger.info("Preparing Qwen3-VL calibration checkpoint in %s", vlm)
        weight_map, size = _rewrite_shards(
            source, vlm, lambda key: key[len("vlm."):]
            if key.startswith("vlm.") else None, "model")
        if "lm_head.weight" not in weight_map:
            raise ValueError("Missing Alpamayo vlm.lm_head.weight.")
        _write_index(vlm, weight_map, size)
        model, loading_info = AutoModelForImageTextToText.from_pretrained(
            vlm, dtype=torch.float16, output_loading_info=True)
        if any(loading_info.values()):
            raise ValueError(f"Alpamayo VLM checkpoint did not load exactly: "
                             f"{loading_info}")
        model.to(device).eval()
        loader = _text_calib_dataloader(tokenizer,
                                        resolve_dataset(text_dataset, "text"),
                                        batch_size=1,
                                        num_samples=num_samples)
        print(
            f"Alpamayo INT8 text calibration: {text_dataset}, "
            f"{len(loader)} batches",
            flush=True)
        mtq.quantize(model,
                     build_quant_config("int8_sq"),
                     forward_loop=lambda m: _calibrate(m, loader))
        for name, module in model.named_modules():
            if getattr(module, "is_enabled", False) and hasattr(
                    module, "amax"):
                if module.amax is None or not torch.isfinite(
                        module.amax).all():
                    raise ValueError(
                        f"Uncalibrated or non-finite quantizer: {name}")
        _fix_generation_config_for_strict_validate(model)
        _normalize_tied_weights_keys(model)
        with torch.inference_mode():
            export_hf_checkpoint(model, export_dir=str(quantized))
        tokenizer.save_pretrained(quantized)
        AutoProcessor.from_pretrained(vlm).save_pretrained(quantized)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _restore_checkpoint(source, quantized, complete, root)
        (complete / "experimental_quantization.json").write_text(
            json.dumps(
                {
                    "method": "int8_sq",
                    "calibration": "text-only",
                    "text_dataset": text_dataset,
                    "num_samples": num_samples,
                    "visual": "fp16",
                    "action": "fp16",
                    "lm_head": "fp16",
                    "kv_cache": "fp16",
                    "orin_validated": False,
                },
                indent=2) + "\n")
        if output.exists():
            output.rmdir()
        complete.rename(output)
    logger.info("Saved experimental Alpamayo INT8 checkpoint to %s", output)
    return str(output)
