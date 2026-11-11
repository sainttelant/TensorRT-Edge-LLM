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
"""Experimental Alpamayo checkpoint remapping and precision preservation."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from tensorrt_edgellm.quantization import alpamayo


def test_rewrite_shards_filters_and_casts_without_modifying_source(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    original = torch.ones(2, 2, dtype=torch.bfloat16)
    save_file(
        {
            "vlm.lm_head.weight": original,
            "expert.weight": original.clone()
        }, source / "model.safetensors")
    keys, size = alpamayo._rewrite_shards(
        source, output, lambda k: k[4:]
        if k.startswith("vlm.") else None, "model")
    assert set(keys) == {"lm_head.weight"}
    assert size == 8
    result = load_file(output / keys["lm_head.weight"])
    assert result["lm_head.weight"].dtype == torch.float16
    assert load_file(
        source / "model.safetensors")["expert.weight"].dtype == torch.bfloat16


def test_restore_preserves_action_visual_and_quantization_scales(tmp_path):
    source, quantized, output = (tmp_path / name
                                 for name in ("source", "quantized", "output"))
    for path in (source, quantized, output):
        path.mkdir()
    save_file(
        {
            "vlm.lm_head.weight": torch.ones(2, 2),
            "expert.layers.0.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        }, source / "model.safetensors")
    quant_tensors = {
        "model.language_model.layers.0.self_attn.q_proj.weight":
        torch.ones(2, 2, dtype=torch.int8),
        "model.language_model.layers.0.self_attn.q_proj.weight_scale":
        torch.ones(2, dtype=torch.float32),
        "model.visual.blocks.0.weight":
        torch.ones(2, 2, dtype=torch.float16),
        "lm_head.weight":
        torch.ones(2, 2, dtype=torch.float16),
    }
    save_file(quant_tensors, quantized / "model.safetensors")
    quant_config = {
        "quantization": {
            "quant_algo": "W8A8_SQ_PER_CHANNEL",
            "kv_cache_quant_algo": None,
            "exclude_modules": ["lm_head", "model.visual*"],
        }
    }
    (quantized / "hf_quant_config.json").write_text(json.dumps(quant_config))
    root = {"model_type": "alpamayo_r1", "traj_token_ids": {"future": 42}}
    alpamayo._restore_checkpoint(source, quantized, output, root)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    tensors = {}
    for shard in set(index["weight_map"].values()):
        tensors.update(load_file(output / shard))
    assert set(tensors) == {"vlm." + k
                            for k in quant_tensors
                            } | {"expert.layers.0.weight"}
    assert tensors["expert.layers.0.weight"].dtype == torch.float16
    for key, tensor in quant_tensors.items():
        assert torch.equal(tensors["vlm." + key], tensor)
        assert tensors["vlm." + key].dtype == tensor.dtype
    assert json.loads((output / "config.json").read_text()) == root
    assert json.loads(
        (output / "hf_quant_config.json").read_text()) == quant_config
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size()
                                                  for t in tensors.values())


def test_restore_rejects_non_int8_quantization(tmp_path):
    quantized = tmp_path / "quantized"
    quantized.mkdir()
    (quantized / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {
            "quant_algo": "FP8"
        }}))
    with pytest.raises(ValueError, match="INT8 SmoothQuant"):
        alpamayo._restore_checkpoint(tmp_path, quantized, tmp_path, {})


def test_experimental_cli_requires_int8_and_rejects_fp8_kv():
    from tensorrt_edgellm.scripts import quantize
    common = [
        "llm", "--model-dir", "/unused", "--output-dir", "/unused-output",
        "--experimental-alpamayo-int8"
    ]
    for flags in ([], ["--quantization", "fp8"], [
            "--quantization", "int8_sq", "--kv-cache-quantization", "fp8"
    ]):
        with pytest.raises(SystemExit) as exc:
            quantize.main(common + flags)
        assert exc.value.code == 2


@pytest.mark.parametrize("missing_weight", [False, True])
def test_experimental_pipeline_on_tiny_qwen_vl(tmp_path, monkeypatch,
                                               missing_weight):
    pytest.importorskip("modelopt.torch.quantization")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (AutoProcessor, PreTrainedTokenizerFast,
                              Qwen3VLConfig, Qwen3VLForConditionalGeneration)

    from tensorrt_edgellm.checkpoint import checkpoint_utils
    from tensorrt_edgellm.quantization import datasets

    config = Qwen3VLConfig(text_config={
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.,
            "mrope_section": [1, 1, 2],
        },
    },
                           vision_config={
                               "depth": 1,
                               "hidden_size": 32,
                               "intermediate_size": 64,
                               "num_heads": 4,
                               "out_hidden_size": 32,
                               "deepstack_visual_indexes": [],
                           })
    model = Qwen3VLForConditionalGeneration(config).half().eval()
    state = {
        "vlm." + key: tensor.clone()
        for key, tensor in model.state_dict().items()
    }
    if missing_weight:
        del state["vlm.model.language_model.layers.0.self_attn.q_proj.weight"]
    state["expert.weight"] = torch.ones(2, 2, dtype=torch.bfloat16)
    source = tmp_path / "source"
    source.mkdir()
    base = tmp_path / "base"
    config.save_pretrained(base)
    save_file(state, source / "model.safetensors")
    root = {
        "model_type": "alpamayo_r1",
        "vlm_backend": "qwenvl3",
        "vocab_size": 64,
        "vlm_name_or_path": str(base),
        "traj_token_ids": {
            "history": 2,
            "future": 3
        },
    }
    (source / "config.json").write_text(json.dumps(root))
    vocab = {
        token: i
        for i, token in
        enumerate(["<pad>", "<unk>", "<|traj_history|>", "<|traj_future|>"] +
                  [f"word{i}" for i in range(60)])
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
                                        pad_token="<pad>",
                                        unk_token="<unk>")

    class Processor:

        def save_pretrained(self, path):
            tokenizer.save_pretrained(path)

    monkeypatch.setattr(AutoProcessor, "from_pretrained",
                        lambda *a, **k: Processor())
    monkeypatch.setattr(checkpoint_utils, "_build_alpamayo_tokenizer",
                        lambda root, path: tokenizer.save_pretrained(path))
    monkeypatch.setattr(datasets, "resolve_dataset",
                        lambda *a: lambda: iter(["word1 word2 word3"] * 2))
    output = tmp_path / "quantized"
    if missing_weight:
        with pytest.raises(ValueError, match="did not load exactly"):
            alpamayo.quantize_alpamayo_int8(str(source),
                                            str(output),
                                            device="cpu",
                                            num_samples=2)
        assert not output.exists()
    else:
        alpamayo.quantize_alpamayo_int8(str(source),
                                        str(output),
                                        device="cpu",
                                        num_samples=2)
        index = json.loads(
            (output /
             "model.safetensors.index.json").read_text())["weight_map"]
        assert set(state) <= set(index)
        key = "vlm.model.language_model.layers.0.self_attn.q_proj.weight"
        assert load_file(output / index[key])[key].dtype == torch.int8
        key = "vlm.model.visual.patch_embed.proj.weight"
        assert torch.equal(load_file(output / index[key])[key], state[key])
        assert json.loads((output / "experimental_quantization.json"
                           ).read_text())["orin_validated"] is False
    assert not list(tmp_path.glob(".alpamayo-int8-*"))
