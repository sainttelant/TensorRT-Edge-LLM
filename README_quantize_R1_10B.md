<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# R1_10B 实验性 INT8 量化与 ONNX 导出

## 1. 范围和限制

本流程针对本地 `R1_10B`，即 `model_type=alpamayo_r1` 的
Alpamayo-R1-10B，不是普通 Qwen 文本模型。

**官方 Alpamayo 示例仅支持 FP16。本流程是显式启用的实验路径，
不代表完整 VLA 模型已在 Jetson Orin 上通过验证。**

| 组件 | 本次使用的精度 |
|---|---|
| 语言骨干的 252 个 Linear 权重及其输入激活 | INT8 SmoothQuant，W8A8 |
| Embedding、归一化、LM head | FP16 |
| Visual encoder | FP16 |
| Action expert | FP16；原有少量 FP32 辅助张量保留 |
| KV cache | FP16 |

不启用 FP8、FP8 KV cache、NVFP4 或 MXFP8。Orin 支持 INT8/FP16，
但精度类型兼容不等于 Alpamayo 的全部算子、引擎和轨迹输出已经验证。
本次只使用 **32 条 WikiText 文本**做校准，不能代表图像、驾驶历史和
轨迹任务的真实激活分布；不能据此作精度、速度或显存承诺。

量化与导出是两个步骤：

```text
原始 R1_10B
  -> trtedge quantize llm：量化 checkpoint（不是 ONNX）
  -> trtedge export：llm / visual / action 三路 ONNX
  -> 在 Orin 上构建 TensorRT engines
  -> 在 Orin 上运行推理并评估轨迹精度
```

后两步尚未执行。不要将 x86 主机生成的 TensorRT engine 直接用于 Orin。

## 2. 本次环境与目录

实际执行环境：

- Ubuntu 24.04，Python 3.12.3；
- NVIDIA RTX 5000 Ada，SM89；量化只使用 GPU 0；
- 虚拟环境：`/root/.venv-edgellm`，继承系统包；
- torch 2.13.0、transformers 5.14.1、nvidia-modelopt 0.39.0；
- datasets 4.4.1、onnx 1.19.0、onnxscript 0.7.1、safetensors 0.8.0。

这是本次实际测试的现有环境，不是新的推荐依赖矩阵。仓库 tools extra
要求更高版本的 ModelOpt；本环境会出现版本未经过联合测试的警告。
不要为了复现盲目替换系统级 torch、CUDA 或 TensorRT。
实验路径不传递新版本才支持的 `extra_state_dict` 参数。

```text
/workspace/
  R1_10B/                         # 原始权重，不修改
  tensorrt-edgellm-workspace/R1_10B/
    quantizedcheckpoint/          # 恢复 Alpamayo 命名结构的混合精度 checkpoint
    quantizedonnx/
      llm/
      visual/
      action/
    logs/
      quantize-int8-sq-text.log    # 成功的文本校准日志
      export-int8-sq.log          # 成功的 ONNX 导出日志
```

`quantize-int8-sq.log` 是早期失败尝试的日志，不是最终成功记录。
本次 checkpoint 约 15 GiB；三路 ONNX 和 sidecar 合计约 14 GiB。
重跑还需要临时 VLM checkpoint、ModelOpt 导出以及校准数据缓存空间，
建议至少留出 60 GiB 可用磁盘空间。临时 `.alpamayo-int8-*` 目录自动清理。

## 3. 进入环境和 CLI

本环境没有独立的 `trtedge` 可执行文件，但源码统一 CLI 可以运行。
在同一个 Bash 会话定义下面的函数；它与 `trtedge` entry point 调用相同：

```bash
cd /workspace
source /root/.venv-edgellm/bin/activate

trtedge() {
    /root/.venv-edgellm/bin/python -m tensorrt_edgellm.cli "$@"
}

trtedge quantize llm --help
trtedge export --help

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

ROOT=/workspace/tensorrt-edgellm-workspace/R1_10B
mkdir -p "$ROOT/logs"
```

量化和 ONNX 导出**不需要**调用 `runtime.load()`。
源码 editable 安装缺失 `_native/variants.json` 不阻止这两步。
原生运行库是后续引擎/推理的另一条依赖路径。

## 4. INT8 SmoothQuant 量化

本次实际使用以下参数。**已有的 `quantizedcheckpoint` 非空时会拒绝覆盖**；
若要重跑，请为 `CHECKPOINT` 指定一个新的目录，并在导出时使用相同变量。

```bash
CHECKPOINT="$ROOT/quantizedcheckpoint"
export CUDA_VISIBLE_DEVICES=0

trtedge quantize llm \
    --model-dir /workspace/R1_10B \
    --output-dir "$CHECKPOINT" \
    --quantization int8_sq \
    --experimental-alpamayo-int8 \
    --dtype fp16 \
    --device cuda \
    --text-dataset wikitext \
    --num-samples 32 \
    > "$ROOT/logs/quantize-int8-sq-text.log" 2>&1

status=$?
tail -n 25 "$ROOT/logs/quantize-int8-sq-text.log"
test "$status" -eq 0 || { echo "Quantization failed; stop before export."; return 1 2>/dev/null || exit 1; }
```

实验路径会：

1. 从配置指定的 VLM（默认 `Qwen/Qwen3-VL-8B-Instruct`）加载架构和 processor，
   **不下载该模型的基础权重**；语言和视觉权重均来自本地 Alpamayo。
2. 复用已有 tokenizer 构造逻辑添加轨迹 token，检查词表为 155697，
   `history=155684`、`future=155685`。
3. 把本地 `vlm.*` 权重暂时转换为标准 Qwen3-VL 命名；要求所有模型权重精确匹配。
4. 只进行文本校准，SmoothQuant 处理语言骨干；视觉和 LM head 量化器关闭。
5. 导出 INT8 checkpoint，恢复 `vlm.*` 前缀，合并原始动作组件；
   BF16 非量化权重转换成 FP16，原始文件不写入。
6. 保存 `hf_quant_config.json` 和 `experimental_quantization.json`，
   后者明确记录 `orin_validated=false`。

首次执行可能需要访问 Hugging Face 下载配置、tokenizer、processor 和 WikiText。
离线使用时，先准备匹配缓存；也可按
[校准数据指南](docs/source/developer_guide/customization/calibration-datasets.md)
设置 `EDGELLM_QUANT_DATASET_WIKITEXT` 指向本地 JSON/JSONL，
每条记录包含 `text` 字段。不能使用虚构文本冒充真实校准或精度验证。

成功日志包含：

```text
Alpamayo INT8 text calibration: wikitext, 32 batches
Smoothed 252 modules
```

视觉和 LM head 的 `not calibrated, skip smoothing` 警告源于它们没有参与校准；
它们保持非量化状态。任何 traceback 或非零退出码都必须先处理。
更大的 `--num-samples` 可以增加覆盖，但不替代驾驶多模态校准和任务评估。

## 5. 导出三路 ONNX

量化成功后，在上述同一个 Bash 会话执行。此步骤使用 CPU，隐藏 GPU：

```bash
export CUDA_VISIBLE_DEVICES=""

trtedge export \
    "$CHECKPOINT" \
    "$ROOT/quantizedonnx" \
    --dtype float16 \
    --max-kv-cache-capacity 4096 \
    > "$ROOT/logs/export-int8-sq.log" 2>&1

status=$?
tail -n 25 "$ROOT/logs/export-int8-sq.log"
test "$status" -eq 0 || { echo "ONNX export failed; inspect the full log."; return 1 2>/dev/null || exit 1; }
```

这里的 `--dtype float16` 是非量化运算/组件的 dtype，
**不会把 checkpoint 中的 INT8 权重恢复成全 FP16**。
请保留每个目录内的全部 `.onnx.data`、embedding、配置、tokenizer、
processor 和动作 sidecar，不要只复制 `model.onnx`。

## 6. 检查输出

```bash
/root/.venv-edgellm/bin/python - <<'PY'
from collections import Counter
from pathlib import Path
import onnx

root = Path("/workspace/tensorrt-edgellm-workspace/R1_10B/quantizedonnx")
for name in ("llm", "visual", "action"):
    path = root / name / "model.onnx"
    model = onnx.load(path, load_external_data=False)
    types = Counter(t.data_type for t in model.graph.initializer)
    ops = Counter(n.op_type for n in model.graph.node)
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                assert (path.parent / entry.value).is_file()
    onnx.checker.check_model(str(path))
    if name == "llm":
        assert types[onnx.TensorProto.INT8] == 252
        assert ops["QuantizeLinear"] == 252
        assert ops["DequantizeLinear"] == 504
    else:
        assert types[onnx.TensorProto.INT8] == 0
    print(name, "PASS", "INT8 weights:", types[onnx.TensorProto.INT8])
PY
```

本次实际结果：

| 检查 | 结果 |
|---|---|
| SmoothQuant | 252 个语言骨干 Linear |
| LLM ONNX | 252 个 INT8 initializer，252 个 Q、504 个 DQ |
| Visual / Action ONNX | 0 个 INT8 initializer |
| 三路 ONNX checker 和外部权重文件存在性 | 通过 |
| LLM runtime config 的 `kv_cache_dtype` | `fp16` |

ONNX checker 只检查图结构，不执行 TensorRT 自定义算子，也不证明数值精度。

## 7. 后续 Orin 验证

保持本次产物为实验性结果，直到在实际 Orin/JetPack/TensorRT 环境完成：

1. 使用相同版本 Edge-LLM 构建语言、视觉和动作引擎；
2. 确认 INT8 Q/DQ 图和全部 VLA 插件在该环境可构建；
3. 保持语言/动作 KV cache 容量为 4096，并核对动作引擎的输入契约；
4. 用相同图像、文本、历史轨迹对比 FP16 基线和 INT8 输出；
5. 测量峰值显存、延迟以及实际轨迹质量。

参考 [Alpamayo 流程](docs/source/user_guide/examples/vla/alpamayo.md)。
该示例的设备步骤面向 Thor，不能将其中的成功描述直接视为 Orin 验证。

## 8. 修改和回归测试

- [CLI 实验开关](tensorrt_edgellm/scripts/quantize.py)
- [Alpamayo INT8 适配](tensorrt_edgellm/quantization/alpamayo.py)
- [回归测试](tests/python-unittests/test_alpamayo_quantize.py)

```bash
cd /workspace
LLM_SDK_DIR=/workspace /root/.venv-edgellm/bin/python -m pytest \
    tests/python-unittests/test_alpamayo_quantize.py -q
```

不带 `--experimental-alpamayo-int8` 时，原有量化流程保持不变。
