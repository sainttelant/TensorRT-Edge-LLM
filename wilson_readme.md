<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TensorRT Edge-LLM：Jetson Orin 部署与 Alpamayo 移植分析

分析日期：2026-10-19。源码基线：`0f52705`，仓库 README 标示版本为 0.11.0。

第 1～8 节是上述日期的静态分析，不是 Orin 上板测试报告；当时没有执行模型下载、导出、Orin 构建或推理。第 9 节于 **2026-10-29** 补充了当前 x86_64 环境中实际完成的 Alpamayo 三组件导出及产物检查，**仍未在 Orin 构建或推理**。下文区分“仓库明确声明支持”“实际执行结果”和“工程可行性推断”。测试清单存在不等于本次验证通过，也不等于所有 Orin 内存规格均可运行。

## 1. 结论先行

| 问题 | 结论 | 证据与边界 |
|---|---|---|
| 本工程能否在 Jetson Orin 部署普通 LLM？ | **可以，是当前仓库明确支持的平台。** | 有 SM87 工具链、注意力内核、Orin LLM/VLM 测试清单。应选择适配的模型规模与精度。 |
| 不支持 NVFP4 是否意味着无法部署 LLM？ | **不是。使用 FP16、INT4 AWQ/GPTQ 或受支持的 INT8 SmoothQuant 路径。** | NVFP4 是一种权重/计算格式，不是整个框架的必选依赖。 |
| 能否直接让 Orin 跑现成 NVFP4 engine？ | **当前工程不支持，不能靠改架构参数绕过。** | 精度格式、算子实现、SM 和 TensorRT engine 均有限制。优先重新选择原始高精度权重，再量化/导出/构建。 |
| Alpamayo-R1-10B 是否被 NVFP4 卡住？ | **不是；其示例明确使用 FP16。** | 主要风险是约 10B 参数、多图视觉输入、额外动作 KV 缓存、动作专家重复计算，以及目标 TensorRT 的算子支持。 |
| Alpamayo 能否移植到 Orin？ | **AGX Orin 64GB 值得做离线原型；32GB 有条件尝试；16GB 及以下不建议直接尝试完整 FP16 链路。** | 这是容量与代码路径推断，不是成功案例或实时性能承诺。当前 Alpamayo 教程标注的是 Thor。 |
| 能否用于实际车辆控制？ | **本报告只支持离线研究可行性判断，不构成可直接接入车辆控制的结论。** | 必须另行完成模型许可、输入输出语义、精度、时延和系统级安全验证。 |

**推荐顺序：先在 Orin 跑通一个小型 FP16/INT4 LLM，再验证 Qwen3-VL 类 VLM，最后在 AGX Orin 64GB 上尝试 Alpamayo FP16、batch=1 的完整链路。**

## 2. 仓库架构及与 Orin 的关系

主部署链路为：

```text
Hugging Face checkpoint
  -> 可选量化（通常在 x86 GPU 主机）
  -> Python 导出 ONNX
  -> 在 Orin 构建 TensorRT engines
  -> C++ runtime 推理
```

相关模块：

| 层次 | 入口/目录 | 对部署的意义 |
|---|---|---|
| Checkpoint 配置及权重读取 | [tensorrt_edgellm/](tensorrt_edgellm/)、[checkpoint/](tensorrt_edgellm/checkpoint/) | 模型结构、权重命名和量化元数据必须受支持，不能只看参数量。 |
| 导出与量化 | [export.py](tensorrt_edgellm/scripts/export.py)、[quantization/](tensorrt_edgellm/quantization/) | 将受支持权重转换为部署图；导出成功不代表目标板可构建。 |
| TensorRT 构建 | [cpp/builder/](cpp/builder/)、[examples/llm/](examples/llm/)、[examples/multimodal/](examples/multimodal/) | 在目标 SM/软件栈上选择 tactic、构建引擎；构建峰值内存可能高于推理。 |
| 推理与 KV cache | [cpp/runtime/](cpp/runtime/)、[cpp/action/](cpp/action/) | 管理文本生成、视觉输入、KV 缓存和动作专家。 |
| CUDA/插件 | [cpp/kernels/](cpp/kernels/)、[cpp/plugins/](cpp/plugins/) | FP16/INT4/FP8/NVFP4 使用不同路径，不能认为所有目录都必须在所有设备执行。 |
| 架构构建选择 | [aarch64_linux_toolchain.cmake](cmake/aarch64_linux_toolchain.cmake)、[cpp/CMakeLists.txt](cpp/CMakeLists.txt) | `jetson-orin` 明确选择 `CMAKE_CUDA_ARCHITECTURES=87`，不是 Thor 的 `110a`。 |

仓库还有实验性 checkpoint-direct builder 和 Python server。首次验证 Orin、尤其 Alpamayo 时，建议先使用文档对应的 ONNX + C++ 流程，避免同时引入实验性前端变量。

## 3. Orin 支持度：不止文档宣称

### 3.1 平台和软件栈

当前 [官方支持矩阵](docs/source/user_guide/getting_started/support-matrix.md) 将 **Jetson Orin + JetPack 7.2 + CUDA 13.2 + 平台 TensorRT** 列为 Official，精度限定为 FP16、INT8、INT4。

这点不能套用旧版本印象：本次访问的 [NVIDIA JetPack 官方页面](https://developer.nvidia.com/embedded/jetpack) 也已说明 JetPack 7 支持 Orin 和 Thor。具体 CUDA/TensorRT 搭配仍以当前仓库支持矩阵和设备实际安装包为准。

注意区分：

- 工具链中 `jetson-orin` 的默认 CUDA 配置仍是 12.6，并保留 CUDA 12/13 不同目标目录分支。
- **保留旧工具链分支不等于当前版本正式认证 JetPack 6.x。** 如果设备不能升级，须单独验证完整依赖；必要时选择对应旧发行版，而不是只修改版本字符串。
- 当前 aarch64 wheel 的 glibc 下限是 2.39，并按 SM、CUDA、TensorRT 等信息精确选择 native payload。旧系统即使能 `pip install` 某个包，也不代表 wheel runtime 一定可用。
- 容器不能为 SM87 增加 FP8/NVFP4 硬件能力，也不能随意替换宿主 BSP/驱动兼容性。

具体安装与 Orin CMake 配置见 [installation.md](docs/source/user_guide/getting_started/installation.md)。

### 3.2 代码和测试配置证据

| 证据 | 可得出的结论 |
|---|---|
| [aarch64_linux_toolchain.cmake](cmake/aarch64_linux_toolchain.cmake) 的 `jetson-orin` 分支 | 原生存在 SM87 构建目标，无需伪装成 Blackwell。 |
| [cpp/CMakeLists.txt](cpp/CMakeLists.txt) 的 `FMHA_ALL_SM_VERSIONS` 包含 87 | 注意力内核构建配置考虑了 Orin。 |
| [cuteDslFMHAV2Runner.cpp](cpp/kernels/contextAttentionKernels/cuteDslFMHAV2Runner.cpp) 接受 SM87，限定相应 FP16 路径 | 并非只有 Blackwell FMHA 可用；具体 head size/mask 仍需满足实现约束。 |
| [decoderXQAJitCompiler.cpp](cpp/kernels/decodeAttentionKernels/decoderXQAJitCompiler.cpp) 的允许 SM 列表包含 87 | 解码注意力有 Orin 路径；FP8 允许列表则不包含 87。 |
| [attentionPlugin.cpp](cpp/plugins/attentionPlugin/attentionPlugin.cpp) 的 FP8 KV 支持列表排除 87 | 即使权重是 INT4，也不能顺带启用 FP8 KV cache。 |
| [l0_e2e_orin.yml](tests/test_lists/l0_e2e_orin.yml) | 配置了 InternVL3-1B INT4 AWQ 引擎构建和推理。 |
| [l1_pipeline_orin_llm.yml](tests/test_lists/l1_pipeline_orin_llm.yml) | 配置了 Qwen3-1.7B/4B、Qwen2.5 等 FP16/INT4 构建及评测。 |
| [l1_pipeline_orin_vlm.yml](tests/test_lists/l1_pipeline_orin_vlm.yml) | 配置了 Qwen3-VL-4B、InternVL3、Qwen2.5-VL 等 FP16/INT4，以及部分 INT8 测试。 |

这些证据足以支持“普通 LLM/VLM 在 Orin 上有明确工程基础”，但不能将其扩大为“模型列表中的任意模型、任意量化组合都已在 Orin 实测通过”。[支持模型文档](docs/source/user_guide/getting_started/supported-models.md) 本身也提示并非所有平台/精度组合都已验证。

## 4. 如何规避 NVFP4/FP8 限制

### 4.1 应选择的精度

| 格式/组件 | 当前 Orin 路线 | 说明 |
|---|---|---|
| FP16 权重与计算 | 推荐基线 | 最容易先建立正确性参照，但占内存大。 |
| INT4 AWQ / INT4 GPTQ | 适合压缩受支持 LLM | 常见为 W4A16，权重 4bit、激活仍为 FP16；不是 NVFP4，也不意味着全图都是 4bit。 |
| INT8 SmoothQuant | 按具体模型支持情况使用 | 需要对应导出和校准路径，不等同于任意 INT8 checkpoint。 |
| FP8 / MXFP8 权重或计算 | 不选 | 当前支持矩阵不支持 Orin 运行这些模型 engine。 |
| FP4 / NVFP4 权重或计算 | 不选 | 不能通过改 CMake、驱动开关或 engine 文件名使其成为 Orin 支持路径。 |
| FP8 KV / FP8 embedding / FP8 视觉塔 | 不选 | 量化模型要同时检查这些辅助组件，不能只看 backbone。 |
| BF16 源 checkpoint | 检查并转换到支持的部署路径 | 不能将“可以读取 BF16 权重”理解成“Orin 完整 BF16 部署已被本工程认证”；先以 FP16 为基线。 |

### 4.2 正确替代方式

1. 同一模型若有 FP16/BF16 原始 checkpoint，优先从原始权重导出 FP16，或量化为 INT4 AWQ/受支持 INT8。
2. 已有受支持 INT4 AWQ/GPTQ checkpoint 时，可以省略自行校准，但仍需核对模型结构、量化元数据及辅助组件。
3. 若只有 NVFP4 checkpoint，理论上可另做反量化/转换，但会保留已发生的量化误差，可能需要重写权重打包和元数据处理；**这不是本仓库已证明可用的一键 Orin 部署方案**。优先寻找原始权重。
4. 如果模型本身依赖未在 SM87 实现的专用算子，仅更换量化格式仍不够；必须继续核对模型结构与全部算子。
5. Thor/桌面 GPU 上生成的 TensorRT engine 不应直接搬到 Orin。传输 ONNX 和 sidecar 文件，在目标板上重新构建三方版本一致的 engines。

参考：[量化方法和 CLI](docs/source/user_guide/features/quantization.md)、[支持模型及 Orin 精度说明](docs/source/user_guide/getting_started/supported-models.md)。

低内存板卡还可考虑导出参数 `--externalize-weights int4_ffn`；MoE 另有 `int4_moe`。这是文档针对 INT4 构建内存的措施，**不是自动降低所有 runtime 内存，也不是 Alpamayo FP16 的现成解决办法**。

## 5. 普通 LLM：按内存规格选型

Orin 为 CPU/GPU 共享物理内存，不能把板卡标称 RAM 全部视为可用 GPU 显存。还需留出系统、摄像头、图像预处理、TensorRT execution context、KV cache 和应用内存。

理论权重下限：`参数数量 × 每参数字节数`。FP16 约 2 字节，INT8 约 1 字节，INT4 约 0.5 字节；实际还包含量化 scale、未量化层、embedding、LM head 和布局开销。

| 参数量 | FP16 理论权重 | INT4 理论权重 |
|---|---|---|
| 1B | 2 GB | 0.5 GB |
| 3B | 6 GB | 1.5 GB |
| 8B | 16 GB | 4 GB |
| 10B | 20 GB | 5 GB |

上表为十进制 GB，仅是权重下限，**不是实际峰值内存或容量承诺**。

| Orin 内存等级 | 建议起点（工程建议，非性能实测） |
|---|---|
| 4GB | 0.5B～1B INT4、batch=1、短上下文；容量很紧，先测完整峰值。 |
| 8GB | 1B～3B INT4 优先；不要仅凭 8B 的理论 INT4 权重就判断能稳定部署。 |
| 16GB | 3B INT4/部分 FP16，进一步尝试 7B～8B INT4；VLM 另算视觉内存。 |
| 32GB | 7B～8B INT4 较适合作为候选，也可评估 FP16；大 VLM 必须单独测峰值。 |
| 64GB | 更适合 8B/10B FP16、多模态和迁移验证，但仍不能保证实时性。 |

先选 Qwen2.5 小模型或 Qwen3-1.7B；若用新版混合结构模型，还要确保对应 CuTe DSL 内核启用并包含 SM87 artifact。不要把量化减少的权重占用，误认为 KV cache 和视觉开销也同比减少。

## 6. Alpamayo-R1-10B：逐项分析

对应教程：[alpamayo.md](docs/source/user_guide/examples/vla/alpamayo.md)。

### 6.1 实际运行的不是一个普通 10B 文本 LLM

```text
多摄像头图像 + 文本 + 16 个历史轨迹点
  -> visual engine
  -> LLM/VLM engine：视觉/轨迹 token + 文本生成 + 保留 KV
  -> 将分页 KV gather 为 action 使用的逐层 K/V 缓冲
  -> action engine：10 次 flow-matching denoise
  -> 64 个 (accel, kappa) 点
```

源码依据：

- [export.py](tensorrt_edgellm/scripts/export.py) 对 `alpamayo_r1` 有专门的权重重映射、视觉导出和动作专家导出。
- 视觉配置默认解析 `Qwen/Qwen3-VL-8B-Instruct`；**这描述的是当前导出器的配置解析路径，不代表可以用普通 Qwen 权重替换 Alpamayo 权重而保持驾驶能力。**
- [modeling_alpamayo_action.py](tensorrt_edgellm/models/alpamayo/modeling_alpamayo_action.py) 使用 `RotaryEmbedding`、`TensorScatter`、`Attention` 等 ONNX 原生域算子实现非因果动作注意力。
- [actionBuilder.cpp](cpp/builder/actionBuilder.cpp) 从 ONNX 的 K cache 输入维度读取固定容量，batch 是 TensorRT profile 的一部分。
- [alpamayo1ActionRunner.cpp](cpp/action/alpamayo1ActionRunner.cpp) 单独分配逐层 action K/V 缓冲，并从 LLM 分页 KV gather 数据，并非动作专家零成本共享原缓存。
- [alpamayo1ActionRunner.h](cpp/action/alpamayo1ActionRunner.h) 固定 `kNumDenoiseSteps=10`。有 64 个输出点并不等于动作专家只执行一次。

在所检查的动作导出、builder 和 runner 中，没有发现“必须为 SM110”或“必须 NVFP4”的显式门槛。因此有开展 Orin 移植验证的理由。**但原生 TensorRT Attention 的目标 SM tactic、ONNX parser 和算子融合能否成功，仍需在目标软件栈实际 build 才能确定。**

### 6.2 当前支持证据的边界

- 教程的构建和推理步骤明确标为 **Thor Device**。
- 教程明确写明 **Only FP16 is supported for Alpamayo export**。不要把通用量化 CLI 可接受 INT4 参数，推断为 Alpamayo 整体 INT4 已受支持。
- 本次搜索未在 `tests/test_lists/` 中找到 Alpamayo 端到端测试条目。其他位置虽有 VLA 类型、数据集与导出相关代码，也不能据此认定 Alpamayo 已通过 Orin 测试。
- Qwen3-VL 的 Orin 测试提供部分共用路径的积极证据，但无法覆盖 Alpamayo 专属动作头。
- [上游模型卡](https://huggingface.co/nvidia/Alpamayo-R1-10B) 给出的通用运行要求包含至少 24GB 显存；这是上游要求，不是 TensorRT Edge-LLM 在 Orin 的实测内存。

### 6.3 内存为什么是第一道门槛

按约 10B 参数估算，FP16 仅权重约 **20GB（18.6GiB）**。实际应统计导出后的三个组件及 sidecar 权重，不能把这个近似值当成精确总量。

传统全注意力 FP16 KV 的估算公式：

```text
KV bytes = 2(K 和 V) × 层数 L × batch B × 容量 S × KV heads Hkv × head_dim D × 2 bytes
```

当前导出器默认引用的 [Qwen3-VL-8B 配置](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json) 为 `L=36, Hkv=8, D=128`。仅用其作为量级参考：

- `B=1, S=4096`：一套完整 K/V 约 **0.5625GiB**。
- `B=6, S=4096`：一套约 **3.375GiB**。
- action runner 又按动作层数、heads 和容量分配另一套 dense K/V。**若动作层数及 KV 维度与上述一致**，两套量级分别约 1.125GiB 和 6.75GiB；实际必须读取导出的 `llm/config.json`、`action/config.json`。
- LLM 使用分页管理，真实分配还受页池策略影响；以上不是其所有内存的精确复刻。action 的 dense 分配则直接与配置上限相关。

另外还有视觉激活、DeepStack 特征、TensorRT workspace/context、ONNX/engine 读取缓冲和系统内存。构建时还可能同时保留 CPU/GPU 权重副本。

因此：**将输入 JSON 中 batch 改为 1，不一定就释放按 engine 最大 batch 预分配的内存；LLM 和 action 的构建 profile 都应改为 1。**

| 设备 | 完整 FP16 Alpamayo 判断 |
|---|---|
| Orin Nano 4GB/8GB、Orin NX 8GB/16GB | 权重本身已超出或显著挤压总内存，不建议直接部署；swap 不能提供等价 GPU 可用内存。 |
| AGX Orin 32GB | 有条件离线尝试，但视觉和构建峰值可能触发 OOM，不作为优先成功目标。 |
| AGX Orin 64GB | 最值得尝试的 Orin 规格；先验证 batch=1 全流程，再讨论速度。 |

### 6.4 时延与输入输出不能混淆

- 总时延包括多图预处理、visual、LLM prefill、逐 token decode、KV gather、10 次 action denoise，不应只报告文本 tokens/s。
- 上游主设置是 **4 摄像头 × 4 时刻 = 16 图**。减少摄像头、历史帧数或分辨率会改变模型输入分布，不是无损优化。
- 教程的视觉 profile 为 `160 / 18432 / 192`（min/total max/per-image max tokens）。这些数值不能简单用“16 × 192”替换：预处理、patch、merge、profile 统计口径需要核对实际 shape。
- Edge-LLM 当前输出 `output_trajectory` 是 **(加速度, 曲率)**，不是直接的 XYZ 路径或执行器控制量。上游卡还描述了位置和旋转输出；二者的后处理语义不能混用。
- 上游 action 配置有 64 个 waypoint、`dt=0.1s`，表示预测时间跨度约 6.4 秒，**不等于设备实现了 10Hz 推理或控制**。
- 压缩文本生成长度可能影响动作所条件化的 KV/语义；减少 denoise 次数属于模型行为修改，需要重新评估精度。

### 6.5 若希望用 INT4 让小内存 Orin 跑 Alpamayo

这是可研究的优化项目，**不是已经打通的部署步骤**：

1. 先在大内存设备建立 Alpamayo FP16 参照，保留正确的视觉、轨迹 token、KV 接口和输出语义。
2. 优先研究仅量化 VLM 文本线性层为 INT4，视觉塔、动作专家与 KV 暂保留 FP16。
3. 适配 `vlm.*` 权重命名、量化工具模型加载、特殊词表/embedding/LM head、量化元数据与导出器。
4. 用真实多摄像头和历史轨迹数据校准及评估。默认通用文本校准集不能证明驾驶动作精度保持。
5. 同时比较文本输出、KV 条件化后的动作输出、轨迹误差及长尾场景；不是只验证“有文字输出”。

即使成功，视觉和动作组件仍占内存；也不能把整模型大小直接按 FP16 的四分之一估计。

## 7. 建议执行路线

### 7.1 先确认目标板信息

在 Orin 上记录以下信息，勿使用当前 x86 工作站信息代替：

```bash
cat /proc/device-tree/model
cat /etc/nv_tegra_release
uname -m
nvcc --version
dpkg-query -W nvidia-jetpack libnvinfer10 libnvinfer-dev
free -h
df -h
```

包名按实际 TensorRT major 调整。另记录电源模式、散热、空闲内存和后台负载；推理期间用 `tegrastats` 观测共享内存、GPU 负载与温度。

### 7.2 构建 Orin runtime

以下来自当前 JetPack 7.2 Orin 文档配置；仅在实际 SDK 匹配时使用。若已有 build 目录，使用单独目录避免污染 Thor/x86 缓存：

```bash
export TRT_PACKAGE_DIR=/usr
git submodule update --init --recursive

cmake -S . -B build-orin \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR="$TRT_PACKAGE_DIR" \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUDA_CTK_VERSION=13.2 \
  -DENABLE_CUTE_DSL=ALL

cmake --build build-orin --parallel 2
export LD_LIBRARY_PATH="$TRT_PACKAGE_DIR/lib:${LD_LIBRARY_PATH:-}"
```

`--parallel 2` 是内存保守起点，可按资源调整。CuTe DSL artifact 必须对应 SM87，不能拿 SM110 产物冒充。若平台 TensorRT 库还在多架构目录，按实际 loader 路径补充环境。

普通模型先参考 [Quick Start](docs/source/user_guide/getting_started/quick-start-guide.md) 完成 **export → build → inference**，将命令中的 `build/` 对应改为 `build-orin/`。推荐 batch=1、较短上下文作为首轮基线。

### 7.3 Alpamayo 首轮实验：只改必要的部署上限

在 x86 主机按 [Alpamayo 教程](docs/source/user_guide/examples/vla/alpamayo.md) 完成 checkpoint 访问授权与 FP16 导出，保留全部 `onnx/llm`、`onnx/visual`、`onnx/action` 及 sidecar 文件。默认会解析关联 VLM 配置/processor，离线环境需提前准备相应资源。

在 Orin 上构建，首轮建议：

| 参数 | Thor 教程值 | Orin 首轮建议 |
|---|---|---|
| LLM `--maxBatchSize` | 6 | **1** |
| action `--maxBatchSize` | 6 | **1** |
| LLM `--maxInputLen` | 3424 | 首轮保留，记录实际 token 数再调整。 |
| 导出 `--max-kv-cache-capacity` / LLM `--maxKVCacheCapacity` | 4096 / 4096 | **保持一致**；容量必须覆盖 prompt、生成文本及动作追加 token 的需求。 |
| visual token profile | 160 / 18432 / 192 | 首轮保留语义有效设置；依据真实预处理 shape 收缩，不盲改。 |
| 输入与推理程序 | 16 图 + 历史轨迹，`action_inference` | 使用同一离线输入，batch=1；不要用普通文本推理替代。 |

动作 ONNX 的 KV 容量是导出时固定的。缩小容量时应重新导出并重新构建相关 engine，不仅修改 LLM CLI。

按顺序执行教程的 `llm_build`、`visual_build`、`action_build`，最后运行 `action_inference`。**本表只是候选实验配置，不代表这些命令已在 Orin 验证。**

### 7.4 验收与故障定位

| 阶段 | 最低验收 | 失败后优先检查 |
|---|---|---|
| Export | 三个 ONNX 组件及配置/processor/tokenizer 完整；无 FP8/NVFP4 组件混入 | checkpoint 类型、权重重映射、词表、关联 VLM 资源。 |
| Engine build | 三个组件均在 Orin 成功构建，记录版本及峰值内存 | parser 不支持算子、找不到 tactic、SM artifact、构建 OOM；不能把它们统称为量化问题。 |
| Runtime | 三个 engine 同时加载；输出文本与 64 个有限的 `(accel,kappa)` 对 | 内存峰值、KV 容量/shape、特殊轨迹 token、数据顺序。 |
| 正确性 | 固定输入、随机种子/噪声和解码配置，与 FP16 参照对齐；使用一致后处理计算轨迹误差 | 图像 processor、M-RoPE、KV gather、动作 denoise、精度变化。 |
| 性能 | 分别记录预处理、视觉、prefill、decode、action、整链路 p50/p95 和峰值内存 | 不用文本吞吐或单次热启动时延代替完整指标；持续负载检查降频。 |

遇到 32GB 构建 OOM，不应直接推断 64GB 也不可行；遇到 TensorRT 原生算子不支持，则增加内存也无济于事。应保存具体失败组件和错误消息，分开解决容量与兼容性问题。

## 8. 最终建议

1. **普通 LLM：值得直接部署。** Orin 的 FP16/INT4 工程路径明确，优先选择已列入 Orin 测试配置的小型模型，完成闭环后扩大模型。
2. **规避 NVFP4：换权重格式与受支持算子路径，而不是伪造硬件。** 同时禁用/避开 FP8 KV、视觉塔和 embedding。
3. **Alpamayo：先做 AGX Orin 64GB、FP16、batch=1 的离线可行性实验。** 当前没有足够证据承诺 Orin 直接可用，更没有实时性保证。
4. **32GB 及更小设备：先做容量评估；完整 FP16 在 16GB 以下不适合作为直接部署目标。** INT4 Alpamayo 应作为独立适配/精度验证项目。
5. **若目标是可交付的多摄像头低时延动作系统，Thor 是更符合当前教程的基线。** 是否接受 Orin 取决于实测整链路时延、内存与任务精度，不应只比较 TOPS 或是否能加载权重。

### 主要依据

- 仓库：[支持矩阵](docs/source/user_guide/getting_started/support-matrix.md)、[支持模型](docs/source/user_guide/getting_started/supported-models.md)、[安装](docs/source/user_guide/getting_started/installation.md)、[量化](docs/source/user_guide/features/quantization.md)、[Alpamayo 教程](docs/source/user_guide/examples/vla/alpamayo.md)。
- 外部公开资料（访问日期同本文）：[JetPack](https://developer.nvidia.com/embedded/jetpack)、[Alpamayo 模型卡](https://huggingface.co/nvidia/Alpamayo-R1-10B)、[Alpamayo 配置](https://huggingface.co/nvidia/Alpamayo-R1-10B/blob/main/config.json)、[Qwen3-VL-8B 配置](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json)。
- 模型卡当前标示 OpenMDW-1.1，并说明非商业用途及商业授权条件；部署前应另行审查具体许可。仓库软件许可不自动覆盖模型权重与数据许可。

## 9. 具体操作：完整导出 Alpamayo，再部署到 Orin

> 验证日期：2026-10-29，源码基线：`c6fed0c`。
> **本次已实际完成：checkpoint 分片检查 → LLM / visual / action 完整 FP16 导出 → ONNX 与运行时 sidecar 检查。**
> 第 9.6～9.7 节是按当前源码整理的目标板候选命令，未执行，不能据此认定 Orin 已跑通。

### 9.1 模型导出：本机实际执行成功的命令

先纠正原命令中的两个问题：

1. 本环境没有 `trtedge` 可执行文件；当前安装的官方入口是 `tensorrt-edgellm-export`。除非另行安装了对应包装器，不要使用 `trtedge export`。
2. 原始 checkpoint 实际位于 `/workspace/R1_10B`。切换到 `/workspace/tensorrt-edgellm-workspace` 后，单写 `R1_10B` 会指向错误位置。**输入模型目录和输出工作目录要分开定义。**

本次使用全部 5 个 safetensors 分片，共 1,166 个索引 tensor、22,157,195,208 字节；没有下载另一份模型权重，也没有改动原始 checkpoint。

```bash
export REPO_DIR=/workspace
export WORKSPACE_DIR=/workspace/tensorrt-edgellm-workspace
export MODEL_NAME=R1_10B
export MODEL_DIR="$REPO_DIR/$MODEL_NAME"
export ONNX_DIR="$WORKSPACE_DIR/$MODEL_NAME/onnx"
export LOG_DIR="$WORKSPACE_DIR/$MODEL_NAME/logs"

mkdir -p "$LOG_DIR"
test -s "$MODEL_DIR/config.json"
test -s "$MODEL_DIR/model.safetensors.index.json"
command -v tensorrt-edgellm-export

# 本次使用 CPU 导出，不占用共享机器的 GPU。
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Bash 内置 time；无需安装 /usr/bin/time。新一次运行使用新日志名。
LOG_FILE="$LOG_DIR/export-$(date +%Y%m%d-%H%M%S).log"
TIMEFORMAT=$'Elapsed: %3R seconds\nUser CPU: %3U seconds\nSystem CPU: %3S seconds'
{
  time tensorrt-edgellm-export \
    "$MODEL_DIR" \
    "$ONNX_DIR" \
    --dtype float16 \
    --max-kv-cache-capacity 4096
} > "$LOG_FILE" 2>&1
status=$?
tail -n 25 "$LOG_FILE"
printf 'Exporter exit status: %s\n' "$status"
test "$status" -eq 0
```

完整导出不要添加 `--skip-llm`、`--skip-visual`、`--skip-action` 或限制组件的 `--components`。
日志中的 `thinker` 对应输出目录 `llm/`，不是少导出了一个组件。

**实际结果：退出码 0，耗时 185.103 秒，三个组件均显示完成。** 这是本次机器上的一次导出耗时，不是推理耗时或性能承诺。本次未测进程峰值内存。
运行前本机可用主机内存约 101GiB、磁盘余量约 411GB；这些是环境记录，不是导出的最低资源要求。

本次输出已保留在 [R1_10B/onnx](tensorrt-edgellm-workspace/R1_10B/onnx/)。
需要重复实验时，建议把 `ONNX_DIR` 改为新的目录，避免混合新旧产物。

### 9.2 Python 环境与外部依赖

本次实际使用 `/usr/bin/python`，Python 3.12.3；console script 使用同一解释器。

| 包 | 实际版本 |
|---|---|
| tensorrt-edgellm | 0.11.0，当前工作区源码 |
| torch | 2.13.0，运行时标识 `2.13.0+cu130` |
| torchvision | 0.28.0，运行时标识 `0.28.0+cu130` |
| transformers | 5.14.1 |
| onnx | 1.19.0 |
| onnxscript | 0.7.1 |
| safetensors | 0.8.0 |
| onnx-graphsurgeon | 0.6.1 |

之前 `--help` 出现的 `operator torchvision::nms does not exist` 已定位为 PyTorch/torchvision 不匹配：
旧 torchvision 为面向 PyTorch 2.10.0a0 的 `0.25.0a0`，与已安装的 PyTorch 2.13.0 不兼容。
本环境使用以下命令修复，并验证了 NMS、Gemma4 配置导入和 CLI help：

```bash
/usr/bin/python -m pip install --no-deps "torchvision==0.28.0"
```

这条修复只适用于已确认的上述版本组合，不是所有 NVIDIA 容器的通用升级命令。不要为了修复 warning 随意升级项目锁定的 torch/transformers。

**原始 Alpamayo 目录只有权重和配置，不包含全部 tokenizer / processor。**
导出器会访问 `Qwen/Qwen3-VL-8B-Instruct` 获取 VLM 配置、tokenizer、processor 等小文件；LLM、视觉和 action 的权重仍来自本地 Alpamayo。
本次解析到的 Qwen revision 为：

```text
0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
```

因此，“checkpoint 在本地”不等于“首次导出可以完全断网”。离线迁移需另行保留并验证关联 Hugging Face 缓存，不能只拷贝 5 个权重分片。
本次是联网导出，未执行离线复验。

### 9.3 实际生成了哪些文件

主要目录结构：

```text
tensorrt-edgellm-workspace/R1_10B/
├── logs/
│   ├── export-20261029.log
│   └── validate-20261029.log
└── onnx/
    ├── llm/
    │   ├── model.onnx
    │   ├── model.onnx.data
    │   ├── config.json
    │   ├── embedding.safetensors
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   └── chat_template.model
    ├── visual/
    │   ├── model.onnx
    │   ├── model.onnx.data
    │   ├── config.json
    │   ├── preprocessor_config.json
    │   ├── processor_config.json
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   └── chat_template.jinja
    └── action/
        ├── model.onnx
        ├── model.onnx.data
        └── config.json
```

本次文件大小如下，单位为十进制 MB/GB：

| 组件 | ONNX 图 | 外部权重 `model.onnx.data` | 额外主要数据 |
|---|---:|---:|---|
| LLM | 1.36 MB | 15.17 GB | embedding 1.28 GB |
| Visual | 2.28 MB | 1.15 GB | processor / tokenizer 等 |
| Action | 4.15 MB | 4.56 GB | action 配置 |

整个 ONNX 目录共 **22,188,750,067 字节，约 22.19 GB**。
导出末尾打印的 ONNX MB 数只包含图文件，不包含数 GB 的 external data。
**只传输三个 `.onnx` 文件是错误的，必须保留完整目录及相对路径。**

### 9.4 导出后验证：本次已执行的检查

本次不仅检查了命令返回码，还执行了以下检查：

| 检查项 | 实际结果 |
|---|---|
| 三个 `model.onnx` 的 `onnx.checker.check_model(path)` | 全部通过 |
| 外部 tensor 数据文件存在性、offset/length 不越界 | 全部通过 |
| tokenizer 词表 | 155,697，与 checkpoint / LLM config / embedding 一致 |
| embedding | FP16，`[155697, 4096]` |
| 原始配置中的 6 个轨迹 special token ID | 全部与生成的 tokenizer 一致 |
| `<i0>` / `<i3999>` | 151669 / 155668 |
| `<\|image_pad\|>` | 151655，与 LLM / visual config 一致 |
| LLM / action KV 合约 | 均为 36 层、8 个 KV heads、head dim 128 |
| Action KV shape | 每层 K/V 为 `[batch_size, 8, 4096, 128]` |
| Action 输入/输出 | `noise_trajectory` / `denoised_trajectory` 均为 `[batch_size, 64, 2]` |
| Action config | `max_kv_cache_capacity=4096`，`n_diffusion_tokens=64` |
| Native chat renderer 标记 | `llm/chat_template.model` 内容为 `alpamayo` |
| Visual processor sidecars | 存在且 JSON 可解析 |

图中实际使用 ONNX opset **24**：

- LLM：36 个 `trt_edgellm::AttentionPlugin`；
- Visual：27 个 `trt_edgellm::ViTAttentionPlugin`；
- Action：36 个 `trt::Attention`、72 个 `trt::RotaryEmbedding`、72 个 `trt::TensorScatter`。

**验证边界：** ONNX checker 检查图结构和标准 schema，不验证这些自定义 domain 算子的 TensorRT 实现，也没有运行 tensor 数值比较。Action 使用 `trt` domain 的原生 TRT 路径，不能把它当作可直接由任意 ONNX Runtime 执行的纯标准 ONNX。
是否能在目标板解析、选到 tactic、加载并给出正确结果，仍需完成 build → inference。

复查基本产物可运行下面的轻量命令；它不把数 GB 的 external data 全量读入 Python：

```bash
export ONNX_DIR=/workspace/tensorrt-edgellm-workspace/R1_10B/onnx
/usr/bin/python - <<'PY'
import json
import os
from pathlib import Path

import onnx
from safetensors import safe_open
from tokenizers import Tokenizer

root = Path(os.environ["ONNX_DIR"])
for component in ("llm", "visual", "action"):
    path = root / component / "model.onnx"
    onnx.checker.check_model(str(path))
    config = json.loads((path.parent / "config.json").read_text())
    model = onnx.load_model(str(path), load_external_data=False)
    print(component, "PASS", "nodes:", len(model.graph.node))

tokenizer = Tokenizer.from_file(str(root / "llm/tokenizer.json"))
assert tokenizer.get_vocab_size() == 155697
assert tokenizer.token_to_id("<|image_pad|>") == 151655
assert tokenizer.token_to_id("<|traj_history|>") == 155684
assert tokenizer.token_to_id("<|traj_future|>") == 155685
assert (root / "llm/chat_template.model").read_text().strip() == "alpamayo"
with safe_open(str(root / "llm/embedding.safetensors"), framework="pt") as f:
    embedding = f.get_slice("embedding")
    assert embedding.get_shape() == [155697, 4096]
    assert embedding.get_dtype() == "F16"
action = json.loads((root / "action/config.json").read_text())
assert action["builder_config"]["max_kv_cache_capacity"] == 4096
assert action["n_diffusion_tokens"] == 64
assert (root / "visual/preprocessor_config.json").is_file()
print("Alpamayo export artifacts: PASS")
PY
```

本次更详细的检查记录保留在：
[导出日志](tensorrt-edgellm-workspace/R1_10B/logs/export-20261029.log)、
[验证日志](tensorrt-edgellm-workspace/R1_10B/logs/validate-20261029.log)。

### 9.5 导出日志中的 warning 如何判断

这次导出并非“零 warning”，以下是实际出现的提示及本次判断依据：

| 提示 | 本次判断 |
|---|---|
| Transformers 不认识 `alpamayo_r1`，回退读取原始 config | Edge-LLM 后续从关联 Qwen VLM 配置解析结构，最终导出正确的 36 层 LLM/action；不需要按提示盲目升级 Transformers。 |
| 部分 HF 文件 HEAD 请求返回 404 | Transformers 探测可选文件名，随后加载实际存在的文件；不能仅凭 404 判断失败，也不能在最终 sidecar 缺失时忽略。 |
| 原始目录找不到 `<\|image_pad\|>` | 原始目录无 tokenizer；最终生成 tokenizer 和 LLM/visual config 中该 ID 均为 151655，已额外核对。 |
| 原始目录找不到 processor 文件 | 后续 Alpamayo 专用步骤从 Qwen processor 生成并保存；最终 `visual/preprocessor_config.json` 已检查。 |
| ModelOpt 的 Transformer Engine 扩展导入失败 / undefined symbol | 环境中仍有可选扩展与 PyTorch 的兼容问题；本次 FP16 导出未依赖该扩展并成功完成。不能把这次成功解释成量化 / Transformer Engine 功能已正常。 |
| Dynamo 导出属性或动态轴重名 warning | 本次未阻止导出；实际 I/O、外部权重及 checker 检查通过，动态 batch 的 TensorRT 执行仍待验证。 |

判断标准是**命令退出码、三个组件与 sidecar 完整性、接口检查和后续引擎验证**，不是只搜索日志中的某个单词。

### 9.6 传输和 Orin engine 构建

在 Orin 上将仓库和模型产物放到目标目录。需要传输完整 `onnx/`，不能遗漏 `.data`、embedding、tokenizer、processor 或 config。使用 checkpoint-backed 权重时还需要原始 checkpoint；本次 ONNX 构建的 engines 使用内嵌权重和 sidecar，不要求原始 checkpoint。输入 JSON 和其中引用的真实多相机图片也需要放到 Orin。

**传输内容：**

1. 完整 `onnx/` 目录。
2. 如果组件配置含非空 `checkpoint_weight_bindings`，传输原始 checkpoint；[action_inference.cpp](examples/multimodal/action_inference.cpp) 通过 `--checkpointDir` 加载这些权重。本次内嵌权重 engines 无需此步骤。
3. 真实的多相机图像、历史轨迹和输入 JSON。

示例：在 x86 主机设置目标后传输。以下用户名和地址必须替换为实际值：

```bash
export TARGET=user@orin-host
export REMOTE_ROOT=/home/user/tensorrt-edgellm-workspace/R1_10B
ssh "$TARGET" "mkdir -p '$REMOTE_ROOT'"
scp -r /workspace/tensorrt-edgellm-workspace/R1_10B/onnx "$TARGET:$REMOTE_ROOT/"
scp -r /workspace/R1_10B "$TARGET:$REMOTE_ROOT/checkpoint"
```

目标目录不存在时，第二条 `scp` 将 checkpoint 内容复制到该目录。重复传输前检查目标，避免意外嵌套成 `checkpoint/R1_10B/`。
后续 `MODEL_DIR` 应直接包含 `config.json` 和 safetensors 分片。

在仓库根目录运行 [build_orin.bash](build_orin.bash)。脚本会检查 Orin、CUDA、TensorRT、ONNX 文件和必需子模块，配置 SM87，并依次编译程序、构建 LLM / visual / action engines。默认模型路径是 `tensorrt-edgellm-workspace/R1_10B`；如实际位置不同，通过环境变量覆盖：

```bash
cd /home/nvidia/disklarge/TensorRT-Edge-LLM
./build_orin.bash --check-only
./build_orin.bash
```

脚本默认使用 `MODEL_ROOT=$REPO_DIR/tensorrt-edgellm-workspace/R1_10B`、`TRT_PACKAGE_DIR=/usr`、系统 CUDA，并将构建日志写入 `$MODEL_ROOT/logs/`。可通过 `MODEL_ROOT`、`ONNX_DIR`、`ENGINE_DIR`、`MODEL_DIR`、`BUILD_DIR`、`TRT_PACKAGE_DIR`、`CUDA_DIR` 和 `BUILD_JOBS` 环境变量覆盖路径或并行度。builder 参数固定为本次导出的 batch=1 配置；LLM 的 KV cache 容量为 4096，与 action 配置一致。

本次已在 Orin 上完成三个 engine 的构建，产物位于 `$MODEL_ROOT/engines/`。不要将其他 GPU 架构上生成的 TensorRT engine 用于 Orin。

### 9.7 完整 action inference（需要真实输入）

按 [Alpamayo 示例](docs/source/user_guide/examples/vla/alpamayo.md) 准备输入 JSON，将所有图片路径替换为 Orin 上的真实文件，并保留正确的相机/时间顺序和历史轨迹。首次验收不要使用空图片、随意伪造的轨迹或仅文本输入。

确认输入 JSON 和图片存在后，在仓库根目录运行。本次内嵌权重 engines 不要求 checkpoint；只有使用 checkpoint-backed 权重时，才需要通过 `MODEL_DIR` 指定包含 `config.json` 和 safetensors 分片的原始 checkpoint 目录：

```bash
cd /home/nvidia/disklarge/TensorRT-Edge-LLM
INPUT_FILE=/path/to/input_action.json ./build_orin.bash --inference
```

`--inference` 会先重新构建 engines，再执行完整 action inference。输入、输出和 checkpoint 路径可用 `INPUT_FILE`、`OUTPUT_FILE`、`MODEL_DIR` 覆盖；默认输出为 `$MODEL_ROOT/output_action.json`，profile 和推理日志保存在 `$MODEL_ROOT/` 与 `$MODEL_ROOT/logs/`。

最低验收要求：

- 三个 engine 能在同一进程加载，完整 VLM → action 路径成功；
- 输出包含文本及 64 个有限的 `(accel, kappa)` 对；
- 与参照使用一致输入、初始噪声和后处理，另做数值比较；相同 seed 不保证跨 RNG 实现产生同样噪声；
- 记录视觉、prefill、decode、action、总时延与内存，而不是只报告文本 tokens/s。

2026-11-02 已使用 [VLA_tests/input_action.json](VLA_tests/input_action.json) 在 Orin 上完成 3 个 clip 的离线冒烟推理，三个请求均成功输出非空文本及 64 个有限的 `(accel, kappa)` 对。结果位于 [output_action.json](tensorrt-edgellm-workspace/R1_10B/output_action.json)，profile 位于 [profile_action.json](tensorrt-edgellm-workspace/R1_10B/profile_action.json)。本次未提供原始 checkpoint；配置不含 `checkpoint_weight_bindings`，运行时直接使用内嵌权重和 sidecar。

**最终状态：导出与静态产物验证、Orin engine 构建和离线流程冒烟推理已通过；动作准确性与真实性能验收尚未完成。** 测试轨迹沿用示例，未与视频对应，相机布局及传感器时间同步也未验证，不能用于动作准确性判断或车辆控制。