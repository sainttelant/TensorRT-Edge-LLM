<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# TensorRT Edge-LLM 算子与 Blackwell 加速分析：以 VLA / Action 模型为重点

> 分析日期：2026-10-29。源码基线：`c6fed0c` 及当前工作区。
>
> 本文是基于源码、项目文档和 NVIDIA 官方资料的工程分析，不是性能实测报告。
> 文中的计算量、内存容量和加速上界属于推导；没有执行模型导出、引擎构建或设备 benchmark。
> **“Blackwell 硬件支持”“仓库提供算子”“某个 VLA 模型已经接入”是三个不同层次。**

## 阅读导航

- 第 1～3 节：结论、工程分层、Alpamayo R1 与 π0.5 的真实执行路径。
- 第 4～5 节：重要算子的数学含义、瓶颈，以及 VLA 特有的优化点。
- 第 6～8 节：Blackwell 架构差异、NVFP4 与 W4A16、当前代码的接入边界。
- 第 9～11 节：优化优先级、性能分析方法、数值与端到端验收。
- 第 12～13 节：源码阅读地图与官方参考资料。

---

## 1. 先给结论

对于这个工程里的 VLA，最值得优先研究的不是“如何把所有算子换成 FP4”，而是：

1. **按执行频率拆分计算。** 视觉和 prefix 每次观测计算一次；action expert 每个去噪步计算一次；只依赖时间步的条件可以进一步预计算。
2. **复用 prefix KV，但每步重算 action KV。** 观测前缀不变，去噪状态会变；二者不能一起缓存。
3. **针对小 batch、短 action chunk 优化 GEMM。** Action 不是普通 LLM 的单 token decode，也不是大规模 prefill。
4. **减少循环中的搬运与 launch。** CUDA Graph、融合 RoPE/KV 写入、设备端 Euler 更新，通常比盲目增大 tile 更值得先测。
5. **准确选择 attention 语义。** 视觉、π0.5 prefix、action chunk 都有非因果 attention；不能直接替换成普通 causal FlashAttention。
6. **再做分模块量化。** 当前 Alpamayo 指南和 π0.5 实现均以 FP16 为支持路径；仓库存在 NVFP4 kernel，不代表 VLA 已支持 NVFP4。
7. **以整次动作生成和正确性为目标。** 看观测到 action chunk 的 P50/P95/P99、显存和功耗，不能只看 GEMM TFLOPS 或文本 tokens/s。

尤其要避免三个误区：

- **Action expert 不等于 MoE routed expert。** 本文两条 VLA 的 action transformer 都是独立的 dense 网络；不能因为名字里有 expert，就套用 MoE 的路由优化结论。
- **Blackwell 不等于单一 SM 架构。** Thor、B200、RTX 50 系列和 GB10 的指令路径、内存系统与 kernel 分派并不相同。
- **动作数值接近不等于控制任务成功。** 数值回归、仿真闭环评估、部署安全约束是不同验证层。

## 2. 这个“大工程”中，算子处于哪一层

```text
Hugging Face checkpoint / 模型配置
    ↓
Python 模型实现、权重读取与重排
    ↓
ONNX：原生算子 + TensorRT Edge-LLM 自定义算子
    ↓
TensorRT builder
    ├─ 原生层的 tactic 选择、融合、内存规划
    └─ Edge-LLM Plugin 的形状检查、分派和专用 kernel
    ↓
C++ runtime
    ├─ 视觉 / prefix / LLM 引擎
    ├─ KV cache / batch / stream / CUDA Graph
    └─ action 循环、采样、后处理
```

这里有三种容易混淆的“算子”：

| 层次 | 例子 | 分析重点 |
|---|---|---|
| 数学算子 | GEMM、softmax、RMSNorm、Euler 更新 | 公式、张量形状、计算量、精度 |
| 图算子 / Plugin | ONNX Attention、AttentionPlugin、NVFP4 GEMM Plugin | 导出语义、布局、动态维度、量化 ABI |
| CUDA kernel | XQA、CuTe DSL FMHA、Marlin、Euler kernel | tile、访存、流水线、寄存器、launch 开销 |

**一个 Python 层不一定对应一个 CUDA kernel。** TensorRT 可以把多个图节点融合，也可能为布局转换生成额外 kernel。
因此，“源码出现 FP16 Linear”不能单独证明引擎最终使用哪种 Tensor Core 指令，必须结合 engine inspector 和 GPU profile。

主要入口：
[导出 CLI](../tensorrt_edgellm/scripts/export.py)、
[C++ runtime 设计](source/developer_guide/software-design/llm-inference-runtime.md)、
[CuTe DSL 构建与 artifact 说明](../kernelSrcs/README.md)。

## 3. 两条 VLA 路径：先理解计算图，再谈优化

### 3.1 Alpamayo R1：VLM 语言生成之后，运行 action flow matching

```text
多相机图像 + 文本 + 历史轨迹
    ↓
Visual encoder → 视觉 token
    ↓
VLM prefill → 自回归语言生成 → VLM paged KV
    ↓
按 action batch 收集并转换为 head-major K/V
    ↓
初始化连续动作噪声 x
    ↓
重复 10 次：
    动作与时间 Fourier 编码 → 输入 MLP / LayerNorm
    → action transformer：QK-norm + RoPE + 非因果 attention + MLP
    → 输出投影 → Euler 更新
    ↓
未来动作序列：(accel, kappa)
```

源码对应：
[action 模型](../tensorrt_edgellm/models/alpamayo/modeling_alpamayo_action.py)、
[action runner](../cpp/action/alpamayo1ActionRunner.cpp)、
[runner 配置](../cpp/action/alpamayo1ActionRunner.h)、
[官方项目流程](source/user_guide/examples/vla/alpamayo.md)。

已确认的实现事实：

- Action 图包含**一次**去噪迭代，C++ runner 的 `kNumDenoiseSteps` 为 10。
- 当前 R1 路径使用 64 个 waypoint；runner 从引擎形状取得实际 waypoint 数，不能把 64 当作所有 action 模型的通用约束。
- 输出是每个 waypoint 的两个动作通道 `(accel, kappa)`，不是直接输出 `[x,y,z]` 空间点；历史轨迹输入与动作输出的语义不同。
- Action attention 使用 TensorRT 原生路径，对应 `RotaryEmbedding`、`TensorScatter`、`Attention`，不是直接调用 π0.5 的 XQA action 路径。
- Attention 覆盖有效的 VLM prefix 和整个 action chunk，使用长度 mask 屏蔽 cache 中无效位置，而非逐 token causal mask。
- 从 paged KV 到连续 head-major K/V 的 gather 在循环前按层执行一次，不是每个去噪步重新 gather。
- 时间区间按 `0 → 1` 排列；Euler 更新已经包含在 action 导出图中。
- 每步仍有两个时间标量的 H2D copy，以及把更新结果回写下一步输入的 D2D copy。
- 此 runner 的这段 action 循环没有 π0.5 那样显式的整循环 CUDA Graph capture；不能从 LLM decode 支持 Graph 推导 action 也支持。

因此总延迟应拆成：

```text
T_R1 =
    T_输入预处理
  + T_视觉
  + T_VLM_prefill
  + T_语言decode
  + T_KV_gather
  + 10 × T_action_step
  + T_输出回传与后处理
```

**优化 action 前先测语言 decode 占比。** 如果语言生成是主要耗时，仅优化 action GEMM 的收益会受 Amdahl 定律限制。
但减少语言生成长度可能改变 action 所依赖的上下文，不是无需验证的性能开关。

### 3.2 π0.5：prefix 仅产生 KV，不进行语言自回归生成

```text
相机图像 → SigLIP vision → visual features
                                ↓
任务文本 / 按 policy 编码的状态 → token embedding
                                ↓
compact prefix → PaliGemma prefix tower → 每层 prefix KV
                                                  ↓
时间步 schedule → cond → 预计算 AdaRMS modulation   ↓
                                                  ↓
噪声 x₀ → 重复 N 次 action expert + 设备端 Euler 更新
                                                  ↓
归一化 action chunk → 按 policy 转换为 robot_actions
```

源码对应：
[视觉模型](../tensorrt_edgellm/models/pi05/modeling_pi05_visual.py)、
[prefix 模型](../tensorrt_edgellm/models/pi05/modeling_pi05_prefix.py)、
[action 模型](../tensorrt_edgellm/models/pi05/modeling_pi05_action.py)、
[设计文档](source/developer_guide/models/pi05.md)。

当前模型中的关键形状：

| 组件 | 实现中的主要配置 | 运行频率 |
|---|---|---|
| Visual | SigLIP，27 层，hidden 1152，16 heads，head dim 72；224×224 / patch 14 → 每视图 256 tokens | 每次请求的有效相机视图 |
| Prefix | 18 层，hidden 2048，8 query heads，1 KV head，head dim 256，MLP intermediate 16384 | 每次请求一次 |
| Action expert | 18 层，hidden 1024，8 query heads，1 KV head，head dim 256，MLP intermediate 4096 | 每个去噪步一次 |
| Cond | 时间 embedding 与 37 处 AdaRMS 的 modulation 投影 | 当前 runner 在步数或 batch 改变时重算 |
| Action 输出 | 网络 padded width 为 32，policy 再裁切为机器人维度 | 每次请求一次 |

注意：action 的 `8 × 256 = 2048` 是 Q 投影宽度，**不等于 hidden size 1024**。估算 GEMM 时应读真实投影形状，不能默认 `heads × head_dim == hidden_size`。

当前 policy 的 chunk 长度：

| Policy | Action horizon | 机器人输出维度 | 相机约束 |
|---|---:|---:|---|
| `pi05_libero` | 10 | 7 | 两个必需视图 |
| `pi05_droid` | 15 | 8 | exterior 与 left wrist |
| `pi05_aloha` | 50 | 14 | `cam_high` 必需，两路 wrist 可选 |

**Action horizon、去噪步数、机器人控制周期是三个不同概念。**
一次生成 50 个未来动作不代表模型做了 50 次去噪，也不代表必须等整个 chunk 执行完才重新观测。

π0.5 的额外优化已经落地：

- Prefix 不输出文本，没有每步 LM head / top-k / top-p sampling；只保留 action 需要的 KV。
- Visual 和 prefix 使用原生非因果 attention，action 使用共享 `AttentionPlugin`。
- Action 以 `kDiffusionDenoise` phase 进入 tree-shaped XQA 路径，all-ones action mask 允许 chunk 内互相可见。**复用 tree kernel 不等于运行 speculative decoding。**
- Prefix KV 在请求内固定，action suffix KV 每步覆盖同一段位置，不会随去噪步数持续增长。
- GPU 上执行 `x ← x + dt × v`，时间方向为 `1 → 0`，即 `dt = -1/N`。
- `--cudaGraph` 可捕获整个去噪循环；batch、步数、KV 地址变化时会使已有 graph 失效。
- Cond 默认从循环中移出；固定模型实例、batch 和 schedule 下可以跨请求复用。

两套模型时间方向不同，来自各自训练与速度场约定；不能直接复制一套 Euler schedule 到另一套。

### 3.3 “Action expert”为什么不是 MoE expert

MoE 是一个 token 通过 router 选择若干专家：

```text
token → router/top-k → 多个 expert MLP → 加权合并
```

这里的 VLA action expert 是：

```text
动作 token 序列 → 完整 dense transformer → 连续动作速度场
```

两者都大量使用 GEMM，但前者还需要路由、token 重排、分组 GEMM、scatter-reduce。
如果 profile 没有经过 MoE Plugin，那么优化 NVFP4 MoE Marlin 测试对应的 kernel，不会自动加速 R1 / π0.5 的 action head。

## 4. 重要算子总览：做什么、瓶颈在哪里

记号：

- `B`：请求 batch；`S`：prefix 长度；`H`：action horizon；`R`：去噪步数。
- `D`：hidden width；`F`：MLP intermediate width；`d`：head dimension。
- `n_q / n_kv`：query / KV head 数；`L`：层数。
- GEMM 记为 `Y[M,N] = X[M,K] × W[N,K]^T`。

### 4.1 全工程算子地图

| 算子 | 数学 / 工程职责 | 常见瓶颈 | 对 VLA 的意义 |
|---|---|---|---|
| 图像 resize、normalize、patch embedding | 像素变为视觉 token；patch 常由 strided Conv2D 实现 | 图像处理、copy；patch projection 的 GEMM/卷积 | 多相机、多帧会放大成本 |
| Embedding gather、prefix 拼接 | 按 token ID 读取向量，拼接视觉与文本特征 | 非连续读、额外 copy、CPU 调度 | 不应每个去噪步重复 |
| Q/K/V、O projection | Attention 前后的 dense GEMM | 小 M 权重带宽，大 M 算力 / 数据流水线 | 每个 action 层、每个去噪步重复 |
| QK-norm、RMSNorm、LayerNorm | 归一化、稳定激活尺度 | reduction、带宽、短 kernel launch | 单次很小，但循环次数多 |
| RoPE / MRoPE | 位置编码，旋转 Q/K | 带宽、sin/cos 取值、布局 | 应尽量融合 QKV 预处理与 KV 写入 |
| QKᵀ、softmax、PV | 从 prefix 与 action 状态聚合信息 | 矩阵乘、KV 读、softmax reduction | 必须保留非因果 / mask 语义 |
| KV cache update / gather | 保存、读取、转换 K/V 布局 | 带宽、页表、padding、拷贝 | R1 有边界 gather；π0.5 共享 paged pool |
| Dense MLP、SwiGLU / GeGLU | 扩展通道、非线性、压回 hidden | GEMM + 中间激活读写 | 通常是重要计算和权重流量来源 |
| MoE router / grouped GEMM / scatter | 稀疏激活部分专家 | 不均衡路由、小 GEMM、元数据与原子操作 | 属于其他主干或未来 MoE-VLA，不是两条 action head 的既有路径 |
| 时间 Fourier / sinusoidal embedding | 编码 flow 时间；R1 还编码当前动作值 | 小 MLP、逐元素算子、重复计算 | 仅时间相关部分有预计算机会 |
| AdaRMSNorm + gated residual | 按时间调制 scale / shift / gate | 条件投影权重读取、norm、逐元素操作 | π0.5 特有重点 |
| Action output projection | hidden → 连续动作速度 / 动作通道 | 小输出 GEMM、数值敏感 | 不是词表 softmax |
| Euler 更新 | 根据速度更新当前连续状态 | 极小带宽工作，launch 开销可能主导 | 应留在设备端 |
| LM head + sampling | hidden → 词表 logits → 下一个 token | 大词表投影、reduction、sampling | R1 语言阶段有；π0.5 action 没有 |
| GDN / Mamba SSD / causal convolution | 线性注意力或状态空间递推 | 状态更新、scan、chunk GEMM | 仓库支持的其他架构，不要套到当前 VLA 图上 |
| All-reduce / 多设备同步 | 张量并行结果合并 | 互联、同步、小消息延迟 | 只在实际多设备路径中相关，单 Thor 不自动受益 |

### 4.2 GEMM：最重要，但要先看 M

计算量近似：

```text
FLOPs = 2 × M × N × K
```

对一次完整权重读取，忽略激活、输出、scale 和缓存等开销：

```text
FP16 权重字节数 ≈ 2NK
理想算术强度 AI ≈ 2MNK / (2NK) = M FLOP/byte
```

所以三个场景不同：

| 场景 | 常见 M | 优先考虑 |
|---|---:|---|
| 文本单 token decode | `B`，常为 1 | GEMV、小 M kernel、权重压缩 |
| VLA action chunk | `B × H`，如 10、15、50、64 | 短 M GEMM、tile 利用率、融合、循环调度 |
| 视觉 / prefix prefill | 图像 patch 数或 `B × S` | Tensor Core 吞吐、TMA 流水线、attention 内存流量 |

**Action 是“短序列反复算”，不是“单 token 反复算”。**
不能拿 `M=1` 的最快 GEMV 直接推断 `M=50` action GEMM 的最优实现。
上面的 AI 是理想估算；实际有权重重复读、cache、量化解码和输出等成本。

对具有 gate/up/down 三个投影的 gated MLP，近似计算量为：

```text
FLOPs_MLP ≈ 6 × M × D × F
```

例如 π0.5 action 的 `D=1024, F=4096, H=10, B=1`：

```text
每层 MLP ≈ 0.252 GFLOP
18 层 × 10 次去噪 ≈ 45.3 GFLOP
```

这仅是 MLP，不含 QKV、attention、norm、输出和访存；也不含已经移出的 cond。
这个例子说明：短 chunk 不等于总工作量小，**层数 × 去噪步数**会放大成本。

### 4.3 Attention：不能只看 FlashAttention 这个名字

基本式：

```text
O = softmax(QKᵀ / sqrt(d) + mask) V
```

对一个 action 层，Q 长度为 `H`，K/V 有效长度为 `S+H`：

```text
QKᵀ + PV FLOPs ≈ 4 × B × n_q × H × (S+H) × d
```

优化方向：

1. 分块计算 QKᵀ 和 PV，使用 online softmax，避免完整 score/probability 矩阵落到全局内存。
2. 复用 GQA/MQA 的共享 K/V，避免为每个 query head 复制 KV。
3. 融合 QKV split、RoPE、cache write，减少中间张量。
4. 避免无效 padding 和过大的 cache capacity 增加读写或 workspace。
5. 选择适合 head dim 和 query 长度的 kernel，而不是只按“是否 Blackwell”分派。

当前路径必须分开理解：

| 场景 | Attention 语义 | 当前实现入口 |
|---|---|---|
| 普通 LLM prefill | 常为 causal / sliding window | AttentionPlugin → context FMHA 等 |
| 普通 LLM decode | 一个或少量 query，读取历史 KV | XQA |
| π0.5 vision / prefix | 全双向；prefix 已 compact | TensorRT 原生 Attention |
| π0.5 action | prefix 可见，action chunk 内全可见 | AttentionPlugin → tree-shaped XQA |
| Alpamayo action | 有效 prefix + action chunk 非因果，显式长度 mask | TensorRT 原生 Attention |

项目有 [Blackwell CuTe FMHA](../kernelSrcs/fmha_cutedsl_blackwell/README.md)，但**不能据此认定 π0.5 visual/prefix 或 R1 action 已经调用该实现**。
对于 TensorRT 原生 Attention，融合程度和最终 kernel 要查看构建结果与 profile。

### 4.4 Normalization、激活和残差：小算子的大累积

RMSNorm：

```text
rms = sqrt(mean(x²) + epsilon)
y = x / rms × weight
```

LayerNorm 还需要均值中心化；Gemma 的 norm 权重形式为 `1 + weight`。
这些算子主要是 reduction 与 elementwise，不应把 Tensor Core 峰值当作其加速上限。

π0.5 的 AdaRMS 还包括：

```text
(scale, shift, gate) = Dense(time_condition)
z = RMSNorm_without_weight(x) × (1 + scale) + shift
residual_out = residual_in + gate × sublayer(z)
```

优化顺序：

- 先消除重复条件投影，再考虑 norm / affine / residual 融合。
- 保留实现中有意使用的 FP32 reduction；它对小方差、时间调制和多步误差传播有意义。
- 不能把 π0.5 GeGLU 的 GELU 近似随意替换为 R1 的 SwiGLU / SiLU。
- TensorRT 原生图可能已经融合，不应仅凭 Python 节点个数判断 launch 数量。

## 5. VLA 专项：真正能减少重复工作的地方

### 5.1 Prefix KV 是一次请求内的“不变量”

一次观测中，prefix 对所有去噪步相同：

```text
视觉/文本 → 每层 prefix KV：只计算一次
x_t → 每层 action KV：每个 t 都重新计算
```

π0.5 的 K/V pool 为 `[2, num_pages, 128, n_kv, d]`，采用 identity page table。
在其 `n_kv=1` 合约下，连续页可以与 prefix 输出的连续序列布局对应，避免每步重排。
这依赖布局条件，不能在任意 GQA 配置上照搬。

估算一个 FP16 KV pool：

```text
KV bytes = L × 2 × B × capacity × n_kv × d × 2
```

π0.5 取 `L=18, B=1, capacity=1024, n_kv=1, d=256`，为 **18 MiB**。
这不包括 TensorRT workspace、临时激活、权重和其他组件的内存。

对 R1，循环前的 paged → head-major gather 是一个明确优化候选：

- 先量化 gather 占比；
- 再评估让 action 直接消费 paged KV 的可行性；
- 必须同时处理非因果 mask、位置编码、GQA、输入/输出别名和数值精度；
- 不能只删除 gather，因为当前 action 引擎的输入布局需要它。

**跨观测缓存不自动成立。** 新相机帧、state 或文本改变后，旧 prefix KV 可能失效。
“请求内复用 KV”与“跨控制周期复用视觉特征”应分开设计和验证。

### 5.2 π0.5 的 cond hoisting：比低精度更直接的优化

当前实现把 37 个 `[3072,1024]` AdaRMS Dense 和时间 embedding 移到 cond 引擎。
这些 Dense 的 FP16 权重约：

```text
37 × 3072 × 1024 × 2 = 232,783,872 bytes ≈ 232.8 MB
```

如果 schedule 有 10 步、batch 为 1，其预计算结果仅：

```text
10 × 37 × 3072 × 2 = 2,273,280 bytes ≈ 2.17 MiB
```

意义不只是算子融合，而是**将仅依赖时间的计算移出热点循环**。
不能简单声称“每个请求必然节省 R × 232.8 MB 的 DRAM 流量”，因为缓存行为与融合会改变实际流量；但从循环中移出这一组投影的事实成立。

源码：
[cond 与 action 导出定义](../tensorrt_edgellm/models/pi05/modeling_pi05_action.py)、
[modulation 缓存和循环](../experimental_models/pi05/cpp/action/pi05ActionRunner.cpp)。

边界：

- `cond` 的结果不依赖当前图像或噪声，才能跨请求复用。
- Action 本身的 attention/MLP 依赖 `x_t`，不能整体预计算。
- R1 输入投影同时处理动作 Fourier 特征和时间信息，不能不加分析地把整个输入 MLP 移出循环。

### 5.3 CUDA Graph：减少提交开销，不减少数学计算

若每步包含大量短 kernel，CPU 连续提交和 GPU launch 间隙可能明显：

```text
普通循环：host enqueue → 多个 GPU kernels → host enqueue → ...
Graph：   host graph launch → 重放预先捕获的多步 GPU 工作
```

π0.5 已提供整循环 capture，并在捕获前 warmup，捕获后恢复初始噪声和时间状态。
否则 warmup 会消费 `x₀`，使正式运行从错误状态开始。

应分别报告：

- 冷启动 / 首次 kernel module 加载；
- Graph warmup、capture、instantiate；
- 固定形状下的稳态 replay；
- batch 或步数改变后的重捕获。

CUDA Graph 并非 Blackwell 新发明，但与 Blackwell 快速 Tensor Core 配合后，短 kernel 的 host overhead 更容易成为瓶颈。

### 5.4 视觉 token：一次减少，多个阶段受益

π0.5 每个有效视图产生 256 tokens。
跳过一个**原本按 policy 就被 mask 掉**的视图，可以同时避免：

1. 一次 vision tower 计算；
2. prefix 中 256 个 token 的 MLP / attention 工作；
3. 后续每个 action 去噪步对这些 prefix KV 的访问。

当前 runtime 已只处理实际提供的有效视图，并 compact prefix。
不能为了速度省掉 required camera，或任意降低分辨率后仍声称模型语义不变。
π0.5 的相机顺序、归一化、horizon、state 编码都由 policy 合约决定。

## 6. Blackwell 到底提供了哪些加速能力

### 6.1 先区分硬件，不把 B200 的规格套到 Thor

| 代表平台 | 本仓库相关 SM 路径 | 应关注的差异 |
|---|---|---|
| B200 / GB200 | SM100 | 数据中心 Tensor Core / HBM，SM100 专用 `tcgen05` 路径 |
| Jetson / DRIVE Thor 相关构建 | CUDA 13 路径为 SM110；仓库还保留 SM101/CUDA 12 artifact 和部分兼容处理 | 边缘 SoC 的带宽、功耗、共享系统资源；不能按 B200 规格估算 |
| RTX 50 / RTX PRO Blackwell | SM120 | 独立 GeForce-family kernel 路径 |
| GB10 / DGX Spark | SM121 | 独立 SM121 artifact；不要默认加载任意 SM100 kernel |

这不是完整 SKU 支持表。实际编译目标必须结合 GPU、CUDA 版本和 artifact metadata。
例如 [CMake](../cpp/CMakeLists.txt) 中 SM110 复用部分 SM101 FMHA cubin 的处理，是**特定 kernel 的项目兼容规则**，不是所有 `101/110` 二进制都通用的保证。

### 6.2 第五代 Tensor Core 与窄精度

Blackwell 的重要能力之一是 FP4 / FP6 / FP8 等窄精度矩阵运算及 block scaling。
对 LLM/VLA 的作用主要在：

- QKV/O projection；
- dense MLP；
- 视觉 transformer 的线性层；
- 真正 MoE 模型的 expert GEMM；
- 条件投影等矩阵算子。

但 softmax、sin/cos、norm reduction、Euler、页表和 scatter 等工作不会仅因 FP4 Tensor Core 峰值提高而同比加速。
也不能将带结构化稀疏条件的宣传峰值套到普通 dense 权重。

### 6.3 `tcgen05.mma` 与 TMEM：面向特定 Blackwell 路径

在 SM100 类和本项目 Thor SM110 专用实现中，可以看到 `tcgen05.mma` 和 Tensor Memory（TMEM）相关流水线。

概念性分工：

```text
全局内存：权重、激活
    ↓ 异步搬运
共享内存：分块输入、流水线缓冲
    ↓ 必要的低比特解码 / operand 准备
Tensor Core / TMEM：矩阵计算及相关片段、累加状态
    ↓ epilogue
全局内存：输出
```

TMEM 是专用片上 tensor 存储资源，不是 KV cache，也不是“免费扩大显存”。
其布局、容量、同步协议会影响 tile 和并发；是否更快取决于实际 workload。

本仓库的 Thor W4A16 kernel 使用专门的搬运、解码和 MMA warp 角色：
[NVFP4-A16 GEMM 源码](../kernelSrcs/nvfp4_a16_blackwell_gemm/nvfp4_a16_blackwell_gemm.py)。

**SM120/SM121 不应套用该实现假设。**
例如项目中的 [GeForce FP16 GEMM](../kernelSrcs/gemm_cutedsl/gemm_blackwell_geforce.py)
实际使用 `MmaF16BF16Op(16,8,16)`、`LdMatrix` 和 TMA；不能因为 README 中出现 “WGMMA-style” 就认定最终指令是 Hopper WGMMA 或 `tcgen05`。

### 6.4 TMA + 多级流水线：让搬运与计算重叠

TMA（Tensor Memory Accelerator）用于按 tensor 描述符异步搬运数据。
它起源于 Hopper，Blackwell 继续利用，并不是 Blackwell 独有的新功能。

典型流水线：

```text
阶段 A：搬运 tile k+1
阶段 B：准备 / 解码 tile k
阶段 C：计算 tile k-1
阶段 D：写回更早 tile 的结果
```

若完全串行：

```text
T_tile ≈ T_load + T_decode + T_mma + T_store
```

充分流水化后的稳定阶段更接近最慢阶段的耗时，而不是所有阶段相加。
但启动/排空、barrier、依赖和资源竞争仍然存在。

VLA 的限制：`M=B×H` 往往较短，可能只有少量 tile，无法充分摊薄流水线启动成本。
多加 stage 会占用更多 shared memory，不保证提升 occupancy 或速度。

### 6.5 Warp specialization、persistent kernel、PDL

| 技术 | 加速机制 | 限制 |
|---|---|---|
| Warp specialization | 不同 warp 分别搬运、解码、MMA、epilogue | register 分配与 barrier 必须平衡 |
| Persistent kernel | 常驻 CTA 连续领取 tiles，降低调度成本、改善复用 | 小 workload 可能浪费资源；不能只追求常驻 |
| Epilogue fusion | GEMM 后直接做 scale、激活、残差或 scatter | 融合后的寄存器/SMEM 成本可能抵消收益 |
| Programmatic Dependent Launch（PDL） | 在显式依赖协议下让相邻 kernel 的无依赖准备阶段重叠 | 不是忽略真实数据依赖；也不是 Blackwell 独有 |
| Thread-block cluster | 部分算法可通过 cluster 协作与数据复用受益 | 不能假定所有 SKU、tile 和小 M 场景同样有效 |

[Thor W4A16 MoE 实现说明](../kernelSrcs/nvfp4_a16_blackwell_moe/README.md)
展示了 TMA、TMEM、warp specialization 和 PDL 的实际组合。
这说明仓库具备相关工程经验，**不表示 VLA 当前在运行这些 MoE kernel**。

## 7. NVFP4、W4A16：为什么省，省在哪里

### 7.1 NVFP4 不是“把 float16 强转成 4 bit”

可将 NVFP4 权重重建概括为：

```text
w ≈ q_E2M1 × s_block_E4M3 × s_global_FP32
```

- `q`：4-bit E2M1，表示范围和精度非常有限；
- 每 16 个值共享一个 FP8 E4M3 block scale；
- 另有 tensor 级 FP32 scale；MoE 实现可能按 expert 保存对应尺度。

忽略 global scale、padding 和布局对齐，每个元素的存储：

```text
4 + 8/16 = 4.5 bits
相对 FP16 的理想权重压缩比 = 16/4.5 ≈ 3.56 倍
```

因此不能把“FP4 有效载荷是 FP16 的四分之一”写成“整个模型显存或延迟必然改善 4 倍”。
KV、activation、workspace、scale、重排 padding 和未量化模块都仍有成本。

### 7.2 W4A16 与原生 FP4×FP4 是不同路线

| 路线 | 存储 / 计算思路 | 主要收益 | 主要代价 |
|---|---|---|---|
| FP16/BF16 dense | 权重和激活高精度 | 简单、数值基线明确 | 权重流量大 |
| NVFP4 W4A16 | 权重压缩；激活保持 16 bit；kernel 内重建权重后进行相应计算 | 降权重带宽，避免 activation FP4 误差 | 解码与 scale 处理成本 |
| NVFP4 W4A4 | 权重、激活均走 FP4 blockscaled 路径 | 进一步降低带宽并利用 FP4 MMA | activation 量化/重标定、额外数值风险 |

本仓库的 Thor W4A16 grouped GEMM 明确采用 `tcgen05 kind::f16` mixed-input 路线。
**存储是 NVFP4，不等于核心 MMA 一定使用 FP4×FP4。**

当前 Thor dense W4A16 分派也不是“一切小 M 都走 GEMV”：

- 默认 `M=1` 走 GEMV；
- 其余满足支持条件的形状默认走 `tcgen05`；
- 显式 GEMV backend 只支持若干 M：1、2、4、8、16；
- 形状受 `N % 128 == 0`、`K % 64 == 0`、TMA stride 等限制。

来源：
[dispatch policy](../cpp/plugins/nvfp4A16GemmPlugin/nvfp4A16BlackwellDispatchPolicy.h)、
[形状与目标 SM 约束](../cpp/kernels/nvfp4A16BlackwellSupport.h)。

对于 π0.5，`H=10/15/50` 以及 vision 的 `intermediate_size=4304` 提醒我们：
不能把所有 Linear 机械替换为某个对齐要求固定的低比特 Plugin。
必须逐层核对 N/K、padding、fusion、dtype 和 export ABI。

### 7.3 你正在看的 NVFP4 Marlin 测试与 VLA 的关系

[SM12x NVFP4 A16 Marlin MoE 测试](../unittests/cpp/kernels/moe/nvfp4A16MoeMarlinSm12xTests.cu)
面向 SM120/SM121 的 MoE W4A16 路径，并检查输出有限性、参考精度和 scale 转换等。

它适合研究：

- packed FP4 权重如何解码；
- E4M3 block scale 如何展开到 FP16/BF16；
- global scale 与局部 scale 的补偿；
- 小规模 routed GEMM 的精度和性能。

其中 [scale helper](../cpp/kernels/moe/moe_marlin/marlin/nvfp4_scale.cuh)
包含 `Times128` 形式的内部尺度变换，以及受 CUDA 版本和 SM110 条件保护的 BF16 scaled-conversion 指令分支。
这说明**同一个 helper 文件可以含有不同架构的条件路径**：运行 SM12x 测试不等于执行了 SM110 的指令分支。
也不能单独删掉 `×128` 而不检查整个 Marlin 路径的尺度补偿。

对 VLA 而言，这些是低比特 kernel 的参考积木，不是现成的 R1 / π0.5 加速证明。
真正接入仍需完成模型层替换、权重布局、ONNX/Plugin ABI、引擎构建和动作数值回归。

### 7.4 为什么 VLA 量化不能只测单层 cosine

Flow matching 是多步递推：

```text
x_(k+1) = x_k + Δt × vθ(x_k, t_k, condition)
```

若量化给速度预测引入误差，后续步骤会在已经偏移的 `x_k` 上继续计算。
局部 Lipschitz 假设下，可用以下上界理解风险：

```text
e_(k+1) ≤ (1 + |Δt|L_v) × e_k + |Δt| × ε_k
```

这不是对真实模型误差必然增长的断言，而是说明：单步误差会通过后续状态依赖传播。
需要检查不同时间步、不同动作维度、最终 chunk 和 robot-unit 输出，不能只验证一个矩阵乘。

## 8. 当前已经实现什么，哪些只是优化方向

| 能力 | Alpamayo R1 | π0.5 |
|---|---|---|
| 视觉 / 主干 / action 分组件 | 已有 | 已有；另含可选 cond |
| 请求内复用 prefix KV | 已有；action 前 gather | 已有；共享 paged pool |
| 非因果 action attention | 原生 TRT 路径 | AttentionPlugin/XQA 路径 |
| Action RoPE/QKV/cache-write Plugin 融合 | 不应按 π0.5 路径推断 | 已有 |
| 整个去噪循环显式 CUDA Graph | 当前 runner 未实现该路径 | 已有可选 `--cudaGraph` |
| 时间条件 hoisting | 未见 π0.5 式独立 cond 路径 | 默认导出，步数/batch 改变时重算 |
| Euler 在设备上执行 | 已包含在 action 图中 | 独立 CUDA kernel，FP32 状态更新 |
| FP8/NVFP4 action 量化 | 不属于当前文档支持流程 | 未接入；plain Linear，FP16 路径 |
| Action FP8 KV | 不应从主干能力推断 | 当前显式关闭 |
| 去掉所有 CPU 参与 | 否 | 否；仍有 policy、绑定、输入输出等 host 工作 |

π0.5 的“FP16 模型”也不是所有运算都 FP16：norm reduction、时间特征中的部分计算、噪声状态与 Euler 更新保留 FP32。
R1 虽有 FP32 的外部 trajectory buffer，导出模型内部会把输入、时间步转到 FP16 后计算 Euler，再转回 FP32；不能声称它与 π0.5 的 FP32 Euler 完全同构。

## 9. 面向 VLA 的优化顺序

以下是**建议的实验顺序**，不是未经测量的瓶颈排名。

### P0：先保证输入合约和基线正确

- 固定模型、policy、相机顺序、图像预处理、prefix、初始噪声、步数。
- 确认运行的确实是目标 SM 的 artifact / tactic。
- 单独记录 vision、prefix/VLM、KV 准备、action loop、后处理。
- 把首次初始化与稳态执行拆开。

### P1：先做不改变模型数学语义的优化

**π0.5：**

1. 确认 cond hoisting 已启用，比较固定配置的 warm run。
2. A/B 测试 `--cudaGraph`，确认没有 capture 失败并退回普通 enqueue。
3. 检查是否有多余相机/padding 被送入 prefix；只去掉 policy 本来就 mask 的输入。
4. 检查 prefix KV 是否按设计直接写入共享 pool。

**Alpamayo R1：**

1. 分开测 VLM decode 和 action loop，避免优化错阶段。
2. 量化循环前 KV gather 成本。
3. 评估整 action loop Graph capture，提前固定缓冲区地址和时间表。
4. 评估移除每步时间 H2D 提交、消除或减少小 D2D copy；必须尊重 TensorRT 的输入输出别名规则。

### P2：围绕真实形状优化 GEMM / attention

- 对 `M=10/15/50/64` 及实际 batch 收集真实 N/K，不能只测整齐的 `4096³`。
- 比较 TensorRT 当前 tactic 与候选 plugin，包含重排、padding、scale 和 epilogue 的总成本。
- 对 head dim 72、128、256 分别确认可用 attention 实现，不做跨模型泛化。
- 测量寄存器 spill、shared-memory 占用、tile 利用率和内存吞吐，不把 occupancy 越高等同于越快。
- 对非因果 attention 的改造，先验证可见性矩阵，再看速度。

### P3：逐模块探索低精度

这是待开发路线，不是当前 VLA 的 CLI 即开能力：

1. 建立每层/每模块误差敏感性分析；
2. 对满足 kernel 形状的 Linear 分别尝试 weight-only W4A16、FP8 或其他受支持方案；
3. 对 norm reduction、时间 embedding、AdaRMS gate、输出投影和 Euler 更新先保留基线精度，再做消融实验；
4. 校准覆盖真实相机、状态、任务以及多个去噪时间步，不能只用普通文本语料；
5. 同时测数值、端到端延迟和显存，拒绝“单 kernel 更快、完整 action 更慢”的方案。

W4A16 减少 activation 量化风险，但不保证一定比 W4A4 更快，也不保证所有模型更准；最终需实测。

### P4：改变算法预算的优化单独立项

包括减少去噪步数、减少输入帧、降低分辨率、缩短语言输出、减少 action horizon、跨观测复用特征、蒸馏等。

这些可能带来更大收益，但会改变模型输入或求解过程，不能当作纯 kernel 优化。
若要引入，需要重新进行动作质量与任务级评估。

## 10. 性能分析：测什么才有意义

### 10.1 端到端指标

| 指标 | 为什么需要 |
|---|---|
| 观测接收 → action chunk 完成的 wall-clock 延迟 | 包含 CPU 预处理、提交和输出，不只 GPU 时间 |
| P50 / P95 / P99 | 边缘设备重视抖动与截止时间 |
| 各组件 CUDA event 时间 | 定位 GPU 上真正耗时阶段 |
| Cold / warm / graph-recapture 延迟 | 首次运行与稳态完全不同 |
| 显存 / 系统内存峰值、workspace | 判断部署容量和量化收益 |
| 平均功耗、能量 / action chunk、温度和频率 | Thor 等边缘设备可能受功耗与散热约束 |
| 数值误差、任务成功率 | 防止以错误输出换取速度 |

`action horizon / latency` 可以描述模型产出速度，但不能直接等价为闭环控制频率。

### 10.2 工具分工

- **Nsight Systems**：看 host enqueue 间隙、stream 顺序、H2D/D2H/D2D、同步、Graph replay 和组件时间线。
- **Nsight Compute**：对选定热点 kernel 看 Tensor Core 利用率、DRAM/L2 流量、stall、寄存器与 shared-memory 限制。
- **TensorRT engine inspector / profiling**：确认原生层是否融合、精度/tactic 与实际调用的 Plugin。
- **项目已有 NVTX/metrics**：R1 runner 有 action 阶段标记；π0.5 action runner 有 CUDA event 计时。

CUDA event 不自动覆盖 host 预处理；Nsight Compute 可能通过 replay 干扰时间。
**不要把 profiler 下的整模型耗时当成无干扰的端到端 benchmark。**

### 10.3 建议的最小实验矩阵

| 维度 | 测试建议 |
|---|---|
| 平台 | Thor 与 SM120/SM121 分别测，记录 CUDA/TRT/artifact |
| Batch | 先 B=1，再测实际业务 batch，而非仅追求吞吐 |
| π0.5 | 分别使用 LIBERO / DROID / ALOHA 合约与对应观测 |
| R1 | 固定图像数量、有效 prefix 长度、语言生成预算 |
| Action loop | 固定噪声，比较 Graph on/off；π0.5 比较 hoist on/off |
| Cache | 固定输入实验之外，增加不同请求，避免只报告热 L2 最佳值 |
| 精度 | FP16 基线；每个低精度候选单独列完整回归结果 |
| 统计 | 声明 warmup 次数、样本数、分位数，尾延迟需要足够样本 |

Roofs 与收益上界：

```text
理想 kernel 时间下界 ≳ max(FLOPs / 有效算力, bytes / 有效带宽)
整体加速比 = 1 / ((1-p) + p/s)
```

若 action 占总延迟 40%，把 action 提速 2 倍，整体仅约 `1.25×`。
这是解释为何必须先按阶段 profile，而不是直接挑最复杂 kernel 改写。

## 11. 正确性验收：必须比“模型能跑”更严格

### 11.1 三层验证

1. **算子级**：真实 shapes、边界 padding、不同 batch、dtype、scale 极值、有限性和误差。
2. **模型级**：相同预处理输入、相同初始噪声、相同步数，比较中间速度、最终 action chunk。
3. **任务级**：在目标仿真或受控评估中检验任务成功、动作平滑性与约束；数值回归不替代此层。

任何模型能力声明都应经过：

```text
export → TensorRT engine build → inference → 数值 / 任务验证
```

只通过 ONNX export，甚至只通过某个 CUDA 单测，都不能证明 NVFP4 VLA 已可用。

### 11.2 π0.5 已有的比较工具

[compare_pi05_actions.py](../experimental_models/pi05/scripts/compare_pi05_actions.py)
的默认门槛：

```text
cosine >= 0.99999
max_abs <= 5e-3
```

这些是项目当前 FP16 数值回归门槛，不是本文的新实测结果，也不是机器人安全判据。
不能为了使一个新量化方案通过而无说明地放宽门槛。

比较时要注意：

- 必须共享实际 `x₀`；不同 RNG 实现使用相同 seed，不保证产生相同噪声。
- 使用相同 pixels、token IDs、相机顺序、policy 和去噪步数。
- `actions` 是归一化 padded chunk，`robot_actions` 是机器人单位输出；二者不能混着比较。
- 使用对应 embodiment 的观测；跨 embodiment 数据不能替代分布内准确性验证。
- Batch 应逐项检查，不能让一个好的样本掩盖另一个坏样本。

### 11.3 Attention 改造的必查项

- Prefix 是否全双向；action token 是否彼此可见；
- 无效 KV 是否被屏蔽；不能把整块 capacity 当成有效长度；
- RoPE/MRoPE 的位置偏移是否正确；
- QK scale 是乘一次还是两次，乘法放置改变是否引入额外 rounding；
- KV head 数和布局是否符合共享 pool 的假设；
- 输入/输出别名是否与 TensorRT 调度一致；
- 去噪后 suffix KV 是否覆盖而非持续 append；
- Graph 捕获后更换地址/步数/batch，是否正确失效并重建。

## 12. 推荐源码阅读路线

| 主题 | 首先读哪里 |
|---|---|
| Alpamayo 完整使用流程 | [Alpamayo VLA 示例](source/user_guide/examples/vla/alpamayo.md) |
| R1 单步 action 图 | [modeling_alpamayo_action.py](../tensorrt_edgellm/models/alpamayo/modeling_alpamayo_action.py) |
| R1 KV gather 与 10 步循环 | [alpamayo1ActionRunner.cpp](../cpp/action/alpamayo1ActionRunner.cpp) |
| π0.5 组件与 policy 合约 | [π0.5 设计](source/developer_guide/models/pi05.md)、[使用示例](source/user_guide/examples/vla/pi05.md) |
| π0.5 模型结构 | [visual](../tensorrt_edgellm/models/pi05/modeling_pi05_visual.py)、[prefix](../tensorrt_edgellm/models/pi05/modeling_pi05_prefix.py)、[action/cond](../tensorrt_edgellm/models/pi05/modeling_pi05_action.py) |
| π0.5 Graph 与缓存 | [pi05ActionRunner.cpp](../experimental_models/pi05/cpp/action/pi05ActionRunner.cpp)、[pi05Runtime.cpp](../experimental_models/pi05/cpp/runtime/pi05Runtime.cpp) |
| π0.5 policy 前后处理 | [pi05Policy.cpp](../experimental_models/pi05/cpp/runtime/pi05Policy.cpp) |
| Euler kernel | [pi05Kernels.cu](../experimental_models/pi05/cpp/kernels/pi05Kernels.cu) |
| Attention phase / kernel 分派 | [attentionPlugin.cpp](../cpp/plugins/attentionPlugin/attentionPlugin.cpp) |
| XQA 与 JIT 生命周期 | [XQA README](../kernelSrcs/xqa/README.md) |
| Blackwell context / vision FMHA | [FMHA README](../kernelSrcs/fmha_cutedsl_blackwell/README.md) |
| RoPE 与 KV 写入融合 | [applyRopeWriteKV.cu](../cpp/kernels/posEncoding/applyRopeWriteKV.cu) |
| KV layout 工具 | [kvCacheUtilsKernels.cu](../cpp/kernels/kvCacheUtilKernels/kvCacheUtilsKernels.cu) |
| Thor dense W4A16 | [GEMM runner](../cpp/kernels/nvfp4A16BlackwellGemm/nvfp4A16BlackwellGemmRunner.cpp)、[CuTe kernel](../kernelSrcs/nvfp4_a16_blackwell_gemm/nvfp4_a16_blackwell_gemm.py) |
| Thor W4A16 MoE | [kernel 说明](../kernelSrcs/nvfp4_a16_blackwell_moe/README.md) |
| 原生 NVFP4 MoE 路径 | [Thor split FC1/FC2](../kernelSrcs/nvfp4_moe_cutedsl/README.md)、[SM12x fused MoE](../kernelSrcs/nvfp4_fused_moe_cutedsl/README.md) |
| SM12x Marlin 精度测试 | [nvfp4A16MoeMarlinSm12xTests.cu](../unittests/cpp/kernels/moe/nvfp4A16MoeMarlinSm12xTests.cu) |
| 其他架构的 GDN / Mamba SSD | [GDN README](../kernelSrcs/gdn_cutedsl/README.md)、[SSD README](../kernelSrcs/ssd_cutedsl/README.md) |
| LLM sampling | [sampling.cu](../cpp/sampler/sampling.cu) |
| 构建与 SM artifact | [CuTe DSL 总览](../kernelSrcs/README.md)、[CuteDsl.cmake](../cmake/CuteDsl.cmake) |

阅读原则：先顺着模型实际调用路径走，再下钻 kernel；不要仅凭目录名含有 `Blackwell`、`FP4` 或 `expert` 就认定它处于 VLA 热路径。

## 13. 官方资料与最终建议

硬件机制参考以下官方资料；具体项目接入以本文链接的源码为准：

1. [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)：确认 GPU 对应的 compute capability。
2. [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)：SM、shared memory、cluster、memory system 的调优背景；注意其不同章节适用的 SKU。
3. [CUTLASS Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)：`tcgen05.mma`、窄精度、blockscaled GEMM 与布局约束。
4. [Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)：E2M1、16 元素 E4M3 block scale 与二级缩放。

**对本工程 VLA 最实用的路线是：**

```text
保证 policy 与 FP16 基线正确
    → 分阶段 profile
    → 消除重复计算、无效 token 和循环提交开销
    → 按真实短 M 形状调 GEMM / attention
    → 分模块接入 Blackwell 低精度能力
    → export / build / inference / 数值 / 任务级验收
```

Blackwell 的价值不只是“矩阵乘更快”，而是更强的矩阵单元、低比特存储与计算、异步搬运和片上流水线能协同工作。
对 VLA，只有把这些能力放进**正确的非因果 attention、KV 生命周期和多步 action 循环**，才会转化为真实、可验证的动作生成延迟收益。
