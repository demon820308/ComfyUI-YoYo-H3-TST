# ComfyUI-YoYo-H3-TST

Training-free temporal attention homeostasis correction (**YoYo H3 Temporal State Transport (TST)**) for **MiniMax H3** video generation in ComfyUI.

Node type: `YoYo_H3TemporalStateTransport` · Menu category: `YoYo/MiniMax H3`

## Overview

Based on the paper **[Temporal State Transport in Video Generation: Diagnosing and Correcting Spectral Imbalance](https://arxiv.org/abs/2609.08505)** (Luyao Tang et al.) and its official reference repository [lytang63/temporal-state-transport](https://github.com/lytang63/temporal-state-transport) (MIT License), this plugin provides an independent ComfyUI implementation adapted for MiniMax H3's packed attention layout:

- **Per-head Spectral Tension diagnosis**: Pools the video segment's post-RoPE `q`/`k` per frame to build the frame-level transport operator $A$, and computes Spectral Tension $T = H_{\text{row}} - H_{\text{vN}}$ (row entropy minus von Neumann entropy of the normalized Gram matrix).
  - $T > 0$: over-mixed transport (physically implausible motion / structural collapse)
  - $T < 0$: fragmented transport (flickering details / identity or outfit drift)
- **Homeostatic query temperature**: Scales only the video-row queries by $\gamma = \exp(T \cdot \tau_{\text{eff}})$ to sharpen over-mixed heads and soften fragmented ones, while leaving balanced heads nearly untouched.
- **Cosine schedules**: $\tau_{\text{eff}}$ is scheduled with cosine layer and step weights (stronger in deeper layers and earlier denoising steps).
- **Non-intrusive integration**: Chains cleanly with any existing `optimized_attention_override` (such as Sol-Attn) rather than overwriting it, and leaves text, keyframe/reference, and audio segments untouched.

## Installation

### Via Comfy Registry / ComfyUI-Manager

```bash
comfy node install comfyui-yoyo-h3-tst
```

Or search for **ComfyUI YoYo H3 TST** in ComfyUI-Manager.

### Manual

Place the `ComfyUI-YoYo-H3-TST` folder into `ComfyUI/custom_nodes/` and restart ComfyUI.

---

## 中文说明

一个节点：**YoYo H3 Temporal State Transport (TST)**  
（节点类型 `YoYo_H3TemporalStateTransport`，菜单位置：`YoYo/MiniMax H3`）

### 原理与功能

基于论文 **[Temporal State Transport in Video Generation: Diagnosing and Correcting Spectral Imbalance](https://arxiv.org/abs/2609.08505)** 及其官方开源仓库 [lytang63/temporal-state-transport](https://github.com/lytang63/temporal-state-transport)（MIT License），按公开数学公式针对 **MiniMax H3** 打包注意力结构独立实现的时间注意力稳态校正节点，专治 H3 视频中的细节闪烁/变形、人物与服装身份漂移、运动不合理等问题：

- 在每个注意力头上，将视频段的 `q`/`k` 按帧池化，构造帧级传输算子 $A$
- 通过谱张力 $T = H_{\text{row}} - H_{\text{vN}}$（行熵 − Gram 矩阵特征值的冯诺依曼熵）逐头诊断：
  - $T > 0$：过混合（物理崩坏方向）
  - $T < 0$：过碎片（细节/身份漂移方向）
- 以逐头 query 温度 $\gamma = \exp(T \cdot \tau_{\text{eff}})$ 仅缩放视频行的 `q`：过混合的头锐化、过碎片的头柔化，已平衡的头几乎不动
- $\tau_{\text{eff}}$ 采用两条余弦调度：深层网络更强、早期采样步更强

### 安装与用法

1. 通过 ComfyUI-Manager 安装，或把 `ComfyUI-YoYo-H3-TST` 文件夹放进 `ComfyUI/custom_nodes/` 并重启 ComfyUI
2. 将节点串进模型链任意位置（`MODEL` 进 → `MODEL` 出），无需更换采样器

参数与特性：
- `tau`：校正强度，默认 `0.2`（论文设定，范围 `0.0`–`1.0`）；`0.5` 通常偏强；`tau = 0` 关闭校正但保留诊断日志
- `log_diagnostics`：是否在控制台打印逐次前向的 `[YoYo TST]` 诊断信息
- 与已有的 `optimized_attention_override`（如 Sol-Attn）**链式组合**而非覆盖：本节点先做 `q` 缩放，再交给先前的 override
- 仅处理打包序列末尾的视频行；文本、关键帧/参考、音频行原样通过
- 非 MiniMax H3 模型会记录警告并原样返回；若序列布局不匹配则静默跳过，不会中断采样

### 验收方法

开启 `log_diagnostics` 后运行一次生成，控制台每次模型 forward 会输出一行：

```text
[YoYo TST] frames=37 step=0/5 heads corrected=47 mean|T|=... mean|gamma-1|=...
```

- 看不到这行日志通常表示未生效（先检查节点是否接在实际使用的模型链上）
- 固定同 seed、同提示词，对比 `tau = 0` 与 `tau = 0.2` 的闪烁与身份漂移改善效果
- 建议先在全尺寸 H3 工作流上验证，再考虑 Flash 等变体流程（分解注意力布局尚未验证）

## References / 参考文献

- **Paper**: [Temporal State Transport in Video Generation: Diagnosing and Correcting Spectral Imbalance](https://arxiv.org/abs/2609.08505)
- **Official Implementation**: [lytang63/temporal-state-transport](https://github.com/lytang63/temporal-state-transport) (MIT License)

```bibtex
@article{tang2026temporal,
  title={Temporal State Transport in Video Generation: Diagnosing and Correcting Spectral Imbalance},
  author={Tang, Luyao and Luo, Bingjun and Yi, Dong and Guo, Jialin and Xi, Haoning and Chen, Cheng and Yu, Yizhou and Chen, Chaoqi},
  journal={arXiv:2609.08505},
  year={2026}
}
```
