---
title: AI Infrastructure 学习路线图
date: 2026-01-01 23:15
modified: 2026-01-02 23:43
tags:
  - roadmap
  - learning-path
  - ai-infrastructure
categories:
  - 学习笔记
excerpt: 从 GPU 架构到分布式训练框架的完整学习路线，涵盖 CUDA 编程、并行策略、推理框架优化、云原生技术、PyTorch、分布式训练框架及开源项目实战。
mathjax: false
comment: true
---

## 一、GPU & 算子层 (底层进阶)
**目标**：深入理解硬件架构与高性能计算。

[ ] 掌握 GPU 核心架构

[ ] 计算架构：深入理解 SM (Streaming Multiprocessor)、Warp 调度机制以及 Tensor Core 的加速原理。

[ ] 内存架构：理清 Register、Shared Memory、L2 Cache 及 HBM 的层级关系与访问延迟。

[ ] 算子性能瓶颈分析

[ ] 理论模型：熟练运用 Roofline Model，能准确判断算子是 compute-bound（计算密集型）还是 memory-bound（访存密集型）。

[ ] 核心算子深入：研究 LLM 核心算子（Attention、GEMM、Softmax）的实现细节与优化空间。

[ ] 实践项目

[ ] 手动实现 FlashAttention：尝试手搓一个简易版，通过实际编码发现并补齐底层知识短板。

---

## 二、深度学习框架层 (PyTorch)
**目标**：深入理解主流深度学习框架的内部实现，掌握从 Python API 到底层 CUDA 算子的完整调用链。

[ ] PyTorch 核心架构

[ ] Tensor 系统：理解 Tensor 的存储机制、视图（view）机制以及自动微分（Autograd）的实现原理。

[ ] CUDA Extension：学习如何为 PyTorch 编写自定义 CUDA 算子，理解 `torch.utils.cpp_extension` 的使用。

[ ] 算子融合：研究 JIT 编译器（`torch.jit`）如何进行算子融合以优化性能。

[ ] 数据加载与预处理

[ ] DataLoader 实现：理解多进程数据加载、内存映射文件等高性能数据加载技术。

[ ] 分布式数据采样：研究 `DistributedSampler` 如何在分布式训练中保证数据不重复。

[ ] 实践项目

[ ] 为 PyTorch 贡献一个自定义 CUDA 算子。

[ ] 阅读并理解 `torch.nn.functional` 中核心算子的实现（如 `linear`, `conv2d`, `softmax`）。

---

## 三、云原生与基础设施层
**目标**：掌握现代 AI 应用的部署、编排和资源管理技术，理解容器化与虚拟化如何支撑大规模 AI 系统。

[ ] 容器化技术 (Docker)

[ ] 镜像优化：学习如何构建最小化的 AI 训练/推理镜像（多阶段构建、层缓存优化）。

[ ] GPU 容器化：理解 NVIDIA Container Toolkit 如何让容器访问 GPU，以及相关的性能考虑。

[ ] 容器编排：研究 Docker Compose 用于多容器 AI 应用的本地开发环境搭建。

[ ] 容器编排与管理 (Kubernetes)

[ ] 核心概念：掌握 Pod、Service、Deployment、DaemonSet 等 K8s 核心资源。

[ ] GPU 调度：理解 Kubernetes 如何通过 Device Plugins 和 Scheduler 管理 GPU 资源。

[ ] 自定义资源（CRD）：学习如何定义 AI 任务特有的资源类型（如 `ModelJob`, `TrainingJob`）。

[ ] Operator 开发：研究 K8s Operator 模式，学习如何用 Kubebuilder 或 Kudo 构建 AI 工作流控制器。

[ ] 虚拟化技术

[ ] GPU 虚拟化：理解 NVIDIA vGPU、MIG (Multi-Instance GPU) 等技术在多租户场景下的应用。

[ ] 轻量级虚拟化：研究 gVisor、Kata Containers 等安全容器的原理与性能权衡。

[ ] 实践项目

[ ] 搭建一个 K8s 集群，并部署分布式训练任务（使用 Kubeflow Training Operator）。

[ ] 编写一个自定义 K8s Operator 来管理模型推理服务的扩缩容。

---

## 四、分布式训练框架层
**目标**：深入理解主流分布式训练框架的设计理念与实现细节，掌握大规模模型训练的核心技术。

[ ] 并行策略深度解析

[ ] 数据并行 (Data Parallel / ZeRO)

[ ] 基础 DP：理解 PyTorch DDP 的实现原理（AllReduce、梯度同步、通信重叠）。

[ ] ZeRO 优化：研究 DeepSpeed ZeRO 如何通过分片优化器状态、梯度和参数来打破显存墙。

[ ] 梯度压缩：学习梯度压缩、通信与计算重叠等优化技术。

[ ] 张量并行 (Tensor Parallel)

[ ] 模型分片：理解如何将单个矩阵乘法操作切分到多个 GPU 上执行（如 `ColumnParallel` 和 `RowParallel`）。

[ ] All-Reduce 优化：研究 Ring AllReduce、Tree AllReduce 等通信模式在 TP 中的应用。

[ ] 流水线并行 (Pipeline Parallel)

[ ] Micro-batching：理解 GPipe、PipeDream 如何将 mini-batch 进一步切分为 micro-batches。

[ ] Bubble 填充：学习如何通过调度策略减少流水线空闲时间。

[ ] 上下文并行 (Context Parallel)

[ ] 长序列切分：研究 Ring Attention 等技术如何将超长序列分布到多个 GPU。

[ ] 专家并行 (Expert Parallel)

[ ] MoE 路由：理解 Mixture-of-Experts 模型中如何动态路由 token 到不同专家。

[ ] 负载均衡：学习如何解决 MoE 训练中的专家负载不均衡问题。

[ ] 主流框架对比

[ ] Megatron-LM

[ ] 架构设计：研究 Megatron-LM 的张量并行与流水线并行实现。

[ ] 混合精度：理解 FP16/BF16 训练及其在 Megatron 中的应用。

[ ] DeepSpeed

[ ] ZeRO 技术：深入理解 ZeRO-1/2/3 的分片策略与实现。

[ ] 显存优化：研究 DeepSpeed 的显存优化技术（梯度检查点、CPU Offload）。

[ ] FSDP (Fully Sharded Data Parallel)

[ ] PyTorch 原生：理解 FSDP 作为 PyTorch 原生分布式训练框架的优势。

[ ] 分片策略：研究 FSDP 的 FULL_SHARD、SHARD_GRAD_OP、NO_SHARD 等不同模式。

[ ] Ray

[ ] 分布式计算：理解 Ray 的 Actor 模型与分布式对象系统。

[ ] RLlib / Ray Train：学习 Ray 在强化学习和大规模数据训练中的应用。

[ ] 实践项目

[ ] 使用 Megatron-LM 或 DeepSpeed 训练一个中等规模模型（如 1B 参数）。

[ ] 对比不同并行策略（DP vs TP vs PP）在同一任务下的性能表现。

---

## 五、推理框架层
**目标**：深入理解 LLM 推理框架的核心技术，掌握高吞吐、低延迟推理系统设计的精髓。

[ ] 核心关注点：推理框架特有

[ ] KV Cache 管理

[ ] PagedAttention：理解 vLLM 如何借鉴操作系统虚拟内存思想来管理 KV Cache。

[ ] Cache 共享：研究多轮对话、跨请求 Cache 共享等高级优化技术。

[ ] Batch 调度

[ ] Continuous Batching：理解 vLLM 的迭代级调度（Iterative-level Scheduling）如何提升吞吐。

[ ] 动态 Batch：学习如何处理变长输入、动态加入/离开请求。

[ ] PD 分离

[ ] Prefill 与 Decoding：研究两个阶段的计算模式差异及优化策略。

[ ] 架构设计：理解分离式架构（如分开的 Prefill Worker 和 Decode Worker）的优势。

[ ] 量化与压缩

[ ] 量化技术：学习 Post-Training Quantization (PTQ)、Quantization-Aware Training (QAT)。

[ ] 量化格式：理解 INT8/INT4/FP8 等不同量化格式在推理中的性能影响。

[ ] Speculative Decoding

[ ] 投机采样：研究如何用小模型辅助大模型加速生成。

[ ] 验证机制：理解如何并行验证多个候选 token。

[ ] 实践项目

[ ] 深入研读 vLLM 源码，理解其调度器与内存管理器实现。

[ ] 深入研读 SGLang 源码，对比其与 vLLM 的设计差异。

---

## 六、大语言模型应用层
**目标**：理解大语言模型的核心技术，从架构设计到训练策略，建立完整的模型认知体系。

[ ] LLM 核心技术

[ ] Transformer 架构

[ ] Attention 机制：深入理解 Self-Attention、Cross-Attention、Multi-Head Attention 的实现。

[ ] 位置编码：研究 Sinusoidal、RoPE (Rotary Position Embedding)、ALiBi 等位置编码方案。

[ ] Normalization：理解 LayerNorm、RMSNorm 在 LLM 中的作用。

[ ] 模型架构演进

[ ] Encoder-Decoder：理解 T5 等架构的设计理念。

[ ] Decoder-only：研究 GPT、LLaMA 等仅解码器架构的优势。

[ ] 混合专家（MoE）：理解 Mixtral 等稀疏激活模型的实现。

[ ] 训练策略

[ ] 预训练：理解大规模语料预处理、数据配比、课程学习等。

[ ] 指令微调（SFT）：研究如何构建指令数据集、不同微调策略的效果。

[ ] 对齐技术：学习 RLHF (Reinforcement Learning from Human Feedback)、DPO (Direct Preference Optimization) 等对齐方法。

[ ] 训练技巧

[ ] 混合精度训练：理解 FP16/BF16 训练及其数值稳定性问题。

[ ] 梯度检查点：研究如何通过重计算减少显存占用。

[ ] Flash Attention：理解 IO 感知的精确 Attention 算法。

[ ] 实践项目

[ ] 从零实现一个简化版 Transformer（参考 nanoGPT）。

[ ] 使用 LLaMA、Qwen 等开源模型进行微调实验。

[ ] 复现一篇 LLM 训练相关的论文（如 FlashAttention、RoPE）。

---

## 七、工程工具语言层 (Golang)
**目标**：掌握 Golang 在 AI 基础设施中的应用场景，理解何时选择 Go 而非 Python。

[ ] Golang 核心特性

[ ] 并发模型：深入理解 Goroutine、Channel、Select 等并发原语。

[ ] 性能优势：理解 Go 在网络服务、微服务中的性能优势。

[ ] 部署简便：学习 Go 的静态编译、交叉编译特性。

[ ] AI 基础设施中的 Go 应用

[ ] 推理服务网关：研究如何用 Go 构建高性能推理网关（负载均衡、请求路由）。

[ ] 模型服务平台：理解 BentoML、KServe 等平台中 Go 组件的作用。

[ ] 基础设施工具：学习 etcd、Prometheus、containerd 等 Go 编写的基础设施组件。

[ ] Go 与 Python 互操作

[ ] gRPC/REST：理解如何用 Go 服务包装 Python 模型。

[ ] 共享内存：研究 Go 与 Python 进程间的高效数据交换。

[ ] 实践项目

[ ] 用 Go 编写一个简单的推理服务网关。

[ ] 研究一个主流 AI 基础设施项目的 Go 代码（如 etcd、Prometheus）。

---

## 八、开源贡献与实战建议
**目标**：通过参与顶级开源项目，将理论转化为实际工程能力。

[ ] 进阶路径：参与主流项目贡献

[ ] 尝试为 vLLM 贡献代码。

[ ] 尝试为 SGLang 贡献代码。

[ ] 入门路径：从简化版框架开始 (难度适中)

[ ] 研读并理解 nano-vLLM 源码。

[ ] 研读并理解 mini-sglang 源码。

[ ] Action：通过看懂这两个简化框架，掌握推理框架的大部分核心功能。