---
title: AI Infrastructure 学习路线图
date: 2026-01-01 23:15
modified: 2026-01-01 23:15
tags:
  - roadmap
  - learning-path
  - ai-infrastructure
categories:
  - 学习笔记
excerpt: 从 GPU 架构到分布式训练框架的完整学习路线，涵盖 CUDA 编程、并行策略、推理框架优化及开源项目实战。
mathjax: false
comment: true
---

一、 GPU & 算子层 (底层进阶)
目标：跨越 Infra 工程师与普通算法工程师的第一道门槛，深入理解硬件架构与高性能计算。

[ ] 掌握 GPU 核心架构

[ ] 计算架构：深入理解 SM (Streaming Multiprocessor)、Warp 调度机制以及 Tensor Core 的加速原理。

[ ] 内存架构：理清 Register、Shared Memory、L2 Cache 及 HBM 的层级关系与访问延迟。

[ ] 算子性能瓶颈分析

[ ] 理论模型：熟练运用 Roofline Model，能准确判断算子是 compute-bound（计算密集型）还是 memory-bound（访存密集型）。

[ ] 核心算子深入：研究 LLM 核心算子（Attention、GEMM、Softmax）的实现细节与优化空间。

[ ] 实践项目

[ ] 手动实现 FlashAttention：尝试手搓一个简易版，通过实际编码发现并补齐底层知识短板。

二、 框架层 (系统集成)
目标：理解从基础库（Torch）到大规模训练/推理框架的全栈内容。

[ ] 核心关注点：通用能力

[ ] 并行策略：深入研究并掌握主流并行方案：

DP / ZeRO (数据并行/零冗余优化)

TP / PP (张量并行/流水线并行)

CP / EP (上下文并行/专家并行)

[ ] 显存管理：学习如何在有限资源下最大化显存利用率。

[ ] 核心关注点：推理框架特有

[ ] KV Cache 管理：理解 PagedAttention 等机制对长文本推理的重要性。

[ ] Batch 调度：学习 Continuous Batching 等动态调度策略。

[ ] PD 分离：研究推理上的 Prefill (预填充) 与 Decoding (解码) 阶段分离技术。

三、 开源贡献与实战建议
目标：通过参与顶级开源项目，将理论转化为实际工程能力。

[ ] 进阶路径：参与主流项目贡献

[ ] 尝试为 vLLM 贡献代码。

[ ] 尝试为 SGLang 贡献代码。

[ ] 入门路径：从简化版框架开始 (难度适中)

[ ] 研读并理解 nano-vLLM 源码。

[ ] 研读并理解 mini-sglang 源码。

[ ] Action：通过看懂这两个简化框架，掌握推理框架的大部分核心功能。