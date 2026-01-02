# AI Infrastructure 学习笔记

AI 基础设施（AI Infrastructure）学习知识库，从 GPU 架构到分布式训练框架的系统性学习记录。

## 学习路线

**Tier 1: GPU & 算子层**
- GPU 架构：SM、Warp、Tensor Core
- 内存层次与性能优化
- 实践：手写 FlashAttention

**Tier 2: 框架层**
- 并行策略：DP/ZeRO、TP/PP、CP/EP
- 推理框架：vLLM、SGLang 核心机制
- 实践：研读 nano-vLLM、mini-sglang

**Tier 3: 开源贡献**
- 为 vLLM 或 SGLang 贡献代码

详细进度见 [road-map.md](road-map.md)

## 目录结构

```
├── cuda/                                    # CUDA 编程与 GPU 架构
│   ├── nvidia_gpu_basic_structures.md      # GPU 基础数据结构
│   └── cuda_programming_basics_summary.md  # CUDA 编程基础
├── road-map.md                              # 学习路线图
├── CLAUDE.md                                # 项目说明与 Git 规范
└── README.md                                # 本文件
```

## 快速开始

1. **GPU 基础**：`cuda/nvidia_gpu_basic_structures.md`
2. **CUDA 编程**：`cuda/cuda_programming_basics_summary.md`
3. **制定计划**：参考 `road-map.md`

## 技术栈

- CUDA 编程与 GPU 优化
- PyTorch 扩展开发
- 分布式训练（NCCL）
- LLM 推理框架（vLLM、SGLang）

## 推荐资源

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)


---

MIT License
