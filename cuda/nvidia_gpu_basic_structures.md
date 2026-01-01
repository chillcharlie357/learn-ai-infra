---
title: NVIDIA GPU 基础数据结构总结
date: 2026-01-01 23:15
modified: 2026-01-01 23:15
tags:
  - cuda
  - gpu-architecture
  - nvidia
  - simt
  - memory-hierarchy
categories:
  - 技术分享
excerpt: 详细介绍 NVIDIA GPU 的基础数据结构，包括执行层次结构（Thread、Warp、Thread Block、Grid）和内存层次结构（Registers、Shared Memory、Global Memory），以及 CUDA 编程模型和性能优化要点。
mathjax: true
comment: true
---

# NVIDIA GPU 基础数据结构总结

> 基于：[GPU Architecture and Programming — An Introduction](https://github.com/ForceInjection/AI-fundermentals/blob/main/gpu_programming/gpu_programming_introduction.md)

## 概述

NVIDIA GPU 以分层方式执行代码，采用 **SIMT（单指令多线程）** 架构。现代 GPU 可以在一个周期内跨数千个数据点运行一条指令，实现高效的并行计算。

---

## 一、执行层次结构

GPU 的执行结构分为四个层级，从下到上依次为：

### 1. Thread（线程）- 最基本的执行单元

**特点：**
- 使用私有寄存器对单个标量值进行操作
- 是 GPU 计算的最小单位
- 每个线程独立执行指令

**局限性：**
- 单个线程效率不高
- 无法独立处理大规模并行任务

### 2. Warp（线程束）- 32个线程的同步执行组

**定义：**
- 由 **32 个线程** 组成的小组
- 以 SIMT 方式执行，即 **单指令多线程**
- Warp 中的所有线程执行相同的指令，但处理不同的数据

**工作原理：**
- 每个 warp 跨 32 个数据点执行单个指令
- 例如在矩阵乘法中，一个 warp 可以处理矩阵的行和列，执行乘法和累加操作

**代码中的表示：**
- CUDA 提供 `threadIdx` 编译器内置变量来访问线程 ID

### 3. Thread Block（线程块）- 多个 Warp 的组合

**用途：**
- 当操作超过 32 个线程的 warp 大小时，GPU 使用线程块来管理更大的维度
- 使用 **tiling（平铺）** 技术将输入分成适合 warp 大小的块

**特点：**
- 将多个 warp 分组在一起
- 允许 warp 之间 **共享内存** 和 **同步执行**
- 用于累积部分结果

**代码中的表示：**
- `blockIdx`：块 ID
- `blockDim`：块中的线程总数

### 4. Grid（网格）- 多个线程块的集合

**用途：**
- 当数据集超过单个线程块的处理能力时，使用多个线程块的网格
- 共享全局内存

**特点：**
- 使 GPU 能够通过将工作负载分布到多个线程块来处理大型数据集
- 所有 GPU 程序（称为 **kernels**）都在此网格结构中执行

**启动参数：**
```cpp
kernel<<<grid_size, block_size>>>()
// grid_size: 线程块的数量
// block_size: 每个块的线程数
```

---

## 二、内存层次结构

GPU 的内存层次结构按照访问速度和容量分为三个层级：

### 1. Registers（寄存器）

**特点：**
- **私有内存**，每个线程独有
- **超低延迟**
- **容量小**：几 KB

**用途：**
- 存储线程的局部变量
- 最快的内存访问速度

### 2. Shared Memory（共享内存）

**特点：**
- 线程块内的多个 warp **共享**
- **中等容量**：默认 **48 KB** per block（可配置 16-96 KB）
- **低延迟**

**配置选项：**
- **配置1**：48 KB shared memory / 16 KB L1 cache（默认）
- **配置2**：16 KB shared memory / 48 KB L1 cache
- **架构差异**：
  - Kepler/Maxwell/Pascal：最大 48-96 KB
  - Volta/Turing：最大 64-96 KB
  - Ampere (A100)：最大 164 KB per SM
  - Hopper (H100)：最大 228 KB per SM

**用途：**
- Warp 之间共享状态
- 数据交换和同步
- 减少 global memory 的访问延迟

**优化技术：**
- **Tiling（平铺）**：使用共享内存来缓存频繁使用的数据
- 显著提高数据局部性
- NVIDIA 建议每个 block 使用不超过 32 KB 以保证跨架构兼容性

### 3. Global Memory（全局内存）

**特点：**
- **全设备可访问**
- **大容量**：
  - 消费级 GPU（如 RTX 4090）：24 GB
  - 数据中心 GPU（如 H100）：80-94 GB
  - 最新 GPU（如 H200）：141 GB
- **高吞吐量**：
  - RTX 4090：~1.0 TB/s
  - H100：3.35-3.9 TB/s
  - H200：4.8 TB/s
- **高延迟**

**优化策略：**
- 使用 **缓存** 来减少延迟
- 合理的内存访问模式（合并访问）
- 避免内存带宽浪费

---

## 三、关键概念

### 1. SIMT vs SIMD

- **SIMD（CPU）**：单指令多数据
- **SIMT（GPU）**：单指令多线程
- GPU 使用多个线程并行执行指令，而不是像 CPU 那样使用向量指令

### 2. Kernel（内核）

- GPU 程序的入口点
- 使用 `__global__` 限定符定义
- **异步执行**：主机代码在内核调用后继续运行

### 3. 线程 ID 计算

```cpp
int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
```

三个关键变量：
- `blockIdx`：块 ID
- `blockDim`：块中的线程总数
- `threadIdx`：线程块内的线程 ID

### 4. 同步机制

- `cudaDeviceSynchronize()`：阻塞主机执行，等待 GPU 完成
- 线程块内的同步使用 `__syncthreads()`

---

## 四、应用示例

### Hello World 程序

```cpp
__global__ void helloFromGPU() {
    printf("Hello World from Thread %d, Block %d, BlockDim %d\n",
           threadIdx.x, blockIdx.x, blockDim.x);
}

int main() {
    // 启动内核：2个块，每个块4个线程
    helloFromGPU<<<2, 4>>>();
    cudaDeviceSynchronize(); // 等待GPU完成
    return 0;
}
```

### 矩阵乘法内核

```cpp
__global__ void matrixMul(const float* A, const float* B, float* C, int n) {
    // 计算全局行和列索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float value = 0.0f;
        for (int k = 0; k < n; ++k) {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}
```

---

## 五、性能优化要点

### 1. 内存访问模式
- **合并访问**：确保相邻线程访问相邻内存
- **使用共享内存**：减少全局内存访问
- **数据局部性**：提高缓存命中率

### 2. 线程配置
- **合理选择 block size**：通常是 32 的倍数（warp size 的倍数）
- **避免分支分歧**：warp 内线程尽量执行相同路径
- **最大化占用率**：让足够的 warp 保持活跃以隐藏内存延迟

### 3. 同步开销
- 减少不必要的同步
- 使用异步复制（现代 GPU）

---

## 六、总结

NVIDIA GPU 的基础数据结构构成了高效的并行计算框架：

1. **分层执行模型**：Thread → Warp → Thread Block → Grid
2. **内存层次结构**：Registers → Shared Memory → Global Memory
3. **SIMT 架构**：通过多线程并行执行实现高性能
4. **CUDA 编程模型**：提供简洁的 API 来管理这些结构

理解这些基础结构对于编写高效的 GPU 程序至关重要。在实际应用中，需要根据具体问题合理配置线程块大小、优化内存访问模式，并充分利用共享内存来减少延迟。

---

## 参考资料

### 主要文档
- **[GPU Architecture and Programming — An Introduction](https://medium.com/@najeebkan/gpu-architecture-and-programming-an-introduction-561bfcb51f54)**
  作者：Najeeb Khan | 发布时间：2024年8月26日 | 阅读时间：5分钟

- **[GitHub 中文翻译版](https://github.com/ForceInjection/AI-fundermentals/blob/main/gpu_programming/gpu_programming_introduction.md)**
  ForceInjection/AI-fundermentals 仓库

### 官方文档
- **[NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)**
  NVIDIA 官方 CUDA 编程指南

- **[CUDA Programming Model - Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)**
  CUDA 编程模型官方文档

- **[PTX ISA 9.1 Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)**
  NVIDIA PTX 指令集架构文档

### 数据验证来源
- **[SIMT and Warps - Cornell Virtual Workshop](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/simt_warp)**
  验证：Warp size = 32 threads

- **[Memory Hierarchy of GPUs - ARC Compute](https://www.arccompute.io/arc-blog/gpu-101-memory-hierarchy)**
  验证：GPU 内存层次结构

- **[NVIDIA H100 GPU Specifications](https://www.nvidia.com/en-us/data-center/h100/)**
  验证：H100 内存带宽 3.35-3.9 TB/s

- **[NVIDIA H200 GPU Specifications](https://www.nvidia.com/en-us/data-center/h200/)**
  验证：H200 内存带宽 4.8 TB/s

- **[Basics on NVIDIA GPU Hardware Architecture - NASA HECC](https://www.nas.nasa.gov/hecc/support/kb/basics-on-nvidia-gpu-hardware-architecture_704.html)**
  验证：Shared memory 容量配置（48-96 KB per block）

- **[NVIDIA Ampere GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/archive/12.3.0/ampere-tuning-guide/index.html)**
  验证：Ampere 架构 shared memory 配置

### 其他学习资源
- **[Using Shared Memory in CUDA C/C++ - NVIDIA Developer Blog](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)**
  Shared memory 使用教程

- **[H100 vs. H200: Two Hopper-based Heavyweights - Vast.ai](https://vast.ai/article/nvidia-h100-vs-h200-two-hopper-based-heavyweights)**
  现代 GPU 性能对比

---

