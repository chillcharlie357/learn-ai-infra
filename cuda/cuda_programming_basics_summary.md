---
title: CUDA 编程基础总结
date: 2026-01-02 15:03
modified: 2026-01-02 15:03
tags:
  - cuda
  - cuda-c++
  - programming-guide
  - asynchronous-execution
  - kernels
  - memory-management
categories:
  - 技术分享
excerpt: 总结 CUDA 编程的核心概念，包括 Kernel 定义与启动、内存管理、异步执行、流和事件等关键主题。
mathjax: false
comment: true
---

# CUDA 编程基础总结

> 基于：[CUDA Programming Guide - Programming GPUs in CUDA](https://docs.nvidia.com/cuda/cuda-programming-guide/part2.html)

## 概述

本文档介绍了使用 CUDA C++ 进行 GPU 编程的基础知识，涵盖了从基本的 Kernel 定义和启动，到异步执行、流和事件等高级特性。重点关注实际编程中常用的概念和最佳实践。

---

## 一、CUDA C++ 基础

### 1.1 编译工具：NVCC

**NVCC（NVIDIA CUDA Compiler）** 是编译 CUDA C++ 代码的编译器驱动。

**基本用法：**
```bash
nvcc vecAdd.cu -o vecAdd
```

NVCC 简化了编译过程，自动处理不同的编译阶段，包括主机代码和设备代码的分离编译。

---

### 1.2 Kernel 定义与启动

#### 1.2.1 Kernel 定义

Kernel 是在 GPU 上执行的函数，使用 `__global__` 说明符定义：

```cpp
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}
```

**Kernel 特点：**
- 返回类型必须是 `void`
- 使用 `__global__` 说明符标识
- 由大量线程并行执行

#### 1.2.2 Kernel 启动（Triple Chevron Notation）

使用 `<<<>>>` 语法启动 Kernel：

```cpp
// 启动 4 个线程块，每个块 256 个线程
vecAdd<<<4, 256>>>(A, B, C, vectorLength);
```

**执行配置参数：**
```cpp
kernel<<<gridDim, blockDim, sharedMemSize, stream>>>(args...);
```

- **gridDim**：网格维度（线程块的数量）
- **blockDim**：线程块维度（每个块的线程数）
- **sharedMemSize**：动态共享内存大小（可选，默认 0）
- **stream**：关联的流（可选，默认为默认流）

**多维示例：**
```cpp
dim3 grid(16, 16);  // 16x16 的线程块网格
dim3 block(8, 8);   // 8x8 的线程块
MatAdd<<<grid, block>>>(A, B, C);
```

#### 1.2.3 线程索引计算

CUDA 提供内置变量用于计算线程索引：

```cpp
// 一维索引
int workIndex = threadIdx.x + blockIdx.x * blockDim.x;

// 二维索引
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int index = row * width + col;
```

**关键内置变量：**
- **`threadIdx`**：线程在线程块内的索引（0 到 blockDim-1）
- **`blockIdx`**：线程块在网格内的索引（0 到 gridDim-1）
- **`blockDim`**：线程块的维度
- **`gridDim`**：网格的维度

#### 1.2.4 边界检查

为了处理任意长度的向量，需要添加边界检查：

```cpp
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;

    // 边界检查：防止越界访问
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}
```

**计算线程块数量：**
```cpp
int threads = 256;
int blocks = (vectorLength + threads - 1) / threads;  // 向上取整
vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
```

或使用 CUDA Core Compute Library (CCCL)：
```cpp
#include <cuda/cmath>

int threads = 256;
int blocks = cuda::ceil_div(vectorLength, threads);
vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
```

---

### 1.3 内存管理

CUDA 提供两种主要的内存管理方式：**统一内存（Unified Memory）**和**显式内存管理（Explicit Memory Management）**。

#### 1.3.1 统一内存（Unified Memory）

统一内存由 NVIDIA 驱动自动管理数据在主机和设备间的迁移。

**分配统一内存：**
```cpp
float* A = nullptr;
cudaMallocManaged(&A, vectorLength * sizeof(float));

// 使用内存
initArray(A, vectorLength);
vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
cudaDeviceSynchronize();

// 释放内存
cudaFree(A);
```

**优点：**
- 简化代码，无需手动管理数据传输
- 适合快速开发和原型验证
- 在支持 ATS 的 Linux 系统上性能优异

#### 1.3.2 显式内存管理

显式管理内存分配和数据传输，可提供更好的性能控制。

**分配和复制内存：**
```cpp
// 分配主机内存（页锁定，提高传输性能）
float* A = nullptr;
cudaMallocHost(&A, vectorLength * sizeof(float));

// 分配设备内存
float* devA = nullptr;
cudaMalloc(&devA, vectorLength * sizeof(float));

// 初始化主机数据
initArray(A, vectorLength);

// 复制数据到设备
cudaMemcpy(devA, A, vectorLength * sizeof(float), cudaMemcpyHostToDevice);

// 启动 Kernel
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
cudaDeviceSynchronize();

// 复制结果回主机
cudaMemcpy(C, devC, vectorLength * sizeof(float), cudaMemcpyDeviceToHost);

// 清理
cudaFree(devA);
cudaFreeHost(A);
```

**`cudaMemcpy` 类型：**
- `cudaMemcpyHostToDevice`：主机到设备
- `cudaMemcpyDeviceToHost`：设备到主机
- `cudaMemcpyDeviceToDevice`：设备到设备
- `cudaMemcpyDefault`：自动推断类型

**最佳实践：**
- 使用 `cudaMallocHost` 分配用于数据传输的主机内存
- 只在必要时使用页锁定内存（避免锁定过多内存）
- 显式内存管理提供更精细的性能优化控制

---

### 1.4 CPU 与 GPU 同步

Kernel 启动是**异步**的，需要同步以确保 Kernel 完成执行。

**最简单的同步方法：**
```cpp
vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
cudaDeviceSynchronize();  // 阻塞主机，等待 GPU 完成
```

**更细粒度的同步（使用流和事件）：**
参见"异步执行"章节。

---

### 1.5 错误检查

每个 CUDA API 都返回 `cudaError_t` 类型的错误码。

**错误检查宏：**
```cpp
#define CUDA_CHECK(expr) do {                        \
    cudaError_t result = expr;                       \
    if(result != cudaSuccess)                        \
    {                                                 \
        fprintf(stderr,                              \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__, __LINE__, result,          \
                cudaGetErrorString(result));         \
    }                                                 \
} while(0)
```

**使用示例：**
```cpp
CUDA_CHECK(cudaMalloc(&devA, vectorLength * sizeof(float)));

vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
CUDA_CHECK(cudaGetLastError());  // 检查 Kernel 启动错误
CUDA_CHECK(cudaDeviceSynchronize());  // 检查执行错误
```

**错误状态函数：**
- **`cudaGetLastError()`**：返回并清除最后的错误
- **`cudaPeekLastError()`**：返回但不清除最后的错误

**调试环境变量：**
```bash
CUDA_LAUNCH_BLOCKING=1  # 每个 Kernel 启动后同步，便于定位错误
CUDA_LOG_FILE=cudaLog.txt  # 记录详细错误日志
```

---

### 1.6 函数和变量说明符

#### 函数说明符

| 说明符 | 含义 |
|--------|------|
| `__global__` | Kernel 函数，从主机调用，在设备上执行 |
| `__device__` | 设备函数，从其他 `__device__` 或 `__global__` 函数调用 |
| `__host__` | 主机函数（默认） |
| `__host__ __device__` | 同时为主机和设备生成代码 |

#### 变量说明符

| 说明符 | 存储位置 |
|--------|----------|
| `__device__` | 全局内存（Global Memory） |
| `__constant__` | 常量内存（Constant Memory） |
| `__managed__` | 统一内存（Unified Memory） |
| `__shared__` | 共享内存（Shared Memory） |

#### 检测设备编译

```cpp
__host__ __device__ void func()
{
#ifdef __CUDA_ARCH__
    // 设备代码
    printf("Running on GPU\n");
#else
    // 主机代码
    printf("Running on CPU\n");
#endif
}
```

---

### 1.7 线程块集群（Thread Block Clusters）

从 Compute Capability 9.0 开始，CUDA 引入了**线程块集群**的概念，允许多个线程块在 GPU Processing Cluster (GPC) 上协同调度。

**使用编译时属性启用集群：**
```cpp
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{
    // Kernel 代码
}

int main()
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // 网格维度必须是集群大小的倍数
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```

**集群特性：**
- 最多支持 8 个线程块组成一个集群
- 集群中的线程块保证在单个 GPC 上同时执行
- 支持**分布式共享内存**（Distributed Shared Memory）
- 使用 Cooperative Groups API 进行集群级同步

---

## 二、异步执行

CUDA 提供强大的异步执行能力，允许主机、设备计算和数据传输并行进行。

### 2.1 CUDA 流（CUDA Streams）

**流**是按顺序执行的操作序列。一个流就像一个工作队列，操作按入队顺序执行。

#### 2.1.1 创建和销毁流

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);  // 创建流

// 使用流执行操作

cudaStreamDestroy(stream);  // 销毁流
```

#### 2.1.2 在流中启动 Kernel

```cpp
kernel<<<grid, block, sharedMemSize, stream>>>(args...);
```

#### 2.1.3 在流中执行异步内存传输

```cpp
// 异步复制：立即返回，在流中执行
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
```

**注意：** 为了实现真正的异步传输，主机内存必须是**页锁定**（page-locked）的：
```cpp
float* h_data;
cudaMallocHost(&h_data, size);  // 分配页锁定内存
```

#### 2.1.4 流同步

**阻塞同步：**
```cpp
cudaStreamSynchronize(stream);  // 等待流中所有操作完成
```

**非阻塞查询：**
```cpp
cudaError_t status = cudaStreamQuery(stream);
if(status == cudaSuccess)
{
    // 流为空，所有操作已完成
}
else if(status == cudaErrorNotReady)
{
    // 流仍在执行
}
```

---

### 2.2 CUDA 事件（CUDA Events）

**事件**是插入到流中的标记，用于跟踪操作进度和建立依赖关系。

#### 2.2.1 创建和销毁事件

```cpp
cudaEvent_t event;
cudaEventCreate(&event);  // 创建事件

// 使用事件

cudaEventDestroy(event);  // 销毁事件
```

#### 2.2.2 记录事件

```cpp
cudaEventRecord(event, stream);  // 在流中插入事件
```

#### 2.2.3 计时操作

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// 记录开始时间
cudaEventRecord(start, stream);

// 启动 Kernel
kernel<<<grid, block, 0, stream>>>(...);

// 记录结束时间
cudaEventRecord(stop, stream);

// 等待流完成
cudaStreamSynchronize(stream);

// 获取耗时
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
printf("Kernel execution time: %.3f ms\n", elapsedTime);

// 清理
cudaEventDestroy(start);
cudaEventDestroy(stop);
cudaStreamDestroy(stream);
```

#### 2.2.4 同步事件

**阻塞同步：**
```cpp
kernel<<<grid, block, 0, stream>>>(...);
cudaEventRecord(event, stream);
kernel2<<<grid, block, 0, stream>>>(...);

// 等待第一个 Kernel 完成
cudaEventSynchronize(event);
dependentCPUTask();  // 现在可以安全执行
```

**非阻塞查询：**
```cpp
if(cudaEventQuery(event) == cudaSuccess)
{
    // 事件已完成
    cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2);
}
```

---

### 2.3 流优先级

可以为流分配优先级，影响调度顺序。

**查询优先级范围：**
```cpp
int minPriority, maxPriority;
cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority);
```

**创建优先级流：**
```cpp
cudaStream_t stream1, stream2;

// 创建低优先级流
cudaStreamCreateWithPriority(&stream1, cudaStreamDefault, minPriority);

// 创建高优先级流
cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, maxPriority);
```

**注意：** 优先级只是提示，不保证执行顺序，主要用于 Kernel 调度。

---

### 2.4 默认流行为

#### 2.4.1 遗留默认流（Legacy Default Stream）

- 默认流（NULL 流，stream ID 0）是**阻塞流**
- 阻塞流会与其他所有阻塞流同步

**问题示例：**
```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

kernel1<<<grid, block, 0, stream1>>>(...);  // 流 1
kernel2<<<grid, block>>>(...);              // 默认流，会等待 stream1
kernel3<<<grid, block, 0, stream2>>>(...);  // 流 2，会等待 kernel2
```

#### 2.4.2 非阻塞流

创建非阻塞流避免与默认流同步：

```cpp
cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
```

#### 2.4.3 每线程默认流（Per-thread Default Stream）

从 CUDA 7 开始，每个主机线程可以有独立的默认流：

**启用方法：**
```bash
# 编译选项
nvcc --default-stream per-thread file.cu
```

或定义宏：
```cpp
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
```

---

### 2.5 CUDA Graphs 简介

对于需要重复执行的相同操作序列，CUDA Graphs 可以减少启动开销。

**使用 Stream Capture 创建 Graph：**
```cpp
cudaGraph_t graph;
cudaGraphExec_t instance;

bool graphCreated = false;

for(int i = 0; i < NSTEPS; i++)
{
    if(!graphCreated)
    {
        // 捕获 Graph
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        // 启动多个 Kernel
        for(int j = 0; j < NKERNEL; j++)
        {
            shortKernel<<<blocks, threads, 0, stream>>>(...);
        }

        // 结束捕获
        cudaStreamEndCapture(stream, &graph);

        // 实例化 Graph
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
        graphCreated = true;
    }

    // 启动 Graph
    cudaGraphLaunch(instance, stream);
    cudaStreamSynchronize(stream);
}
```

**CUDA Graphs 优势：**
- 减少 CPU 开销
- 减少启动延迟
- 适用于需要重复执行的固定操作序列

---

## 三、异步执行最佳实践

### 3.1 提高并发性的原则

1. **在依赖操作之前发出独立操作**
2. **尽可能延迟同步**
3. **使用多个流重叠计算和数据传输**

### 3.2 示例：重叠计算与数据传输

```cpp
cudaStream_t computeStream, copyStream;
cudaStreamCreate(&computeStream);
cudaStreamCreate(&copyStream);

cudaEvent_t event;
cudaEventCreate(&event);

// 启动第一个 Kernel
kernel1<<<grid, block, 0, computeStream>>>(data1);
cudaEventRecord(event, computeStream);

// 在 computeStream 中启动第二个 Kernel
kernel2<<<grid, block, 0, computeStream>>>(data2);

// 在 CPU 上执行工作
while(!allCPUWorkDone())
{
    doNextChunkOfCPUWork();

    // 检查 kernel1 是否完成，如果是则启动异步复制
    if(cudaEventQuery(event) == cudaSuccess && !copyStarted)
    {
        cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, copyStream);
        copyStarted = true;
    }
}

// 等待所有流完成
cudaStreamSynchronize(computeStream);
cudaStreamSynchronize(copyStream);
```

---

## 四、总结

### 4.1 核心 API

| 类别 | API | 用途 |
|------|-----|------|
| **Kernel 启动** | `<<<>>>` | 启动 GPU Kernel |
| **内存分配** | `cudaMalloc` / `cudaMallocManaged` | 分配设备/统一内存 |
| **内存复制** | `cudaMemcpy` / `cudaMemcpyAsync` | 同步/异步内存传输 |
| **同步** | `cudaDeviceSynchronize` | 等待 GPU 完成 |
| **流管理** | `cudaStreamCreate` / `cudaStreamDestroy` | 创建/销毁流 |
| **事件管理** | `cudaEventCreate` / `cudaEventRecord` | 创建/记录事件 |
| **错误检查** | `cudaGetLastError` / `cudaGetErrorString` | 获取错误信息 |

### 4.2 关键概念

1. **Kernel 编程**
   - 使用 `__global__` 定义 Kernel
   - 通过 `<<<>>>` 配置执行参数
   - 使用 `threadIdx`、`blockIdx` 计算索引
   - 添加边界检查确保安全性

2. **内存管理**
   - **统一内存**：简化编程，适合原型开发
   - **显式管理**：精细控制，优化性能
   - 使用页锁定内存提高传输效率

3. **异步执行**
   - **流（Streams）**：按顺序执行的操作序列
   - **事件（Events）**：标记和同步点
   - **优先级**：影响调度顺序
   - **CUDA Graphs**：减少重复操作的启动开销

4. **错误处理**
   - 始终检查 API 返回值
   - 使用环境变量辅助调试
   - 区分启动时错误和执行时错误

### 4.3 学习路径

- **入门**：掌握 Kernel 定义和启动、基本内存管理
- **进阶**：学习异步执行、流和事件
- **高级**：CUDA Graphs、多 GPU 编程、性能优化

---

## 参考资料

### 主要文档
- **[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)**
  完整的 CUDA 编程指南

- **[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)**
  完整的 CUDA C++ 编程指南

### 相关章节
- **[2.1. Intro to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)**
  CUDA C++ 基础介绍

- **[2.3. Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)**
  异步执行详解

### 扩展阅读
- **[CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)**
  CUDA 运行时 API 参考

- **[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)**
  CUDA 最佳实践指南

---

> **本文档基于 CUDA Programming Guide v13.1 总结，涵盖了 CUDA 编程的核心内容。**
