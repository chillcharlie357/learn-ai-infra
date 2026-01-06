---
title: CUDA 共享内存详解
date: 2026-01-04 22:14
modified: 2026-01-06 22:30
tags:
  - cuda
  - shared-memory
  - gpu-architecture
  - performance-optimization
  - parallel-computing
  - hardware-limits
categories:
  - 技术分享
excerpt: 深入讲解CUDA共享内存的使用方法、优化技巧和实际应用场景，包括矩阵乘法、归约操作等经典案例，以及CUDA硬件限制常数详解。
mathjax: true
comment: true
---

# CUDA 共享内存详解

> 基于：[CUDA C++ Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)

## 目录

- [概述](#概述)
- [CUDA硬件限制常数](#cuda硬件限制常数)
- [一、共享内存基础](#一共享内存基础)
- [二、经典应用：矩阵乘法优化](#二经典应用矩阵乘法优化)
- [三、高级应用](#三高级应用)
- [四、性能优化最佳实践](#四性能优化最佳实践)
- [五、常见问题与解决方案](#五常见问题与解决方案)
- [六、性能提升示例对比](#六性能提升示例对比)
- [七、总结](#七总结)

## 概述

**共享内存（Shared Memory）**是CUDA编程中最强大的性能优化工具之一。它是GPU芯片上的高速用户管理缓存，可以被同一个线程块（Thread Block）内的所有线程共享访问。

### 共享内存的关键特性

| 特性 | 说明 |
|------|------|
| **速度** | 比全局内存快约 **20-100倍**，延迟 ~25-35 cycles |
| **带宽** | 极高带宽（H100: ~33 TB/s）[¹] |
| **作用域** | 线程块级别，块内所有线程共享 |
| **生命周期** | 线程块执行期间存在，块结束时销毁 |
| **容量** | 架构相关（Fermi/Kepler: 48KB, H100: 256KB/SM）[²] |
| **管理方式** | 程序员显式管理，用户可编程缓存 |

### 为什么使用共享内存？

**问题**：现代GPU的计算速度远快于全局内存访问速度
- 全局内存带宽：2.0-3.35 TB/s（H100，取决于配置）[³]
- L1/共享内存带宽：~33 TB/s（H100）[¹]
- GPU计算能力：67 TFLOPS（FP32，H100 SXM）[⁴]
- **Roofline模型**：许多kernel是内存带宽受限的

**解决方案**：使用共享内存缓存频繁访问的数据
$$ \text{性能提升} \approx \frac{\text{共享内存带宽}}{\text{全局内存带宽}} \times \text{数据复用率} = \frac{33\text{ TB/s}}{2\text{ TB/s}} \times \text{数据复用率} $$

---

## CUDA硬件限制常数

在CUDA编程中，了解硬件限制对于编写高性能代码至关重要。本节总结了CUDA的关键硬件常数，这些常数直接影响kernel的性能和正确性。

### 核心硬件常数

#### 1. Warp相关

| 常数 | 值 | 说明 | 来源 |
|------|-----|------|------|
| **Warp Size** | **32 threads** | 所有现代NVIDIA GPU架构的标准warp大小 | [⁵] |
| **Warp Scheduling** | SIMT架构 | 一个warp内的32个线程执行相同指令（存在分化时部分线程屏蔽） | [⁵] |

**关键说明**：
- Warp是GPU调度的基本单位，所有线程以warp（32个线程）为单位执行
- 从Fermi架构到Hopper架构，warp size始终为32 threads
- CUDA库（如CUB）支持1-32线程的"逻辑warp"，但硬件warp始终是32线程

#### 2. Thread Block限制

| 限制项 | Compute Capability 1.x | Compute Capability 2.x - 9.x | 说明 |
|--------|----------------------|----------------------------|------|
| **Max Threads per Block** | 512 | **1024** | 单个block内的最大线程数 | [⁶] |
| **Max X-Dimension** | 512 | **1024** | blockIdx.x的最大值 | [⁷] |
| **Max Y-Dimension** | 512 | **1024** | blockIdx.y的最大值 | [⁷] |
| **Max Z-Dimension** | 64 | **64** | blockIdx.z的最大值 | [⁷] |
| **Max Registers per Thread** | 63 (CC 1.0-1.2)<br>127 (CC 1.3) | **255** | 单个线程可用的最大寄存器数 | [⁸] |
| **Max Registers per Block** | 16K | **64K** (65,536) | 单个block可用的最大32位寄存器数 | [⁸] |

**重要提示**：
- Max threads per block = blockDim.x × blockDim.y × blockDim.z ≤ 1024
- 实际可用的threads per block会受到registers和shared memory的限制

#### 3. Grid限制

| 限制项 | Compute Capability 1.x - 2.x | Compute Capability 3.0 - 9.x | 说明 |
|--------|---------------------------|----------------------------|------|
| **Max Grid X-Dimension** | 65,535 | **2³¹ - 1** (2,147,483,647) | gridDim.x的最大值 | [⁹] |
| **Max Grid Y-Dimension** | 65,535 | **65,535** | gridDim.y的最大值 | [⁹] |
| **Max Grid Z-Dimension** | 65,535 | **65,535** | gridDim.z的最大值 | [⁹] |

**实践意义**：
- 现代GPU（CC 3.0+）可以在X维度启动超过20亿个block
- 对于大多数应用，grid大小不会成为瓶颈

#### 4. Shared Memory限制

| 架构 | Compute Capability | Shared Memory per Block | Shared Memory per SM | 数据来源 |
|------|-------------------|------------------------|---------------------|----------|
| **Fermi/Kepler** | 2.x - 3.x | 48 KB | 48 KB / 16 KB L1 | [¹⁰] |
| **Pascal (GP100)** | 6.0 | 64 KB | 64 KB / 32 KB L1 | [¹⁰] |
| **Volta (V100)** | 7.0 | 96 KB | 96 KB / 32 KB L1 | [¹⁰] |
| **Ampere (A100)** | 8.0 | **164 KB** | 164 KB / 128 KB L1 | [¹¹] |
| **Hopper (H100)** | 9.0 | **227 KB** | 228 KB (L1/Shared统一) | [¹²] |

**说明**：
- "Shared Memory per Block"是单个线程块可分配的最大shared memory
- "Shared Memory per SM"是每个流多处理器（SM）的shared memory总容量
- 从Volta架构开始，shared memory和L1 cache可以动态配置
- Hopper架构采用统一L1/Shared Memory设计，总容量228 KB per SM

#### 5. Streaming Multiprocessor (SM) 资源限制

| 限制项 | A100 (CC 8.0) | H100 (CC 9.0) | 说明 |
|--------|---------------|---------------|------|
| **Max Threads per SM** | 2048 | 2048 | 单个SM可同时运行的最大线程数 | [¹³] |
| **Max Warps per SM** | 64 | 64 | 单个SM可同时运行的最大warp数 (2048÷32) | [¹³] |
| **Max Blocks per SM** | 32 | 32 | 单个SM可驻留的最大线程块数 | [⁸] |
| **Max Registers per SM** | 64K × 32 (保守估计) | 64K × 32 (保守估计) | 受限于块数量和每块寄存器数 | [⁸] |

**资源分配关系**：
```
实际可驻留的blocks per SM = min(
    Max Blocks per SM (32),
    floor(Max Threads per SM / Threads per Block),
    floor(Max Shared Memory per SM / Shared Memory per Block),
    floor(Max Registers per SM / Registers per Block)
)
```

### 查询设备属性的代码示例

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

void print_cuda_device_limits()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("========================================\n");
    printf("CUDA Device Limits: %s\n", prop.name);
    printf("========================================\n\n");

    // Warp和Thread信息
    printf("=== Warp & Thread Limits ===\n");
    printf("Warp Size: %d threads\n", prop.warpSize);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("\n");

    // Block维度限制
    printf("=== Block Dimension Limits ===\n");
    printf("Max Block Dimensions: [%d, %d, %d]\n",
           prop.maxThreadsDim[0],
           prop.maxThreadsDim[1],
           prop.maxThreadsDim[2]);
    printf("\n");

    // Grid维度限制
    printf("=== Grid Dimension Limits ===\n");
    printf("Max Grid Dimensions: [%d, %d, %d]\n",
           prop.maxGridSize[0],
           prop.maxGridSize[1],
           prop.maxGridSize[2]);
    printf("\n");

    // Shared Memory
    printf("=== Shared Memory Limits ===\n");
    printf("Shared Memory per Block: %zu KB\n",
           prop.sharedMemPerBlock / 1024);
    printf("Shared Memory per SM: %zu KB\n",
           prop.sharedMemPerMultiprocessor / 1024);
    printf("\n");

    // Registers
    printf("=== Register Limits ===\n");
    printf("Registers per Block: %d (32-bit)\n", prop.regsPerBlock);
    printf("Max Registers per Thread: %d\n", 255);  // 固定值
    printf("Registers per SM: %d (32-bit)\n", prop.regsPerMultiprocessor);
    printf("\n");

    // SM资源
    printf("=== SM Resources ===\n");
    printf("Max Blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Max Warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("\n");

    // Clock rates
    printf("=== Clock Rates ===\n");
    printf("Clock Rate: %.2f GHz\n", prop.clockRate / 1e6);
    printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("\n");

    // Bandwidth
    float memory_bandwidth = (prop.memoryClockRate * 1e3) *
                            (prop.memoryBusWidth / 8) / 1e9;
    printf("Memory Bandwidth: %.2f GB/s\n", memory_bandwidth);

    printf("========================================\n");
}

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if(deviceCount == 0)
    {
        printf("No CUDA devices found!\n");
        return 1;
    }

    for(int i = 0; i < deviceCount; i++)
    {
        cudaSetDevice(i);
        print_cuda_device_limits();
    }

    return 0;
}
```

**示例输出（A100）**：
```
========================================
CUDA Device Limits: NVIDIA A100-SXM4-80GB
========================================

=== Warp & Thread Limits ===
Warp Size: 32 threads
Max Threads per Block: 1024
Max Threads per SM: 2048

=== Block Dimension Limits ===
Max Block Dimensions: [1024, 1024, 64]

=== Grid Dimension Limits ===
Max Grid Dimensions: [2147483647, 65535, 65535]

=== Shared Memory Limits ===
Shared Memory per Block: 163 KB
Shared Memory per SM: 164 KB

=== Register Limits ===
Registers per Block: 65536 (32-bit)
Max Registers per Thread: 255
Registers per SM: 65536 (32-bit)

=== SM Resources ===
Max Blocks per SM: 32
Max Warps per SM: 64
Number of SMs: 108
Compute Capability: 8.0

=== Clock Rates ===
Clock Rate: 1.41 GHz
Memory Clock Rate: 1.59 GHz
Memory Bus Width: 5120 bits

Memory Bandwidth: 2039.00 GB/s
========================================
```

### 实践建议

#### 1. 选择合适的Block大小

**推荐配置**：
```cpp
// 推荐的block大小（2的幂次，≤1024）
dim3 block1d(256);      // 1D: 256 threads
dim3 block2d(16, 16);   // 2D: 256 threads
dim3 block3d(8, 8, 8);  // 3D: 512 threads
```

**原则**：
- Block size应该是warp size（32）的倍数，以避免warp内的空闲线程
- 常用大小：128, 256, 512, 1024 threads
- 避免使用非2的幂次大小的block（除非特殊需求）

#### 2. Shared Memory分配策略

```cpp
// 查询可用的shared memory
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
size_t max_shared_mem = prop.sharedMemPerBlock;

// 安全分配shared memory（留有余量）
size_t safe_shared_mem = max_shared_mem * 0.9;  // 使用90%
__shared__ float data[(int)(safe_shared_mem / sizeof(float))];

// 或使用动态shared memory
extern __shared__ float dynamic_shared[];
```

#### 3. 避免超出硬件限制

**错误示例**：
```cpp
// 错误：超过1024 threads per block
dim3 block_too_large(32, 32, 2);  // 32×32×2 = 2048 threads ✗

// 错误：超过shared memory限制
__shared__ float huge_array[100000];  // 可能超过per-block限制 ✗
```

**正确示例**：
```cpp
// 正确：≤1024 threads per block
dim3 block_ok(32, 32);  // 32×32 = 1024 threads ✓

// 正确：动态分配shared memory并查询限制
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
size_t shared_mem_size = 100000 * sizeof(float);
if(shared_mem_size <= prop.sharedMemPerBlock)
{
    kernel<<<grid, block, shared_mem_size>>>(...);
}
else
{
    printf("Error: Requested shared memory exceeds limit!\n");
}
```

### 常用快速查询表

| 查询内容 | CUDA API | 说明 |
|---------|----------|------|
| Warp Size | `prop.warpSize` | 固定为32 |
| Max Threads per Block | `prop.maxThreadsPerBlock` | 通常1024 |
| Max Threads per SM | `prop.maxThreadsPerMultiProcessor` | 通常2048 |
| Max Blocks per SM | `prop.maxBlocksPerMultiProcessor` | 通常32 |
| Shared Memory per Block | `prop.sharedMemPerBlock` | 架构相关 |
| Shared Memory per SM | `prop.sharedMemPerMultiprocessor` | 架构相关 |
| Registers per Block | `prop.regsPerBlock` | 通常65536 |
| Compute Capability | `prop.major, prop.minor` | 架构版本 |

---

## 一、共享内存基础

### 1.1 声明共享内存

#### 静态声明（编译时确定大小）

```cpp
__global__ void static_shared_example()
{
    // 声明静态共享内存数组
    __shared__ float sharedData[256];
    __shared__ int tile[32][32];

    int tid = threadIdx.x;

    // 写入共享内存
    sharedData[tid] = tid * 1.0f;

    // 同步：确保所有线程完成写入
    __syncthreads();

    // 读取其他线程写入的数据
    float value = sharedData[(tid + 1) % 256];
}
```

**特点**：
- 编译时大小已知
- 编译器可以优化访问模式
- 适合大小固定的场景

#### 动态声明（运行时确定大小）

```cpp
__global__ void dynamic_shared_example(float* input, int size)
{
    // 声明动态共享内存（extern关键字）
    extern __shared__ float dynamicShared[];

    int tid = threadIdx.x;

    // 写入
    if(tid < size)
    {
        dynamicShared[tid] = input[tid];
    }

    __syncthreads();

    // 读取并处理
    if(tid < size)
    {
        input[tid] = dynamicShared[tid] * 2.0f;
    }
}

// 启动时指定动态共享内存大小（字节）
int main()
{
    int blockSize = 256;
    int sharedMemSize = blockSize * sizeof(float);

    // 第三个参数：动态共享内存大小
    dynamic_shared_example<<<1, blockSize, sharedMemSize>>>(d_data, blockSize);
}
```

**特点**：
- 运行时灵活配置大小
- 通用性强，一个kernel适配多种场景
- 需要在kernel启动时指定大小

---

### 1.2 线程同步：`__syncthreads()`

**`__syncthreads()`** 是线程块级别的同步屏障，确保所有线程都执行到同步点后再继续。

```cpp
__global__ void sync_example()
{
    __shared__ float data[256];
    int tid = threadIdx.x;

    // === 阶段1：写入数据 ===
    data[tid] = tid * 2.0f;

    // 同步：确保所有线程完成写入
    __syncthreads();

    // === 阶段2：读取数据 ===
    // 此时可以安全地读取其他线程写入的数据
    float sum = data[tid] + data[(tid + 1) % 256];

    // 再次同步
    __syncthreads();

    // === 阶段3：处理数据 ===
    data[tid] = sum * 0.5f;
}
```

**重要注意事项**：

1. **所有线程必须到达同步点**
```cpp
// 错误示例：条件分支中的同步
__global__ void bad_sync()
{
    __shared__ float data[256];
    int tid = threadIdx.x;

    data[tid] = tid;

    // 错误！如果tid < 128，这128个线程会死锁等待其他128个线程
    if(tid < 128)
    {
        __syncthreads();  // 危险！
    }

    // 正确做法：确保所有线程都能执行到同步点
    if(tid < 128)
    {
        data[tid] *= 2.0f;
    }
    __syncthreads();  // 正确：所有线程都会执行
}
```

2. **Warp分化不会影响同步**
```cpp
// 不会有问题：虽然条件执行会导致warp分化，但最终所有线程都会到达同步点
__global__ void conditional_sync()
{
    int tid = threadIdx.x;

    if(tid % 2 == 0)
    {
        // 偶数线程的工作
    }
    else
    {
        // 奇数线程的工作
    }

    // 所有线程（无论奇偶）都会到达这里
    __syncthreads();
}
```

---

### 1.3 共享内存布局与Bank访问

#### 共享内存的Bank结构

共享内存被分为 **32个Bank**（对应一个Warp的32个线程），每个Bank每个周期可以服务一个访问。

**理想情况**：32个线程访问32个不同的Bank → 所有访问并行执行
**Bank Conflict**：多个线程访问同一个Bank → 串行执行，性能下降

#### 访问模式分析

```cpp
// ===== 情况1：无冲突（最优） =====
__global__ void no_conflict()
{
    __shared__ float data[32];
    int tid = threadIdx.x;

    // 相邻线程访问相邻地址（32个bank）
    // tid=0 → bank 0, tid=1 → bank 1, ..., tid=31 → bank 31
    float x = data[tid];  // ✓ 无冲突
}

// ===== 情况2：严重冲突 =====
__global__ void severe_conflict()
{
    __shared__ float data[32];
    int tid = threadIdx.x;

    // Warp中所有32个线程访问同一个bank（bank 0）
    // 需要32个周期才能完成所有访问
    float x = data[tid * 32];  // ✗ 32-way bank conflict
}

// ===== 情况3：通过填充避免冲突 =====
__global__ void padding_solution()
{
    // 添加一列填充，使每行跨32个bank
    __shared__ float data[32][33];  // 33列而不是32列

    int row = threadIdx.y;
    int col = threadIdx.x;

    data[row][col] = threadIdx.x;

    __syncthreads();

    // 现在访问不会冲突
    float x = data[row][col];  // ✓ 无冲突
}
```

**为什么33列能避免冲突？**

对于二维数组 `matrix[row][col]`，访问时展平为：
$$ \text{Index} = \text{row} \times \text{columns} + \text{col} $$
$$ \text{Bank ID} = \text{Index} \bmod 32 $$

**32列的情况**（有冲突）：
- `matrix[0][0]` → Index = 0×32 + 0 = 0 → Bank 0
- `matrix[1][0]` → Index = 1×32 + 0 = 32 → Bank 0 (冲突！)
- `matrix[2][0]` → Index = 2×32 + 0 = 64 → Bank 0 (冲突！)
- 所有第0列的元素都在Bank 0

**33列的情况**（无冲突）：
- `matrix[0][0]` → Index = 0×33 + 0 = 0 → Bank 0
- `matrix[1][0]` → Index = 1×33 + 0 = 33 → Bank 1
- `matrix[2][0]` → Index = 2×33 + 0 = 66 → Bank 2
- `matrix[row][0]` → Bank = row，每个线程访问不同bank ✓

**核心原因**：**33与32互质**（gcd(33, 32) = 1），使得每行偏移到不同的bank。

#### Bank Conflict规则总结

| 访问步长 | Bank Conflict |
|---------|---------------|
| 1 | 无冲突（最优） |
| 2 | 2-way conflict |
| 4 | 4-way conflict |
| 8 | 8-way conflict |
| 16 | 16-way conflict |
| 32 | 32-way conflict（最差） |

**一般公式**：
$$ \text{Conflict Degree} = \gcd(\text{stride}, 32) $$
其中 `gcd` 是最大公约数，`stride` 是访问步长。

#### 深入理解：为什么 `tid * 32` 会导致32-way冲突？

##### Bank映射机制

CUDA共享内存被分成 **32个bank**（对应一个warp的32个线程），每个bank宽度为4字节。

**Bank计算公式**：
$$ \text{Bank ID} = \left\lfloor \frac{\text{byte\_address}}{4} \right\rfloor \bmod 32 $$

或者更直观的理解：
$$ \text{Bank ID} = (\text{float\_index}) \bmod 32 $$

##### 逐线程分析 `tid * 32`

```cpp
__shared__ float data[1024];  // 足够大的数组
int tid = threadIdx.x;        // tid = 0, 1, 2, ..., 31
float x = data[tid * 32];    // 为什么会冲突？
```

让我们逐个线程分析其访问模式：

| tid | 访问元素 | Float Index | 地址偏移（字节） | Bank ID 计算 | Bank ID |
|-----|---------|-------------|----------------|-------------|---------|
| 0   | data[0]   | 0           | 0              | 0 % 32      | **0**   |
| 1   | data[32]  | 32          | 128            | 32 % 32     | **0**   |
| 2   | data[64]  | 64          | 256            | 64 % 32     | **0**   |
| 3   | data[96]  | 96          | 384            | 96 % 32     | **0**   |
| 4   | data[128] | 128         | 512            | 128 % 32    | **0**   |
| ... | ...      | ...         | ...            | ...         | ...     |
| 31  | data[992] | 992         | 3968           | 992 % 32    | **0**   |

**结论**：所有32个线程都访问 **Bank 0** → **32-way bank conflict**

##### 可视化对比

**情况1：无冲突（步长=1）**
```cpp
__global__ void no_conflict()
{
    __shared__ float data[32];
    int tid = threadIdx.x;
    float x = data[tid];  // 步长 = 1
}
```

访问分布：
```
data[0]  → Bank 0  ← thread 0  ✓
data[1]  → Bank 1  ← thread 1  ✓
data[2]  → Bank 2  ← thread 2  ✓
...
data[31] → Bank 31 ← thread 31 ✓

所有32个线程访问32个不同的bank → 完美并行！
```

**情况2：32-way冲突（步长=32）**
```cpp
__global__ void severe_conflict()
{
    __shared__ float data[1024];
    int tid = threadIdx.x;
    float x = data[tid * 32];  // 步长 = 32
}
```

访问分布：
```
data[0]   → Bank 0  ← thread 0  ✓
data[32]  → Bank 0  ← thread 1  ✗ 冲突！
data[64]  → Bank 0  ← thread 2  ✗ 冲突！
data[96]  → Bank 0  ← thread 3  ✗ 冲突！
...
data[992] → Bank 0  ← thread 31 ✗ 冲突！

所有32个线程争用Bank 0 → 需要串行执行32次！
```

##### 性能影响

**无冲突 vs 32-way冲突的执行时间对比**：

| 访问模式 | Bank Conflict | 执行周期 | 相对性能 |
|---------|--------------|---------|---------|
| `data[tid]` | 无冲突 | ~30 cycles | 1.0x (基准) |
| `data[tid * 2]` | 2-way | ~60 cycles | 0.5x |
| `data[tid * 4]` | 4-way | ~120 cycles | 0.25x |
| `data[tid * 8]` | 8-way | ~240 cycles | 0.125x |
| `data[tid * 16]` | 16-way | ~480 cycles | 0.0625x |
| `data[tid * 32]` | 32-way | ~960 cycles | **0.03125x** |

**结论**：32-way bank conflict 会使共享内存访问性能降低 **32倍**！

##### 实际案例：二维数组的访问冲突

```cpp
// 问题场景
__global__ void matrix_row_access()
{
    // 32x32的矩阵，每行32个元素
    __shared__ float matrix[32][32];

    int row = threadIdx.y;  // 0-31
    int col = threadIdx.x;  // 0-31

    // 每个线程负责一列，访问同一列的不同行
    matrix[row][col] = threadIdx.x;  // 写入
    __syncthreads();

    // 问题：当row相同时，所有线程访问同一列的不同元素
    // 例如：row=0时，32个线程访问
    //   matrix[0][0], matrix[0][1], ..., matrix[0][31]
    // 这些元素的地址是连续的，理论上应该无冲突？
    // 但如果访问模式变成：
    float x = matrix[col][row];  // 转置访问 → 可能冲突！
}

// 解决方案：使用填充
__global__ void matrix_padded_access()
{
    // 每行33个元素（多1列填充）
    __shared__ float matrix[32][33];

    int row = threadIdx.y;
    int col = threadIdx.x;

    matrix[row][col] = threadIdx.x;
    __syncthreads();

    // 现在32个线程访问32个不同的bank
    float x = matrix[col][row];  // 无冲突！
}
```

**关键要点**：
- 行优先存储中，`matrix[row][col]` 的 `col` 维度访问通常是连续的
- 但当访问模式改变（如转置访问、跨步访问）时，可能产生冲突
- 添加填充（Pad到33列）可以打破对齐模式，避免冲突

---

## 二、经典应用：矩阵乘法优化

### 2.1 问题定义

计算矩阵乘法：$C = A \times B$

**矩阵维度**：
- $A$: $M \times K$ 矩阵
- $B$: $K \times N$ 矩阵
- $C$: $M \times N$ 结果矩阵

$$ C_{ij} = \sum_{k=0}^{K-1} A_{ik} \times B_{kj} $$

**示例：$2048 \times 1024$ 乘以 $1024 \times 512$**
```
A (2048x1024)    B (1024x512)    C (2048x512)
┌──────────┐     ┌────────┐      ┌────────┐
│  2048行  │  ×  │ 1024行 │  =   │ 2048行 │
│  1024列  │     │ 512列  │      │ 512列  │
└──────────┘     └────────┘      └────────┘
```

**朴素实现**的问题：
- 每个元素 $A_{ik}$ 和 $B_{kj}$ 需要从全局内存加载 $K$ 次
- 对于 $2048 \times 1024$ 乘以 $1024 \times 512$，每个A元素重复加载512次！

### 2.2 使用共享内存的分块算法

**核心思想**：将矩阵分成小块（Tile），加载到共享内存中复用

```
A (M x K)              B (K x N)
┌────────────┐         ┌────────────┐
│  Tile A0   │   ...   │  Tile B0   │
│  (16x16)   │         │  (16x16)   │
├────────────┤         ├────────────┤
│  Tile A1   │   ...   │  Tile B1   │
│  (16x16)   │         │  (16x16)   │
└────────────┘         └────────────┘
      ...                    ...
```

**分块策略**：
- 将矩阵A按行分成 $(M/16) \times (K/16)$ 个tile
- 将矩阵B按列分成 $(K/16) \times (N/16)$ 个tile
- 每个tile大小：$16 \times 16 = 256$ 元素
- 每次加载一个tile到共享内存，计算部分积

**性能提升分析**：
- 原始全局内存访问：$O(M \times K \times N)$
- 优化后全局内存访问：$O(M \times K \times N / \text{TILE\_SIZE})$
- **理论加速比**：约 **TILE_SIZE倍**（假设数据完全复用）

### 2.3 完整实现代码

```cpp
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16

// ===== 朴素实现（基准） =====
// C = A x B，其中 A是 MxK，B是 KxN，C是 MxN
__global__ void matrix_mul_naive(float* A, float* B, float* C, int M, int K, int N)
{
    // 计算当前线程负责的C矩阵元素坐标
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // C的行索引 (0 ~ M-1)
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // C的列索引 (0 ~ N-1)

    // 边界检查
    if(row < M && col < N)
    {
        float sum = 0.0f;
        // 计算 C[row][col] = sum(A[row][k] * B[k][col])
        for(int k = 0; k < K; k++)
        {
            // 每次访问都从全局内存读取
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ===== 共享内存优化版本 =====
__global__ void matrix_mul_tiled(float* A, float* B, float* C, int M, int K, int N)
{
    // 声明共享内存存储Tile
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    // 计算全局坐标
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // C的行索引
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // C的列索引

    float sum = 0.0f;

    // 遍历K维度的所有Tile
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for(int tile = 0; tile < numTiles; tile++)
    {
        // === 步骤1：加载数据到共享内存 ===
        int a_col = tile * TILE_SIZE + threadIdx.x;  // A的列索引 (0 ~ K-1)
        int b_row = tile * TILE_SIZE + threadIdx.y;  // B的行索引 (0 ~ K-1)

        // 加载A的tile: A[row][a_col]
        if(row < M && a_col < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        // 加载B的tile: B[b_row][col]
        if(b_row < K && col < N)
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        // 同步：确保所有线程完成加载
        __syncthreads();

        // === 步骤2：使用共享内存计算 ===
        for(int k = 0; k < TILE_SIZE; k++)
        {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        // 同步：确保所有线程完成计算，才能加载下一个Tile
        __syncthreads();
    }

    // 写回结果
    if(row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

// ===== 性能测试函数 =====
float test_performance(void (*kernel)(float*, float*, float*, int, int, int),
                       float* d_A, float* d_B, float* d_C,
                       int M, int K, int N, const char* name)
{
    // 配置Kernel启动参数
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    printf("%s: Time: %.3f ms\n", name, elapsed);

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed;
}

int main()
{
    // 定义非正方形矩阵维度
    int M = 2048;  // A的行数，C的行数
    int K = 1024;  // A的列数，B的行数
    int N = 512;   // B的列数，C的列数

    printf("Matrix Multiplication: A(%dx%d) x B(%dx%d) = C(%dx%d)\n", M, K, K, N, M, N);
    printf("Tile Size: %dx%d\n\n", TILE_SIZE, TILE_SIZE);

    // 分配内存
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    // 测试朴素版本
    float time_naive = test_performance(matrix_mul_naive, d_A, d_B, d_C, M, K, N, "Naive");

    // 测试共享内存优化版本
    float time_tiled = test_performance(matrix_mul_tiled, d_A, d_B, d_C, M, K, N, "Tiled (Shared Memory)");

    // 计算加速比
    printf("\nSpeedup: %.2fx\n", time_naive / time_tiled);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```

**预期输出**：
```
Matrix Multiplication: A(2048x1024) x B(1024x512) = C(2048x512)
Tile Size: 16x16

Naive: Time: 423.156 ms
Tiled (Shared Memory): Time: 38.234 ms

Speedup: 11.07x
```

---

## 三、高级应用

### 3.1 归约操作（Reduction）

归约是将数组转换为单个值的操作，如求和、求最大值等。

```cpp
__global__ void reduction_sum(float* input, float* output, int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // 加载数据到共享内存
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // 归约：相邻元素配对求和
    // 第一轮：tid=0 和 tid=16 求和, tid=1 和 tid=17 求和, ...
    // 第二轮：tid=0 和 tid=8 求和, tid=1 和 tid=9 求和, ...
    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 线程0写回结果
    if(tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

// 使用示例
void parallel_sum(float* d_input, float* d_output, int n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 每个block处理一部分数据
    reduction_sum<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(
        d_input, d_output, n
    );

    // 将各block的结果再次求和（可能需要多次迭代）
}
```

**性能优化技巧**：
1. **循环展开**：减少循环开销
2. **Warp shuffle**：现代GPU可以使用更高效的shuffle指令
3. **处理非2次幂大小**：添加边界处理

### 3.2 卷积操作（图像处理）

```cpp
#define FILTER_SIZE 5
#define TILE_SIZE (16 + FILTER_SIZE - 1)  // 考虑边缘像素

__global__ void convolution_2d(
    float* input, float* output, float* filter,
    int width, int height)
{
    // 共享内存：输入图像块（包含边缘）
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    // 全局坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 加载到共享内存（包含边缘像素）
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int in_x = x - FILTER_SIZE / 2;
    int in_y = y - FILTER_SIZE / 2;

    if(in_x >= 0 && in_x < width && in_y >= 0 && in_y < height)
        tile[ty][tx] = input[in_y * width + in_x];
    else
        tile[ty][tx] = 0.0f;

    __syncthreads();

    // 卷积计算
    if(x < width && y < height)
    {
        float sum = 0.0f;
        for(int fy = 0; fy < FILTER_SIZE; fy++)
        {
            for(int fx = 0; fx < FILTER_SIZE; fx++)
            {
                sum += tile[ty + fy][tx + fx] * filter[fy * FILTER_SIZE + fx];
            }
        }
        output[y * width + x] = sum;
    }
}
```

### 3.3 直方图计算

```cpp
#define HISTOGRAM_BINS 256

__global__ void histogram_kernel(unsigned char* input, unsigned int* output, int n)
{
    // 每个block独立的共享内存直方图
    __shared__ unsigned int private_hist[HISTOGRAM_BINS];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // 初始化共享内存
    private_hist[tid] = 0;
    if(tid + HISTOGRAM_BINS / 2 < HISTOGRAM_BINS)
        private_hist[tid + HISTOGRAM_BINS / 2] = 0;

    __syncthreads();

    // 计算直方图
    while(i < n)
    {
        unsigned char bin = input[i];
        atomicAdd(&private_hist[bin], 1);
        i += blockDim.x * gridDim.x;
    }

    __syncthreads();

    // 合并到全局内存
    if(tid < HISTOGRAM_BINS)
    {
        atomicAdd(&output[tid], private_hist[tid]);
    }
}
```

**注意**：直方图计算中使用 `atomicAdd` 避免冲突，但可能存在性能瓶颈。

---

## 四、性能优化最佳实践

### 4.1 共享内存使用原则

| 原则 | 说明 | 示例 |
|------|------|------|
| **最大化数据复用** | 数据加载后尽可能多次使用 | 矩阵乘法中每个Tile元素被使用16次 |
| **最小化同步次数** | `__syncthreads()` 有开销 | 合并多个阶段，减少同步点 |
| **避免Bank Conflicts** | 优化访问模式 | 使用填充或调整步长 |
| **合理分配容量** | 静态+动态共享内存总和受限制 | 根据算法需求平衡 |
| **边界检查** | 防止越界访问 | 所有共享内存访问都要检查 |

### 4.2 调试与性能分析

#### 使用CUDA工具检查共享内存使用

```bash
# 使用nvprof分析共享内存使用
nvprof --metrics shared_mem_utilization ./your_program

# 查看共享内存配置
nvprof --metrics shared_load_transactions_per_request ./your_program
nvprof --metrics shared_store_transactions_per_request ./your_program
```

#### 常见性能指标

```cpp
// 查询共享内存属性
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

printf("Shared Memory per Block: %zu KB\n",
       prop.sharedMemPerBlock / 1024);
printf("Max Threads per Block: %d\n",
       prop.maxThreadsPerBlock);
printf("Warp Size: %d\n", prop.warpSize);
```

### 4.3 静态与动态共享内存的选择

**静态共享内存**：
```cpp
__shared__ float tile[16][16];  // 1KB
```
- ✓ 编译时确定，代码清晰
- ✓ 编译器可以优化
- ✗ 灵活性差

**动态共享内存**：
```cpp
extern __shared__ float dynamic_mem[];  // 大小在启动时指定
```
- ✓ 运行时灵活配置
- ✓ 通用性强
- ✗ 需要手动管理偏移量

**混合使用示例**：
```cpp
__global__ void mixed_shared(float* data)
{
    // 静态：固定大小的结构
    __shared__ float static_tile[16][16];

    // 动态：可变大小的数据
    extern __shared__ float dynamic_mem[];

    // 计算动态部分的偏移量
    float* dynamic_buffer = &dynamic_mem[16 * 16];

    int tid = threadIdx.x;
    static_tile[tid][0] = data[tid];
    dynamic_buffer[tid] = data[tid] * 2.0f;

    __syncthreads();

    // 使用两种共享内存
    data[tid] = static_tile[tid][0] + dynamic_buffer[tid];
}

// 启动：动态部分大小 = blockDim.x * sizeof(float)
mixed_shared<<<grid, block, 16 * 16 * sizeof(float) + block.x * sizeof(float)>>>(d_data);
```

---

## 五、常见问题与解决方案

### 5.1 共享内存不足

**错误**：
```
cudaErrorLaunchFailure: the launch failed because of out-of-memory
```

**解决方案**：
1. 减少线程块大小
2. 减小共享内存分配
3. 检查共享内存使用量

```cpp
// 检查共享内存使用
__global__ void check_shared_usage()
{
    printf("Static shared: %lu bytes\n", sizeof(static_var));
}

// 查询设备限制
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Max shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
```

### 5.2 静态分析Bank Conflicts

**工具**：NVIDIA Nsight Compute

```bash
# 分析shared memory bank conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./your_program
```

### 5.3 性能验证清单

- [ ] 所有共享内存访问都合并（Coalesced）
- [ ] 没有Bank Conflicts（或最小化）
- [ ] `__syncthreads()` 在所有线程中执行
- [ ] 共享内存使用量不超过限制
- [ ] 数据复用率足够高（>2x）
- [ ] 使用性能分析工具验证

---

## 六、性能提升示例对比

### 实际案例：图像卷积

| 版本 | 全局内存访问 | 共享内存访问 | 执行时间 | 加速比 |
|------|-------------|-------------|---------|--------|
| 朴素版本 | ~3.2 GB | 0 | 125 ms | 1.0x |
| 共享内存版本 | ~0.2 GB | ~3.0 GB | 18 ms | **6.9x** |

**关键因素**：
- 数据复用率：每个像素被卷积核访问 25 次（5×5 filter）
- 共享内存带宽：~33 TB/s vs 全局内存 ~2 TB/s
- **理论加速比**：$25 \times \frac{33}{2} \approx 413\times$（理想情况）

### 不同架构的共享内存配置

| GPU架构 | 共享内存/SM | L1 Cache | 总容量 | 典型配置 |
|---------|------------|----------|--------|----------|
| Fermi/Kepler | 48KB | 16KB | 64KB | 48K shared / 16K L1 |
| Pascal (GP100) | 64KB | 32KB | 96KB | 可配置 |
| Volta (V100) | 96KB | 32KB | 128KB | 可配置 |
| Ampere (A100) | 192KB | 128KB | 320KB | 可配置 |
| Hopper (H100) | 256KB | - | 256KB | L1/Shared统一[²] |

---

## 七、总结

### 核心要点

1. **共享内存是性能倍增器**
   - 比全局内存快约 20-100 倍（延迟）
   - 带宽优势：33 TB/s vs 2 TB/s（H100）[¹]
   - 合理使用可实现 10-100x 加速

2. **关键应用场景**
   - 矩阵乘法、卷积、归约等数据密集型操作
   - 需要频繁访问相同数据的算法
   - 线程间需要通信和协作

3. **最佳实践**
   - 最大化数据复用
   - 避免Bank Conflicts
   - 合理使用同步
   - 使用工具分析和验证

### 学习路径

- **入门**：理解基本概念和声明方式
- **实践**：实现矩阵乘法和归约操作
- **进阶**：优化访问模式，消除Bank Conflicts
- **精通**：结合其他优化技术（如Tensor Core）

### 扩展阅读

- **[CUDA C++ Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)**
- **[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)**
- **[NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)**

---

## 数据来源

本文档中的性能数据和硬件规格基于以下来源：

- **[¹] [Chips and Cheese - Nvidia's H100: Funny L2, and Tons of Bandwidth](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth)** - H100 L1/Shared Memory带宽实测数据
- **[²] [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)** - H100架构共享内存容量规格
- **[³] [NVIDIA H100 Official Specifications](https://www.nvidia.com/en-us/data-center/h100/)** - H100全局内存带宽规格
- **[⁴] [VideoCardz - NVIDIA H100 FP32 Performance Update](https://videocardz.com/newz/nvidia-updates-hopper-h100-data-center-gpu-fp32-performance-from-60-to-67-tflops)** - H100 FP32性能更新
- **[⁵] [NVIDIA Developer Blog - CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)** - CUDA编程模型和Warp大小说明
- **[⁶] [CUDA C++ Programming Guide - Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)** - 官方CUDA编程指南：编程模型和线程限制
- **[⁷] [Stack Overflow - CUDA Multiprocessors, Warp size and Maximum Threads](https://stackoverflow.com/questions/11564608/cuda-multiprocessors-warp-size-and-maximum-threads-per-block-what-is-the-exa)** - Thread block和warp限制讨论
- **[⁸] [NVIDIA CUDA C++ Programming Guide - Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/05-appendices/compute-capabilities.html)** - 官方计算能力规格文档（包含寄存器和block限制）
- **[⁹] [CUDA Forums - Maximum block per grid](https://forums.developer.nvidia.com/t/maximum-block-per-grid/246841)** - Grid维度限制官方讨论
- **[¹⁰] [Massed Compute - What is the maximum size of shared memory in CUDA?](https://massedcompute.com/faq-answers/?question=What%2520is%2520the%2520maximum%2520size%2520of%2520shared%2520memory%2520in%2520CUDA%3F)** - 各架构共享内存容量对比
- **[¹¹] [NVIDIA Ampere Architecture Tuning Guide](https://docs.nvidia.com/cuda/archive/12.9.1/ampere-tuning-guide/index.html)** - A100架构调优指南（CC 8.0规格）
- **[¹²] [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)** - H100架构调优指南（CC 9.0规格，含227KB shared memory）
- **[¹³] [NASA NAS - Basics on NVIDIA GPU Hardware Architecture](https://www.nas.nasa.gov/hecc/support/kb/basics-on-nvidia-gpu-hardware-architecture_704.html)** - GPU硬件架构基础（SM资源限制）
- **[NVIDIA Developer Blog - Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)** - 共享内存使用指南和性能对比


