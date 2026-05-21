# CUTLASS 第一周学习计划

> **机器环境**: H20 × 8 (SM90a, Hopper), CUDA 12.8, CUTLASS 4.4.2
> **本周主题**: CUDA GEMM 原理 + CUTLASS 环境搭建 + 基础示例运行与拆解
> **总耗时预估**: 25~35 小时（每天 5~6 小时）
> **最终目标**: 能独立编译运行 3 个基础示例，理解 CUTLASS 2.x Device API，并搭建起个人性能计量脚本

---

## 总览

| 天　　| 主题　　　　　　　　　　　　　　　　　　　　 | 核心产出　　　　　　　　　　　　　　　　　　　　　 |
| -------| ----------------------------------------------| ----------------------------------------------------|
| Day 1 | CUDA 内存层次 + GEMM 分块理论　　　　　　　　| 能画出 ThreadBlock→Warp→Thread 三级 GEMM 分块图　　|
| Day 2 | 搭建编译环境 + 运行 Example 00　　　　　　　 | build 成功，`basic_gemm` 跑通，能解读输出　　　　　|
| Day 3 | Example 01 工具类 + Example 03 Layout 可视化 | 理解 HostTensor，能可视化 3 种不同 Layout　　　　　|
| Day 4 | 拆解 Example 00 源码 + 手动改造实验　　　　　| 改变数据类型 / 矩阵大小，记录性能变化　　　　　　　|
| Day 5 | 搭建个人 Profile 工具脚本　　　　　　　　　　| 完成 `dive/tools/bench_gemm.sh` 和 `gemm_timer.cu` |

---

## Day 1：CUDA 内存层次 + GEMM 分块理论

### 必读材料（优先级排序）

**① Efficient GEMM in CUDA（最重要，必读）**
[https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)

重点精读以下小节：
- **Hierarchical Decomposition**: 理解 Grid → ThreadBlock → Warp → Thread 四级映射
- **Shared Memory Staging**: 为什么要把 Global Memory 数据先搬到 Shared Memory
- **Double Buffering**: 计算与搬运如何 overlap（预习概念，阶段二细讲）
- **Warp-level GEMM**: Warp Tile 如何分配到各个线程

**② CUDA C++ Programming Guide - Memory Hierarchy（回顾）**
[https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)

重点关注：
- Global Memory latency vs Shared Memory latency（~100 cycles vs ~4 cycles）
- Coalesced Access 条件（连续线程访问连续地址）
- Bank Conflict 成因（Shared Memory 32 banks，同 warp 不同线程访问同一 bank）

**③ CUTLASS Quick Start（环境准备）**
[https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html)

**④ GTC 2018 CUTLASS PPT（选读，建立历史感）**
[http://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf](http://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf)
看 Slides 1~20，了解 CUTLASS 诞生的动机和设计理念。

### Day 1 手绘练习

在纸上画出以下 GEMM 分块示意图，要标出每一级 tile 的尺寸：

```
矩阵 C = A × B，假设 M=4096, N=4096, K=2048

Grid Level:   整个矩阵 C 被切成多少个 ThreadBlock Tile？
              ThreadBlock Tile: 128×128 (M维) × 8 (K 步进)

Block Level:  128×128 的 C 块被 Warp 如何切分？
              Warp Tile: 64×64 (典型)，一个 Block 里有几个 Warp？

Thread Level: 每个 Thread 负责几个元素的累加？
              Thread Tile: 8×8 (典型)
```

---

## Day 2：搭建编译环境 + 运行 Example 00

### 编译环境搭建

```bash
# 确认环境
nvidia-smi                   # 确认 8 卡 H20, SM 9.0
nvcc --version               # 确认 CUDA 12.8

# 进入项目根目录
cd /apdcephfs_zwfy/share_303937731/onealliu/cutlass

# 创建 Release 构建目录（区分已有的 Debug 目录）
mkdir -p build_release && cd build_release

# 配置：仅编译 SM90a（H20 架构），加速编译
cmake .. \
  -DCUTLASS_NVCC_ARCHS=90a \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_ENABLE_EXAMPLES=ON \
  -DCUTLASS_ENABLE_TESTS=OFF \
  2>&1 | tee cmake_config.log

# 检查 cmake 是否识别了 SM90a
grep "CUTLASS_NVCC_ARCHS" cmake_config.log

# 编译第一周所需的 3 个示例（约 5~10 分钟）
make 00_basic_gemm 01_cutlass_utilities 03_visualize_layout -j16 \
  2>&1 | tee build_week01.log
```

> ⚠️ **编译时间提示**: 每个 Example 首次编译约 2~5 分钟，因为 CUTLASS 是全模板，展开量大。耐心等待。

### 运行 Example 00

```bash
cd /apdcephfs_zwfy/share_303937731/onealliu/cutlass/build_release

# 最简运行
CUDA_VISIBLE_DEVICES=0 ./examples/00_basic_gemm/00_basic_gemm

# 带参数运行（修改矩阵大小）
CUDA_VISIBLE_DEVICES=0 ./examples/00_basic_gemm/00_basic_gemm \
  --m=2048 --n=2048 --k=2048

# 运行并记录输出
CUDA_VISIBLE_DEVICES=0 ./examples/00_basic_gemm/00_basic_gemm \
  --m=4096 --n=4096 --k=4096 2>&1 | tee ../dive/logs/day2_basic_gemm.log
```

**预期输出解读**：
```
Problem Size: 4096x4096x4096
Operator:     cutlass_simt_sgemm_128x128x8_nn
Timing:       X.XX ms
GFLOP/s:      X,XXX.X
Passed        ← 数值正确性检查通过
```

### 源码速读清单（Example 00）

阅读 `examples/00_basic_gemm/basic_gemm.cu`，重点理解这几处：

| 行号区间 | 内容 | 要理解的问题 |
|----------|------|-------------|
| ~95~115 | `CutlassGemm` 类型定义 | 模板参数各是什么含义？ |
| ~115~135 | `CutlassGemm::Arguments` 构造 | `{M,N,K}` 是问题大小，`lda/ldb/ldc` 是 leading dimension |
| ~135~140 | `gemm_operator(args)` 调用 | CUTLASS 的 functor 调用模式 |
| ~160~210 | `ReferenceGemm_kernel` | 朴素 CPU/GPU 参考实现（用于验证） |
| ~300~400 | `main` 函数 | 计时、结果比较流程 |

---

## Day 3：CUTLASS 工具类（Example 01）+ Layout 可视化（Example 03）

### 运行 Example 01：工具类

```bash
cd /apdcephfs_zwfy/share_303937731/onealliu/cutlass/build_release

CUDA_VISIBLE_DEVICES=0 ./examples/01_cutlass_utilities/01_cutlass_utilities
```

阅读源码 `examples/01_cutlass_utilities/cutlass_utilities.cu`，理解：

| API                                        | 作用　　　　　　　　　　　　　　　|
| --------------------------------------------| -----------------------------------|
| `cutlass::HostTensor<T, Layout>`           | CPU+GPU 双端 Tensor，自动管理内存 |
| `.sync_device()`                           | CPU → GPU 数据同步　　　　　　　　|
| `.sync_host()`                             | GPU → CPU 数据同步　　　　　　　　|
| `cutlass::reference::host::TensorFill()`   | 用随机数填充　　　　　　　　　　　|
| `cutlass::reference::host::TensorEquals()` | 数值比较　　　　　　　　　　　　　|

> **HostTensor 是第一周最重要的工具类**，后续所有示例都用它管理数据。

### 运行 Example 03：Layout 可视化

```bash
# 可视化 ColumnMajor 布局
CUDA_VISIBLE_DEVICES=0 ./examples/03_visualize_layout/03_visualize_layout \
  --layout=ColumnMajor --m=4 --n=4

# 可视化 RowMajor 布局
CUDA_VISIBLE_DEVICES=0 ./examples/03_visualize_layout/03_visualize_layout \
  --layout=RowMajor --m=4 --n=4

# 可视化 TensorNHWC 布局
CUDA_VISIBLE_DEVICES=0 ./examples/03_visualize_layout/03_visualize_layout \
  --layout=TensorNHWC --n=2 --h=2 --w=2 --c=4
```

**动手练习**：运行后，用手画出每种 Layout 下的内存地址排列图，理解：
- `ColumnMajor` 中 stride = (1, M)，意味着同列相邻元素地址连续
- `RowMajor` 中 stride = (N, 1)，意味着同行相邻元素地址连续
- Tensor Core 数据布局要求（为什么要 ColumnMajor A × RowMajor B）

### 必读文档补充

**CUTLASS Terminology（术语速查）**
[https://docs.nvidia.com/cutlass/latest/media/docs/cpp/terminology.html](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/terminology.html)
重点：`ThreadBlockShape`, `WarpCount`, `InstructionShape`, `kAlignmentA/B` 的含义

**CUTLASS Layout 文档**
[https://docs.nvidia.com/cutlass/latest/media/docs/cpp/layout.html](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/layout.html)

---

## Day 4：深度拆解 Example 00 + 改造实验

### 实验 1：修改矩阵大小，观察性能曲线

创建脚本 `dive/logs/day4_size_sweep.sh`：

```bash
#!/bin/bash
# 不同矩阵大小对性能的影响
EXE=/apdcephfs_zwfy/share_303937731/onealliu/cutlass/build_release/examples/00_basic_gemm/00_basic_gemm

echo "M,N,K,GFLOPS"
for SIZE in 512 1024 2048 4096 8192; do
  RESULT=$(CUDA_VISIBLE_DEVICES=0 $EXE --m=$SIZE --n=$SIZE --k=$SIZE 2>&1 | grep "GFLOP/s")
  GFLOPS=$(echo $RESULT | awk '{print $NF}')
  echo "$SIZE,$SIZE,$SIZE,$GFLOPS"
done
```

```bash
mkdir -p /apdcephfs_zwfy/share_303937731/onealliu/cutlass/dive/logs
bash /apdcephfs_zwfy/share_303937731/onealliu/cutlass/dive/logs/day4_size_sweep.sh \
  | tee /apdcephfs_zwfy/share_303937731/onealliu/cutlass/dive/logs/day4_size_sweep.csv
```

**预期现象**：
- 小矩阵（512）：GFLOPS 很低，GPU 未打满（Launch overhead 为主）
- 中等矩阵（2048~4096）：GFLOPS 开始提升
- 注意：这是 SIMT SGEMM，**不是** Tensor Core，H20 上 FP32 SIMT 峰值很低

### 实验 2：理解 CUTLASS 2.x 模板参数

在 `build_release` 目录下新建文件 `dive/exp/exp_gemm_template.cu`：

```cpp
// 实验：用不同的 Tile 大小实例化 CUTLASS GEMM
// 目的：理解 ThreadBlockShape 对性能的影响

#include "cutlass/gemm/device/gemm.h"
#include <stdio.h>

// 方案A: 128x128x8 (default)
using GemmA = cutlass::gemm::device::Gemm<
  float, cutlass::layout::ColumnMajor,   // A
  float, cutlass::layout::ColumnMajor,   // B
  float, cutlass::layout::ColumnMajor,   // C
  float,                                  // Accumulator
  cutlass::arch::OpClassSimt,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128, 128, 8>,  // ThreadBlock Tile
  cutlass::gemm::GemmShape<32, 64, 8>,   // Warp Tile
  cutlass::gemm::GemmShape<1, 1, 1>      // Instruction Shape (SIMT = 1x1x1)
>;

// 方案B: 64x64x8 (更小的 Tile)
using GemmB = cutlass::gemm::device::Gemm<
  float, cutlass::layout::ColumnMajor,
  float, cutlass::layout::ColumnMajor,
  float, cutlass::layout::ColumnMajor,
  float,
  cutlass::arch::OpClassSimt,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<64, 64, 8>,
  cutlass::gemm::GemmShape<32, 32, 8>,
  cutlass::gemm::GemmShape<1, 1, 1>
>;

int main() {
  // 打印 Tile 信息
  printf("GemmA ThreadBlock Tile: %dx%dx%d\n",
    GemmA::ThreadblockShape::kM,
    GemmA::ThreadblockShape::kN,
    GemmA::ThreadblockShape::kK);
  printf("GemmA Warp Tile: %dx%dx%d\n",
    GemmA::WarpShape::kM,
    GemmA::WarpShape::kN,
    GemmA::WarpShape::kK);
  printf("GemmA Warp Count: %d warps per block\n",
    GemmA::ThreadblockShape::kM / GemmA::WarpShape::kM *
    GemmA::ThreadblockShape::kN / GemmA::WarpShape::kN);
  return 0;
}
```

编译方法（在 `build_release` 目录）：
```bash
cd /apdcephfs_zwfy/share_303937731/onealliu/cutlass/build_release
nvcc -std=c++17 \
  -I/apdcephfs_zwfy/share_303937731/onealliu/cutlass/include \
  -I/apdcephfs_zwfy/share_303937731/onealliu/cutlass/tools/util/include \
  -arch=sm_90a \
  /apdcephfs_zwfy/share_303937731/onealliu/cutlass/dive/exp/exp_gemm_template.cu \
  -o /apdcephfs_zwfy/share_303937731/onealliu/cutlass/dive/exp/exp_gemm_template \
  && /apdcephfs_zwfy/share_303937731/onealliu/cutlass/dive/exp/exp_gemm_template
```

### Day 4 思考题（写在 `dive/notes/day4_notes.md`）

1. `basic_gemm.cu` 中 `CutlassGemm` 的默认 ThreadBlock Tile 是多少？在哪里定义的？
   - 提示：查 `include/cutlass/gemm/device/default_gemm_configuration.h`
2. `leading dimension` (lda/ldb/ldc) 和矩阵大小的关系是什么？ColumnMajor 时 lda = ?
3. H20 的 FP32 SIMT 理论峰值是多少 TFLOPS？这个 Example 达到了多少？为什么差距这么大？

---

## Day 5：搭建个人 Profile 工具集

> **目标**：建立一套可复用的性能测量基础设施，后续所有实验都基于此

### 工具 1：`bench_gemm.sh` — 多规格性能扫描脚本

创建 `dive/tools/bench_gemm.sh`：

```bash
#!/bin/bash
# CUTLASS GEMM Benchmark Script
# 用法: bash bench_gemm.sh <executable> [gpu_id]

EXE=${1:-/apdcephfs_zwfy/share_303937731/onealliu/cutlass/build_release/examples/00_basic_gemm/00_basic_gemm}
GPU_ID=${2:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="/apdcephfs_zwfy/share_303937731/onealliu/cutlass/dive/logs/bench_${TIMESTAMP}.csv"

echo "# CUTLASS GEMM Benchmark"
echo "# GPU: $(CUDA_VISIBLE_DEVICES=$GPU_ID nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "# Date: $(date)"
echo "# Executable: $EXE"
echo ""
echo "M,N,K,Layout,GFLOPS,Time_ms"

# 方形矩阵（典型训练场景）
for SIZE in 1024 2048 4096 8192; do
  OUT=$(CUDA_VISIBLE_DEVICES=$GPU_ID $EXE --m=$SIZE --n=$SIZE --k=$SIZE 2>&1)
  GFLOPS=$(echo "$OUT" | grep "GFLOP/s" | awk '{print $NF}')
  TIME_MS=$(echo "$OUT" | grep "Timing" | awk '{print $2}')
  echo "$SIZE,$SIZE,$SIZE,NN,$GFLOPS,$TIME_MS"
done

# 推理典型形状（M小，NK大 — decode场景）
for M in 1 4 8 32 128; do
  N=4096; K=4096
  OUT=$(CUDA_VISIBLE_DEVICES=$GPU_ID $EXE --m=$M --n=$N --k=$K 2>&1)
  GFLOPS=$(echo "$OUT" | grep "GFLOP/s" | awk '{print $NF}')
  TIME_MS=$(echo "$OUT" | grep "Timing" | awk '{print $2}')
  echo "$M,$N,$K,NN(decode),$GFLOPS,$TIME_MS"
done
```

```bash
mkdir -p /apdcephfs_zwfy/share_303937731/onealliu/cutlass/dive/tools
mkdir -p /apdcephfs_zwfy/share_303937731/onealliu/cutlass/dive/logs
mkdir -p /apdcephfs_zwfy/share_303937731/onealliu/cutlass/dive/notes
mkdir -p /apdcephfs_zwfy/share_303937731/onealliu/cutlass/dive/exp

chmod +x /apdcephfs_zwfy/share_303937731/onealliu/cutlass/dive/tools/bench_gemm.sh
bash /apdcephfs_zwfy/share_303937731/onealliu/cutlass/dive/tools/bench_gemm.sh \
  | tee /apdcephfs_zwfy/share_303937731/onealliu/cutlass/dive/logs/week01_baseline.csv
```

### 工具 2：`gemm_timer.cu` — 带 CUDA Event 精确计时的 GEMM 模板

创建 `dive/tools/gemm_timer.cu`：

```cpp
/**
 * gemm_timer.cu
 *
 * 用 CUDA Event 精确计时的 CUTLASS GEMM wrapper
 * 包含: warmup runs + 多次取平均 + 理论FLOPS计算
 *
 * 编译:
 *   nvcc -std=c++17 \
 *     -I/apdcephfs_zwfy/share_303937731/onealliu/cutlass/include \
 *     -I/apdcephfs_zwfy/share_303937731/onealliu/cutlass/tools/util/include \
 *     -arch=sm_90a -O3 \
 *     gemm_timer.cu -o gemm_timer
 */

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

// ============================================================
// 辅助宏
// ============================================================
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                << " - " << cudaGetErrorString(err) << std::endl; \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

// ============================================================
// CUTLASS SGEMM (SIMT, SM80+ compatible)
// ============================================================
using Gemm = cutlass::gemm::device::Gemm<
  float, cutlass::layout::ColumnMajor,
  float, cutlass::layout::ColumnMajor,
  float, cutlass::layout::ColumnMajor
>;

// ============================================================
// 精确计时函数：warmup + 多次测量
// ============================================================
double benchmark_gemm(int M, int N, int K, int warmup = 3, int repeats = 10) {
  // 分配并初始化矩阵
  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> A({M, K});
  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> B({K, N});
  cutlass::HostTensor<float, cutlass::layout::ColumnMajor> C({M, N});

  cutlass::reference::host::TensorFillRandomUniform(A.host_view(), 42, 1.0f, -1.0f);
  cutlass::reference::host::TensorFillRandomUniform(B.host_view(), 43, 1.0f, -1.0f);
  cutlass::reference::host::TensorFill(C.host_view(), 0.0f);

  A.sync_device();
  B.sync_device();
  C.sync_device();

  Gemm gemm_op;
  Gemm::Arguments args(
    {M, N, K},
    {A.device_data(), A.stride(0)},
    {B.device_data(), B.stride(0)},
    {C.device_data(), C.stride(0)},
    {C.device_data(), C.stride(0)},
    {1.0f, 0.0f}
  );

  // 检查参数合法性
  cutlass::Status status = gemm_op.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "GEMM cannot implement: " << (int)status << std::endl;
    return -1.0;
  }

  // Warmup
  for (int i = 0; i < warmup; ++i) {
    gemm_op(args);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // 精确计时
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeats; ++i) {
    gemm_op(args);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return static_cast<double>(elapsed_ms) / repeats;  // 平均每次 ms
}

int main(int argc, char* argv[]) {
  int M = (argc > 1) ? atoi(argv[1]) : 4096;
  int N = (argc > 2) ? atoi(argv[2]) : 4096;
  int K = (argc > 3) ? atoi(argv[3]) : 4096;

  std::cout << "=" << std::string(60, '=') << "\n";
  std::cout << "  CUTLASS GEMM Benchmark (SIMT SGEMM)\n";
  std::cout << "  Problem: M=" << M << " N=" << N << " K=" << K << "\n";
  std::cout << "=" << std::string(60, '=') << "\n";

  double avg_ms = benchmark_gemm(M, N, K);

  // 计算 GFLOPS：GEMM FLOPs = 2 * M * N * K
  double flops = 2.0 * M * N * K;
  double gflops = flops / (avg_ms * 1e-3) / 1e9;

  // 计算内存带宽使用（估算）
  double bytes = (M * K + K * N + M * N) * sizeof(float);
  double gbps = bytes / (avg_ms * 1e-3) / 1e9;

  // H20 参考值：FP32 SIMT 理论峰值约 ~67 TFLOPS (tensor core) 但 SIMT 远低
  // H20 HBM3 带宽: 4096 GB/s

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  Avg Time:        " << avg_ms << " ms\n";
  std::cout << "  GFLOP/s:         " << gflops << " GFLOPS\n";
  std::cout << "  Arithmetic Intensity: " << (flops / bytes) << " FLOP/byte\n";
  std::cout << "  Est. BW used:    " << gbps << " GB/s\n";
  std::cout << "  (H20 HBM3 peak:  4096 GB/s)\n";
  std::cout << "=" << std::string(60, '=') << "\n";

  return 0;
}
```

编译和运行：

```bash
cd /apdcephfs_zwfy/share_303937731/onealliu/cutlass/dive/tools

CUDA_HOME=/usr/local/cuda
CUTLASS_ROOT=/apdcephfs_zwfy/share_303937731/onealliu/cutlass

${CUDA_HOME}/bin/nvcc -std=c++17 \
  -I${CUTLASS_ROOT}/include \
  -I${CUTLASS_ROOT}/tools/util/include \
  -arch=sm_90a -O3 \
  gemm_timer.cu -o gemm_timer

# 运行
CUDA_VISIBLE_DEVICES=0 ./gemm_timer 4096 4096 4096
CUDA_VISIBLE_DEVICES=0 ./gemm_timer 1024 1024 1024
CUDA_VISIBLE_DEVICES=0 ./gemm_timer 128 4096 4096   # decode 场景
```

### 工具 3：快速 Nsight Compute 性能分析命令

```bash
# 用 ncu 采集单次 Kernel 的关键指标
CUDA_VISIBLE_DEVICES=0 ncu \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
sm__cycles_elapsed.avg \
  --target-processes all \
  /apdcephfs_zwfy/share_303937731/onealliu/cutlass/build_release/examples/00_basic_gemm/00_basic_gemm \
  --m=4096 --n=4096 --k=4096 \
  2>&1 | tee /apdcephfs_zwfy/share_303937731/onealliu/cutlass/dive/logs/day5_ncu_basic.log

# 查看 SM 利用率
grep "sm__throughput" /apdcephfs_zwfy/share_303937731/onealliu/cutlass/dive/logs/day5_ncu_basic.log
```

> **注意**: `ncu` 采集会让 Kernel 极慢（10~100x），仅用于分析，不用于计时。

---

## 每日 Checklist

### Day 1
- [x] 阅读完 Efficient GEMM in CUDA 全文
- [x] 手绘 GEMM 三级分块图
- [x] 理解为什么 Shared Memory 是性能关键

### Day 2
- [ ] `build_release` 目录创建并 cmake 成功
- [ ] `00_basic_gemm` 编译并运行通过 `Passed`
- [ ] 能解释输出中 `GFLOP/s` 是怎么算出来的

### Day 3
- [ ] `01_cutlass_utilities` 运行通过
- [ ] `03_visualize_layout` 对 ColumnMajor/RowMajor 输出分别运行一次
- [ ] 手画 4×4 ColumnMajor 矩阵的内存地址排列图

### Day 4
- [ ] 完成 `day4_size_sweep.sh` 并记录数据
- [ ] 回答 Day4 思考题（写入 `dive/notes/day4_notes.md`）
- [ ] 阅读 `default_gemm_configuration.h` 了解默认 Tile 大小

### Day 5
- [ ] `bench_gemm.sh` 可运行并输出 CSV
- [ ] `gemm_timer.cu` 编译并能输出 GFLOPS + 算术强度
- [ ] 记录 H20 上 SIMT SGEMM 的实测性能，与理论峰值对比

---

## 第一周预期观察与思考

完成第一周后，你应该能观察到：

1. **SIMT vs Tensor Core 性能差距**: `00_basic_gemm` 用的是 SIMT SGEMM，在 H20 上只能达到很低的 TFLOPS，因为没有用 Tensor Core。这将是 Week 3~4 的核心主题。

2. **矩阵大小与 GPU 利用率**: 小矩阵（M=512）时 GPU 利用率极低，因为不够多的 CTA 来填满 H20 的 132 个 SM（H20 有约 132 SM）。`M=4096` 时效率会明显更高。

3. **算术强度与性能瓶颈**: 通过 `gemm_timer.cu` 的输出，判断你的 GEMM 是 compute-bound 还是 memory-bound。对于大矩阵 SGEMM，算术强度约为 `2*M*N*K / ((M*K + K*N + M*N)*4)`，应该是 compute-bound。

4. **编译时间极长**: 这是 CUTLASS 的固有特点，后续通过明确指定 `CUTLASS_NVCC_ARCHS` 和 `-DCUTLASS_ENABLE_TESTS=OFF` 来控制。

---

## 参考文件位置速查

| 文件 | 说明 |
|------|------|
| `examples/00_basic_gemm/basic_gemm.cu` | 本周主读源码 |
| `examples/01_cutlass_utilities/cutlass_utilities.cu` | HostTensor 工具使用 |
| `examples/03_visualize_layout/visualize_layout.cpp` | Layout 可视化主程序 |
| `include/cutlass/gemm/device/gemm.h` | Device GEMM 接口声明 |
| `include/cutlass/gemm/device/default_gemm_configuration.h` | 默认 Tile 大小配置 |
| `include/cutlass/layout/matrix.h` | ColumnMajor/RowMajor 定义 |
| `tools/util/include/cutlass/util/host_tensor.h` | HostTensor 实现 |

---

## 第二周预告

第二周（Week 2）将进入 **CUTLASS 类型系统与 Layout 深度**：
- 学习 `half_t`, `bfloat16_t`, `float_e4m3_t` 等数值类型（FP16/BF16/FP8 精度对比）
- 理解 Tensor Core GEMM（从 SIMT 升级到 `OpClassTensorOp`，在 H20 上利用 Tensor Core）
- 运行 `07_volta_tensorop_gemm` 和 `14_ampere_tf32_tensorop_gemm`
- 开始接触 CuTe Layout 代数预备知识
