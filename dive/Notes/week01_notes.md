
[toc]
#  一、为什么必须把 Global Memory 搬到 Shared Memory？

  核心原因是访存延迟 + 带宽复用，需要从两个角度理解：

  1.1 延迟角度：Global Memory 太慢，必须隐藏延迟

  H20 各级存储延迟（近似值）：

  Global Memory (HBM3)  ~400~600 cycles   ← 慢！
  L2 Cache              ~200 cycles
  L1 Cache / Shared Mem ~20~32 cycles     ← 快 20x
  Register              ~1 cycle          ← 最快

  GEMM 的每个线程需要反复读 A 矩阵的同一行、B 矩阵的同一列。如果每次都从 Global Memory 读：

  Thread 0: 读 A[0,0], A[0,1], ..., A[0,K-1]  → K 次 Global 访问
  Thread 1: 读 A[1,0], A[1,1], ..., A[1,K-1]  → K 次 Global 访问
  ...
  Thread M-1: 读 A[M-1,0], ..., A[M-1,K-1]   → K 次 Global 访问

  问题：同一个 ThreadBlock 里的不同线程在读 A 的同一列（相同的 A[*,k]），这是重复读取！

  1.2 带宽复用角度：数据重用比（Reuse Factor）

  以 128×128×8 的 ThreadBlock Tile 为例，分析一次 K 方向步进（k_step = 8）：

  需要读入 Shared Memory 的数据量：
    A tile: 128 × 8  × 4 bytes = 4 KB
    B tile: 8   × 128 × 4 bytes = 4 KB
    共 8 KB

  这 8 KB 数据，被复用了多少次？
    计算量: 128 × 128 × 8 × 2 = 262,144 FLOPs
    每字节计算量: 262,144 / (8×1024) = 32 FLOP/byte

  如果不用 Shared Memory，直接从 Global 读：
    每个 Thread 读 A 的 8 个元素（1次），需要被 128 列的 B 线程全读
    → 同一列 A[row, k] 被 128 个线程各读一遍
    → 等效每个数据读 128 次

    实际上需要的 Global Memory 带宽：
    8 KB × 128 = 1 MB（逻辑上，无复用）

  关键公式：

  有了 Shared Memory 后的带宽需求 = 原始数据量 / 数据重用因子

  ThreadBlock Tile (M_t, N_t, K_t) 的重用因子：
    A 的重用因子 = N_t（N方向的每个输出列都要用同一块A）
    B 的重用因子 = M_t（M方向的每个输出行都要用同一块B）

  以 128×128×8 举例：
    不用 Shared Memory：需要 Global BW = (128×8 × 128 + 8×128 × 128) × 4B = 1MB per tile step
    用了 Shared Memory： 需要 Global BW = (128×8 + 8×128) × 4B = 8KB per tile step
    节省带宽：128x ！！

  1.3 用图来理解（K 方向分块迭代）

  C[m, n] = Σ_k  A[m, k] × B[k, n]

  Global Memory:        Shared Memory:
  ┌─────────────────┐   ┌──────────┐
  │  A (M×K)        │   │  A_tile  │  128×8
  │                 │──→│  (4KB)   │──→  Register → 计算 C tile
  │  B (K×N)        │   │  B_tile  │
  │                 │──→│  (4KB)   │
  └─────────────────┘   └──────────┘

  每次从 Global 搬 8KB 到 Shared，
  Shared 里的数据被 128×128 = 16384 个线程复用，
  每个线程从 Shared 读 8+8=16 个元素（快速），
  而非从 Global 读 16 个元素（慢速）。

  1.4 另一个角度：L2 Cache 为什么不够？

  你可能问：现代 GPU 有 L2 Cache，为什么还需要手动管 Shared Memory？

  H20 L2 Cache: ~50 MB（全芯片共享）
  H20 Shared Memory: 最大 228 KB / SM，SM数量 × 这个容量

  问题1: 132 个 SM 同时运行，L2 是竞争资源，Cache 命中率无法保证
  问题2: L2 是 set-associative，无法精确控制哪些数据留在 cache 里
  问题3: Shared Memory 延迟 < L2 延迟（SM 片上 vs 芯片级互联）
  问题4: Shared Memory 可以做 double buffering（编程精确控制），L2 不行

  Shared Memory 本质是程序员控制的 scratchpad（便签内存），而不是 cache（透明缓存）。

  ---
# 二、Threadblock Rasterization（光栅化顺序）详解

  这是 CUTLASS 中一个非常有意思但文档里鲜少深讲的优化。

  2.1 什么是 Rasterization？

  GPU 启动一个 GEMM Kernel 时，需要把输出矩阵 C 切成很多 ThreadBlock Tile，然后决定以什么顺序把这些 Tile 分配给 SM 执行。这个顺序就是 Rasterization（借用图形学的光栅化概念）。

  2.2 朴素顺序（Row-major Rasterization）

  最简单的做法是按行优先顺序分配 Tile：

  C 矩阵被切成 Grid_M × Grid_N 个 Tile，Grid_M = M/128, Grid_N = N/128

  朴素顺序（blockIdx.x = tile_n, blockIdx.y = tile_m）：

  Tile 分配顺序：
  (0,0) → (0,1) → (0,2) → (0,3) → ...  ← 第0行tile全部分配完
  (1,0) → (1,1) → (1,2) → (1,3) → ...  ← 第1行tile
  ...

  在 GPU 上实际执行时（假设SM数量 = 4）：
  SM0: (0,0), (0,4), (0,8), ...
  SM1: (0,1), (0,5), (0,9), ...
  SM2: (0,2), (0,6), ...
  SM3: (0,3), (0,7), ...

  问题在哪？ → L2 Cache 浪费！

  SM0 处理 tile (0,0): 读入 A[0:128, 0:K] + B[0:K, 0:128]
  SM1 处理 tile (0,1): 读入 A[0:128, 0:K] + B[0:K, 128:256]
                        ↑ 这部分 A 和 SM0 完全一样！如果 L2 还有 → cache hit
  SM2 处理 tile (0,2): 读入 A[0:128, 0:K] + B[0:K, 256:384]
                        ↑ A 还是一样的...
  SM3 处理 tile (0,3): 读入 A[0:128, 0:K] + B[0:K, 384:512]

  -- 执行完第0行后 --

  SM0 处理 tile (1,0): 读入 A[128:256, 0:K] + B[0:K, 0:128]
                        ↑ A 变了，但 B 的这块和之前 SM0 处理(0,0)时一样！
                        但是 L2 里 B 的这块已经被后来的 B[0:K, 384:512] 驱逐了

  在大矩阵情况下，L2 Cache 容量不够装下整行/整列的 A 或 B，导致 Cache 命中率很低。

  2.3 Swizzled Rasterization（CUTLASS 的解法）

  CUTLASS 实现了一种 Swizzled（交错）的 Tile 分配顺序，核心思想是：让同时执行的 Tile 共享更多 A 或 B 的数据，提升 L2 命中率。

  // CUTLASS 实现位置：
  // include/cutlass/gemm/kernel/gemm_universal.hpp
  // include/cutlass/gemm/threadblock/threadblock_swizzle.h

  // 典型的 Swizzle 策略
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  // 或
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle;

  Swizzled 顺序的核心思想：以 tile_group_size（如 8）为单位，先在 M 方向分配多个 tile，再移动 N 方向：

  Swizzle = 8 的情况（8列 tile 为一组）：

  朴素顺序：               Swizzled 顺序：
  (0,0)(0,1)(0,2)(0,3)    (0,0)(1,0)(2,0)(3,0)
  (1,0)(1,1)(1,2)(1,3)    (4,0)(5,0)(6,0)(7,0)  ← 先把M方向8个tile做完
  (2,0)(2,1)(2,2)(2,3)    (0,1)(1,1)(2,1)(3,1)
  ...                      (4,1)(5,1)(6,1)(7,1)  ← 再移到下一列

  为什么 Swizzled 更好？

  SM 并发数量 = 132（H20），假设每 SM 同时运行 1 个 Tile

  Swizzled 情况下：
    同时执行的 tile (0,0), (1,0), ..., (7,0) 共 8 个 tile：
      A: A[0:128,0:K], A[128:256,0:K], ..., A[896:1024,0:K]  ← 不同行，L2不帮忙
      B: B[0:K, 0:128]  ← 全部使用同一块 B ！！ → L2命中率极高

    而朴素顺序下同时执行的 (0,0), (0,1), ..., (0,7)：
      A: A[0:128, 0:K]  ← 全部使用同一块 A，但K维很大，A很大
      B: B[0:K,0:128], B[0:K,128:256], ..., B[0:K,896:1024]  ← 8块不同B

    哪种更好取决于矩阵形状！
    - M >> N：Swizzle 更好（B复用更容易进L2）
    - N >> M：朴素更好（A复用更容易进L2）

  2.4 Rasterization 对 L2 命中率的量化影响

  CUTLASS 文档中的 Efficient GEMM 图表展示了一个关键数据：

  以 M=N=4096, K=4096, Tile=128×128 为例：
    Grid 大小 = (4096/128) × (4096/128) = 32 × 32 = 1024 个 Tile
    H20 有 132 个 SM
    需要约 1024/132 ≈ 8 个 Wave 执行完

  朴素顺序（行优先）时，每 Wave 读入的 B 数据：
    Wave 0: SM 0~131 → tile (0,0)~(0,131 % 32) + (1~4, ...) → B 的 L2 复用率 ~30%

  Swizzled 顺序（GemmHorizontal）时：
    Wave 0 中的 SM 倾向于执行同一列的 tile，B 的这列数据在 L2 中持续命中
    B 的 L2 复用率可达 ~70~80%

  实测影响：在 M=N=4096, K=4096 时，Swizzle 带来约 5~15% 的性能提升（取决于 L2 大小与矩阵大小的比）

  2.5 CUTLASS 中 Swizzle 的使用方式

  // Example 47 中有对比不同 Swizzle 策略的代码
  // include/cutlass/gemm/threadblock/threadblock_swizzle.h

  // 常见 Swizzle 类型：
  using Swizzle1 = GemmIdentityThreadblockSwizzle<1>;  // 朴素，无交错
  using Swizzle4 = GemmIdentityThreadblockSwizzle<4>;  // 4列一组
  using Swizzle8 = GemmIdentityThreadblockSwizzle<8>;  // 8列一组（最常用）
  using SwizzleH = GemmHorizontalThreadblockSwizzle;   // 水平优先

  // CUTLASS 3.x 中（Hopper）
  using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

  最优 Swizzle 值的选择规则：

  tile_group_size 应该约等于：sqrt(L2_size / tile_size)

  H20 L2 = 50 MB，tile B = 128×128×2B(FP16) = 32 KB
  √(50MB / 32KB) ≈ √1600 ≈ 40

  但实际受 SM 数量约束：tile_group_size ≤ Grid_M
  通常 8 是一个工程上的好选择，但需要根据实际 profiling 调整

  ---
  总结

  ┌────────────────────────┬───────────────────────────────────────────────────────────────────────────────┐
  │          概念          │                                   核心要点                                    │
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ Global→Shared 搬运     │ 数据重用：同 ThreadBlock 内线程共享 Shared，节省 128x Global BW               │
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ Shared Memory 本质     │ 程序员控制的 scratchpad，不是 cache，延迟更低，可精确控制                     │
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ Rasterization 朴素顺序 │ 行优先，L2 复用率低，因为相邻 tile 共享 A 或 B 的概率低                       │
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ Swizzled Rasterization │ 按列/块方向优先分配，同时执行的 tile 共享同一块 B（或 A），大幅提升 L2 命中率 │
  ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ 最优 Swizzle           │ tile_group_size ≈ √(L2_size / B_tile_size)，经验值通常为 8                    │
  └────────────────────────┴───────────────────────────────────────────────────────────────────────────────┘

  第一周 Day 5 的 bench_gemm.sh 里可以后续加入对比不同 Swizzle 策略的实验，等你到 Week 4 运行 examples/47_ampere_gemm_universal_streamk/ 时这个效果会非常明显。

---

# 三、CUTLASS 诞生的动机与设计理念（GTC 2018 核心思想重建）

> 来源：`media/docs/cpp/efficient_gemm.md`、`cutlass_3x_design.md`、`programming_guidelines.md`、`code_organization.md`，结合 GTC 2018 演讲核心论点

## 3.1 诞生动机：cuBLAS 之痛

### 背景：2017年的痛点

在 CUTLASS 诞生之前（2017年），GPU 上的 GEMM 计算有两条路：

**路线 A：直接用 cuBLAS**
```
优点：性能极好（NVIDIA 内部精心调优）
缺点：
  ✗ 完全黑盒，无法修改内部行为
  ✗ 无法融合自定义操作（如 GEMM + 自定义 Activation + LayerNorm）
  ✗ 接口固定，不支持新数据类型（当时 INT8/FP16 混合精度兴起）
  ✗ 每次新架构（Volta/Turing/Ampere）发布，用户必须等 NVIDIA 更新
```

**路线 B：自己从零写 CUDA GEMM**
```
优点：完全可控
缺点：
  ✗ 极其困难：手写一个接近 cuBLAS 性能的 GEMM 需要数月工作
  ✗ 不可复用：换个数据类型、Tile 大小、架构全部要重写
  ✗ 没有统一抽象：bank conflict、coalescing、Tensor Core 布局要求
    每次都要从头研究
  ✗ 维护代价极高：每新增一个功能要改遍所有地方
```

**现实需求的爆发（2016~2017）：**
```
深度学习训练：需要 GEMM + Batch Norm + ReLU 融合（减少 Global Memory 读写）
INT8 推理：需要 INT8 Tensor Core + FP32 Accumulator + Dequantize 融合
稀疏计算：需要 Sparse GEMM + 自定义稀疏格式
多头注意力：需要 Batched GEMM + Softmax + Dropout 融合

→ 以上需求 cuBLAS 全部无法满足，从零写又太难
→ CUTLASS 因此而生
```

### 核心矛盾

CUTLASS 要解决的根本矛盾：

```
cuBLAS 的性能  ←→  用户代码的灵活性
     ↑                    ↑
  内部优化          可组合、可扩展
  架构专用          跨架构统一接口
  黑盒不透明        可读、可学习
```

**CUTLASS 的答案**：用 C++ 模板元编程，把 GEMM 的所有"移动部件"分解为可复用的模块，像乐高积木一样组合，且每个模块都可以被替换为自定义实现，同时编译器保证零额外开销。

---

## 3.2 核心设计理念一：分层分解（Hierarchical Decomposition）

这是 GTC 2018 演讲的核心贡献，也是理解 CUTLASS 一切设计的基础。

### 理念来源：模仿 cuBLAS 内部结构

CUTLASS 官方文档原文（`doxygen_mainpage.md`）：
> "It incorporates strategies for hierarchical decomposition and data movement **similar to those used to implement cuBLAS**."

cuBLAS 内部（非公开）本质上也是分层的，CUTLASS 把这个分层结构**显式化、公开化、可编程化**。

### 分层与硬件的精确对应

每一层抽象，都对应 GPU 硬件执行模型的一个层次：

```
软件抽象层              硬件执行层              内存层次
─────────────────────────────────────────────────────────
Device Layer         整个 GPU (Grid)         DDR/HBM (Global)
    ↓                      ↓                      ↓
Threadblock Layer    SM (Thread Block)       SRAM (Shared Memory)
    ↓                      ↓                      ↓
Warp Layer           Warp (32 threads)       RF (Registers)
    ↓                      ↓                      ↓
Thread Layer         CUDA Core (1 thread)    RF (Registers)
    ↓                      ↓                      ↓
Instruction Layer    MMA 硬件指令             Tensor Core
```

这个对应不是偶然的，而是刻意设计的：
- 每个层次的数据搬运 **对应且仅对应** 一种内存介质之间的传输
- 每个层次的计算 **对应且仅对应** 一种硬件并行单元
- 层次之间的接口是**正交的**（改变一层不影响其他层）

### 层次化分块的 Loop Nest 视角

从循环嵌套的角度看（来自 `efficient_gemm.md` 原文）：

```cpp
// Layer 1: Device → 对应 CUDA Grid Launch，隐式
for (int cta_n = 0; cta_n < GemmN; cta_n += CtaTileN) {    // threadblock 并发
  for (int cta_m = 0; cta_m < GemmM; cta_m += CtaTileM) {

    // Layer 2: ThreadBlock → Shared Memory staging
    for (int cta_k = 0; cta_k < GemmK; cta_k += CtaTileK) {  // GEMM mainloop

      // Layer 3: Warp → Register fragments
      for (int warp_n = 0; warp_n < CtaTileN; warp_n += WarpTileN) {
        for (int warp_m = 0; warp_m < CtaTileM; warp_m += WarpTileM) {
          for (int warp_k = 0; warp_k < CtaTileK; warp_k += WarpTileK) {

            // Layer 4: Instruction → Tensor Core MMA
            for (int mma_k = 0; mma_k < WarpTileK; mma_k += MmaK) {
              for (int mma_n = 0; mma_n < WarpTileN; mma_n += MmaN) {
                for (int mma_m = 0; mma_m < WarpTileM; mma_m += MmaM) {
                  mma_instruction(d, a, b, c);   // ← 硬件指令
                }
              }
            }
          }
        }
      }
    }
  }
}
```

**关键洞察**：
- 最外两层循环（cta_m, cta_n）由 CUDA Grid 隐式并行执行
- mainloop（cta_k 循环）是每个 ThreadBlock 的主工作循环，**不展开**
- warp 层和 instruction 层的循环**全部展开**（`CUTLASS_PRAGMA_UNROLL`），让编译器做寄存器分配和指令调度

---

## 3.3 核心设计理念二：模板元编程驱动的零开销抽象

### 为什么用模板，而不是虚函数？

GPU 内核代码有一个根本约束：**不能有运行时开销**。

```
虚函数（Virtual Function）：
  → vtable 查找需要额外内存访问
  → 阻止函数内联1
  → 阻止编译器优化（分支预测、常量折叠）
  → 在 GPU 上每次虚函数调用可能损失 10~50 cycles

C++ 模板：
  → 编译期确定所有类型和尺寸
  → 编译器完全内联所有函数调用
  → 循环展开 + 常量折叠 + 死代码消除
  → 零运行时开销，等效于手写汇编
```

### 模板参数 = 性能调优旋钮

CUTLASS 中每个模板参数都是一个**可独立调节的性能旋钮**，且在编译期固定：

```cpp
using CutlassGemm = cutlass::gemm::device::Gemm<
  float,                              // ← 数据类型 A（影响内存布局）
  cutlass::layout::ColumnMajor,       // ← 内存布局 A（影响访问模式）
  float,                              // ← 数据类型 B
  cutlass::layout::ColumnMajor,       // ← 内存布局 B
  float,                              // ← 数据类型 C（输出类型）
  cutlass::layout::ColumnMajor,       // ← 内存布局 C
  float,                              // ← 累加器类型（精度保持）
  cutlass::arch::OpClassSimt,         // ← 指令类型：SIMT or TensorOp
  cutlass::arch::Sm80,                // ← 目标架构（决定可用指令）
  cutlass::gemm::GemmShape<128,128,8>,// ← ThreadBlock Tile（Shared Mem 用量）
  cutlass::gemm::GemmShape<32, 64, 8>,// ← Warp Tile（寄存器用量）
  cutlass::gemm::GemmShape<1,  1,  1> // ← MMA 指令形状
>;
// → 编译器根据以上参数生成一个完全特化、完全展开的 Kernel
// → 不同参数组合生成完全不同的机器码，零共享开销
```

**这就是 CUTLASS 相比 cuBLAS 的关键差异**：
- cuBLAS 是一个二进制库，内部预编译了有限几种配置
- CUTLASS 是一个模板库，用户在自己代码中实例化，每个配置都是独立的最优 Kernel

### 编译期 vs 运行时的权衡

```
CUTLASS 的选择：把一切能移到编译期的都移到编译期

编译期确定（模板参数）：
  ✓ Tile 大小（ThreadBlock/Warp/Instruction）
  ✓ 数据类型（float/half/int8/fp8）
  ✓ 内存布局（RowMajor/ColumnMajor）
  ✓ 目标架构（SM80/SM90a/SM100a）
  ✓ 流水线阶段数（Stages）

运行时确定（函数参数）：
  ✓ 矩阵尺寸（M, N, K）
  ✓ 数据指针（A_ptr, B_ptr, C_ptr）
  ✓ alpha/beta 缩放系数

代价：编译时间较长（数分钟 per Kernel），这是 CUTLASS 的固有成本
收益：零运行时开销，每个 Kernel 都是硬件最优路径
```

---

## 3.4 核心设计理念三：可复用的模块化组件

### GTC 2018 的核心主张

GTC 2018 演讲题目就叫 **"Software Primitives for Dense Linear Algebra at ALL LEVELS and SCALES within CUDA"**，强调"在 CUDA 的**所有层次和规模**上"提供软件原语。

这意味着 CUTLASS 不仅仅是一个 GEMM 库，而是：

```
一套可以在任意粒度被调用的线性代数原语：

用户可以调用 Device 层：  cutlass::gemm::device::Gemm（最简单）
用户可以调用 Kernel 层：  自定义 Grid Launch + CUTLASS Kernel
用户可以调用 Collective层：在自己 Kernel 里嵌入 CUTLASS Mainloop
用户可以调用 Warp 层：    用 CUTLASS Warp-level MMA 写自定义 Kernel
用户可以调用 Instruction层：直接用 cute::arch::mma_sm90 PTX 封装
```

### 正交性原则（Orthogonality）

官方文档（`cutlass_3x_design.md`）：
> "CUTLASS 3.0 detaches its interface layers from the hardware, centering them instead around the natural structure of GEMM algorithms not tied to any particular GPU generation."

各个维度的选择**独立正交**，可以任意组合：

```
维度1：数据类型     FP32 / FP16 / BF16 / TF32 / FP8 / INT8 / INT4
维度2：内存布局     RowMajor / ColumnMajor / TensorNHWC / ...
维度3：计算指令     SIMT / TensorOp (HMMA/WMMA/WGMMA/UMMA)
维度4：目标架构     SM70 / SM75 / SM80 / SM89 / SM90a / SM100a
维度5：Tile大小     (64/128/256) × (64/128/256) × (8/16/32/64)
维度6：流水线       1 stage / 2 stage / 3~8 stage (multistage)
维度7：调度策略     DataParallel / SplitK / StreamK / Persistent

→ 理论上可以组合出数万种 Kernel 配置
→ CUTLASS Profiler 正是通过枚举这些配置来找最优 Kernel
```

### 可复用性的实现方式：Policy-Based Design

每个 CUTLASS 组件接受一个"策略（Policy）"模板参数，而不是对特定实现做假设：

```cpp
// Threadblock Mainloop 的策略注入
template <
  typename Shape,           // Tile 形状
  typename IteratorA,       // A 矩阵访问策略（可替换）
  typename SmemLayoutA,     // A 共享内存布局（可替换）
  typename IteratorB,       // B 矩阵访问策略（可替换）
  typename SmemLayoutB,     // B 共享内存布局（可替换）
  typename Mma,             // Warp-level MMA 策略（可替换）
  typename Policy           // 流水线策略（可替换）
>
struct MmaMultistage { ... };

// → 用户可以只替换其中一个参数，其他保持默认
// → 例如只换 SmemLayoutA 来测试不同 Swizzle 对 bank conflict 的影响
```

---

## 3.5 核心设计理念四：性能 > 易用性（当两者冲突时）

官方文档（`programming_guidelines.md`）原文直接表明：
> "Given a tradeoff between simplicity and performance, **CUTLASS chooses performance**."

这个取舍产生了几个对学习者重要的设计决策：

### 决策1：循环必须展开（CUTLASS_PRAGMA_UNROLL）

```cpp
// CUTLASS 内层循环必须展开，以便：
// 1. 编译器把数组元素映射到寄存器（而非访问堆栈内存）
// 2. 编译器能做指令调度，填满 MMA 流水线
// 3. 常量折叠消除索引计算开销

CUTLASS_PRAGMA_UNROLL
for (int mma_k = 0; mma_k < WarpTileK; mma_k += MmaK) {
  // 每次迭代被完全内联，MmaK 是编译期常量
}

→ 代价：Tile 大小必须是编译期常量（模板参数），不能运行时配置
→ 收益：生成的汇编代码等效于手写的最优寄存器分配序列
```

### 决策2：Params 结构体放入 Constant Memory

```cpp
// 问题：Kernel 参数（stride、指针偏移）在所有线程中相同，
//       如果每个线程从 L1 Cache 读取会造成 Cache 压力

// CUTLASS 的解法：把 Kernel 参数封装进 Params 结构体
// 在 Host 端构造好，作为 Kernel 参数传入（存入 Constant Memory）

struct Params {
  int64_t stride_a;   // 编译期不可知，但运行时所有线程共享
  int64_t stride_b;
  void*   ptr_a;
  void*   ptr_b;
  // ...
};
// → Constant Memory 对所有线程广播，延迟极低（~4 cycles）
// → 而不是让每个线程各自计算这些值
```

### 决策3：Shared Memory 显式管理（SharedStorage 模式）

CUTLASS 借鉴 [NVIDIA CUB 库](https://nvlabs.github.io/cub/)的模式，要求每个需要 Shared Memory 的组件定义内部 `SharedStorage` 结构体，并支持 `union` 复用：

```cpp
struct SharedStorage {
  union {
    typename Mma::SharedStorage mma;      // GEMM mainloop 期间使用
    typename Epilogue::SharedStorage epi; // Epilogue 期间使用（与 Mma 不重叠）
  };
};
// → union 使两部分 Shared Memory 复用同一块地址
// → GEMM 结束后，Mma 的 Shared Memory 可被 Epilogue 覆写
// → 最小化 Shared Memory 占用，提升 Occupancy
```

---

## 3.6 从 2.x 到 3.x：架构迭代的动机

理解了 2.x 的设计，才能理解 3.x（引入 CuTe）为什么是必然的重构。

### 2.x 的根本局限：Iterator 是一维的

CUTLASS 2.x 的数据访问抽象是"Iterator"（迭代器）——本质上是一个**一维的**前进指针。

```
CUTLASS 2.x Iterator 模式：
  PredicatedTileIterator<Shape, Element, Layout, ...>
  → 内部维护一个 1D 地址偏移量
  → 每次 advance() 向前移动固定步长
  → Thread-to-data 映射隐式嵌入迭代器的索引计算中

问题：
  ✗ 矩阵是 2D 的，但 Iterator 是 1D 的
    → 行主序和列主序需要完全不同的 Iterator 实现
  ✗ Thread 到数据的映射逻辑分散在不同 Iterator 类的实现里
    → 要理解一个 Kernel，必须追踪多个 Iterator 的实现
  ✗ Hopper 的 WGMMA 是 Warpgroup 级的，不是 Warp 级的
    → 完全不符合 2.x 的 Warp → Thread 层次，无法干净地表达
```

### 2.x 的 Named Types 爆炸问题

官方文档（`cutlass_3x_design.md`）原文：
> "CUTLASS 2.x design preferred introducing bespoke named types for each architecture specific thread and data layout... `gemm::threadblock` namespace contains `MmaMultistage`, `MmaPlanarComplexMultistage`, `MmaPipelined` etc. despite them providing mainloops for GEMMs."

```
CUTLASS 2.x 中实现一个新的 GEMM 变种需要：
  ✗ 新的 ThreadMap 类（定义线程-数据映射）
  ✗ 新的 Iterator 类（实现数据访问）
  ✗ 新的 Warp-level MMA 类
  ✗ 新的 Epilogue Thread 类
  ✗ 新的 default_*_configuration.h 配置
  ✗ 新的 device-level GEMM 类

→ 加入一个新架构特性（如 Hopper TMA），需要修改 10+ 个文件
→ 代码库膨胀，可读性极差，新人学习曲线极陡峭
```

### 3.x 的解法：CuTe Layout 统一所有映射

CUTLASS 3.x 引入 CuTe，用一个统一的 `cute::Layout` 类型替代了所有 Iterator：

```
CUTLASS 2.x：                    CUTLASS 3.x：
多种 Iterator 类                 → cute::Tensor (= 数据指针 + cute::Layout)
多种 ThreadMap 类                → cute::Layout (= Shape + Stride 代数)
多种 SmemLayout 类               → cute::Layout
多种 TileIterator 类             → cute::TiledCopy / cute::TiledMma

reduction: 数十种 Named Types    → 1 种词汇类型 cute::Layout
```

官方图示说明（`cutlass_3x_design.md`）：
> `cute::Layout`s always maintain logical consistency of their coordinates, allowing us to check pre- and post-conditions at compile time for all static inner loops.

---

## 3.7 设计哲学总结：CUTLASS 是什么

```
CUTLASS 不是：
  ✗ 一个调用简单的黑盒 GEMM 库（那是 cuBLAS）
  ✗ 一个方便快速使用的 Python API（那是 PyTorch/JAX 内置算子）
  ✗ 一个自动调优系统（那是 autoTVM/OpenAI Triton）

CUTLASS 是：
  ✓ GPU 高性能线性代数的"标准教材"——NVIDIA 内部如何做 GEMM 的公开版
  ✓ 一套 C++ 模板积木——可以在任意粒度组合、替换，且零额外开销
  ✓ 架构演进的"前沿阵地"——每代 GPU 的新硬件特性（TMA/WGMMA/UMMA）
    都最先在 CUTLASS 中有干净的软件抽象
  ✓ 生产级 Kernel 的起点——FlashAttention/vLLM/TensorRT-LLM 都基于它
```

### 一句话总结各版本的设计理念演进

| 版本　　　　　　　 | 核心理念　　　　　　　　　　　　　　 | 关键抽象　　　　　　　　　　　　　　|
| --------------------| --------------------------------------| -------------------------------------|
| CUTLASS 1.x (2017) | "把 cuBLAS 的内部结构公开化"　　　　 | `gemm::threadblock::Mma`　　　　　　|
| CUTLASS 2.x (2019) | "模板化的分层积木，支持 Tensor Core" | `PredicatedTileIterator` + 分层 Mma |
| CUTLASS 3.x (2023) | "正交可组合，CuTe Layout 统一一切"　 | `cute::Layout` + Collective　　　　 |
| CUTLASS 4.x (2024) | "Python 原生，DSL 降低门槛"　　　　　| CuTe DSL (Python AST → PTX)　　　　 |

---

## 3.8 从动机到代码：在 Example 00 中看设计理念的体现

回到本周的 `examples/00_basic_gemm/basic_gemm.cu`，用上面的理念重新审视：

```cpp
// ① 分层分解体现在：只暴露 Device 层接口，隐藏所有内部层次
using CutlassGemm = cutlass::gemm::device::Gemm< /* 模板参数 */ >;

// ② 模板元编程 = 编译期特化，零开销
// 不同的 ThreadBlockShape 编译后是完全不同的 Kernel 机器码
cutlass::gemm::GemmShape<128, 128, 8>   // ← 编译期常量，Loop Unroll 的关键

// ③ 正交性体现在：数据类型、布局、指令类型是独立的模板参数
//    改 float → half_t 只需改第1个参数，其他不变
cutlass::gemm::device::Gemm<
  float,                           // 只改这里就能换数据类型
  cutlass::layout::ColumnMajor,    // 只改这里就能换内存布局
  ...

// ④ Params 结构体体现了 Constant Memory 设计模式
CutlassGemm::Arguments args(
  {M, N, K},    // ← Problem size（运行时，但通过 Params 进 Constant Memory）
  {A, lda},     // ← 数据指针和 stride
  ...
);
// → gemm_operator(args) 时，args 被构造为 Params 传入 Kernel

// ⑤ Epilogue 的可扩展性（这里用默认 LinearCombination）
// 可以替换为任意自定义 Epilogue，这是 CUTLASS 的核心价值之一
{alpha, beta}   // ← 默认 Epilogue = alpha * AB + beta * C
```

理解了这 5 点，你对 CUTLASS 的认知就从"一个 GEMM 库"升级到了
**"一套用于构建任意高性能 GPU 矩阵计算内核的软件工程框架"**。