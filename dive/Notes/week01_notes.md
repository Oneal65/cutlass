
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