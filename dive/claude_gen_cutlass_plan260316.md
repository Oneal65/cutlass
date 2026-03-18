```text
请ultrathink 基于以下要求为我制定一个成长为 cutlass 专家的学习和实践路径规划，若下面内容列的不合理请修改补充，并最终输出到同级目录learn.md下

1. 首先全面扫描和分析当前代码仓库中的 cutlass 相关代码结构和实现
2. 检索互联网上关于 cutlass 的最佳实践和学习资源
3. 制定分阶段的学习计划，包括：
   - 基础概念掌握阶段
   - 核心功能实践阶段
   - 高级特性深入阶段
   - 实际项目应用阶段
4. 针对每个关键学习输出，请明确标注其参考来源，包括：
   - 官方文档引用
   - 权威技术博客
   - 相关论文资料
   - 优质开源项目参考

请确保规划具有可操作性，包含具体的学习目标、时间安排和实践建议。


# 原始
请ultrathink 该仓库全貌及实现原理，根据实现原理和examples 为我制定一个成长为 cutlass 专家的学习和实践路径规划, 过程中请参考互联网上的最佳实践和材料。请自行将任务拆分细致一点，内容整理后输出在learn.md文件中
```



# CUTLASS 专家成长路径规划

> **仓库版本**: CUTLASS 4.4.1 (Feb 2026)
> **目标**: 从零到精通 CUTLASS，成为能独立开发高性能 GPU Kernel 的专家
> **适用背景**: 有 CUDA 基础，熟悉底层算子开发（MoE、Attention、TopK、GEMM）、FP8/Block-wise 量化
> **总周期**: 约 26~36 周（6~9 个月）

---

## 第一部分：仓库全貌与实现原理

### 1.1 项目定位

CUTLASS 是 NVIDIA 开源的 **Header-only C++ 模板库**，用于实现高性能 GEMM 及相关线性代数运算。它将 GPU 编程中的"移动部件"分解为 **可复用、可组合的模块化软件组件**，覆盖从 Volta 到 Blackwell 全系列架构。

CUTLASS 4.x 新增了 **CuTe DSL** —— 基于 Python 的原生 GPU Kernel 编写接口，与 C++ CuTe 抽象完全一致，提供数量级更快的编译时间和与 PyTorch/JAX 的原生集成。

### 1.2 顶层目录架构

```
cutlass/                         # 项目根目录 (v4.4.1)
├── include/                     # 核心头文件库（Header-only，使用者只需 include 此目录）
│   ├── cutlass/                 # CUTLASS C++ 模板库
│   │   ├── arch/               # 架构特性暴露（指令级 MMA、barrier、memory、SIMD）
│   │   ├── gemm/               # GEMM 核心（collective/kernel/device/thread/warp/threadblock）
│   │   ├── conv/               # 卷积特化（2D/3D fprop/dgrad/wgrad, implicit GEMM）
│   │   ├── epilogue/           # Epilogue 实现（fusion/collective/thread/threadblock/warp）
│   │   ├── layout/             # 内存布局（RowMajor/ColumnMajor/TensorNHWC/Affine2等）
│   │   ├── pipeline/           # 异步流水线（SM90/SM100 TMA pipeline + Barrier 协作）
│   │   ├── transform/          # 数据变换（collective/device/kernel/thread/threadblock/warp）
│   │   ├── reduction/          # 规约操作
│   │   ├── thread/             # 线程级 SIMT 代码
│   │   ├── detail/             # 内部实现细节（blockscaled layout, collective helpers, mma helpers）
│   │   ├── experimental/       # 实验性功能（distributed GEMM）
│   │   ├── platform/           # CUDA 标准库兼容组件
│   │   └── *.h                 # 核心类型：Array、数值类型（FP8/BF16/TF32/FP4/FP6）、Tensor等
│   └── cute/                   # CuTe 核心库（CUTLASS 3.0+ 引入，整个 3.x/4.x 的基石）
│       ├── algorithm/          # 核心操作：copy, gemm, prefetch, fill, tuple_algorithms
│       ├── arch/               # PTX 指令封装（sm50/61/70/75/80/89/90/100/120 + cluster + TMA）
│       ├── atom/               # MMA Atom / Copy Atom（硬件操作元信息 + Tiled 扩展）
│       ├── container/          # Array, Tuple, BitField 等容器
│       ├── numeric/            # 数值类型与运算（integral_constant, arithmetic_tuple）
│       ├── util/               # 调试打印工具（print, print_svg, print_latex, type_traits）
│       ├── layout.hpp          # Layout 核心定义（63KB！Shape + Stride 的代数系统）
│       ├── tensor.hpp          # Tensor 抽象入口（Engine + Layout）
│       ├── stride.hpp          # Stride 操作
│       ├── swizzle.hpp         # Swizzle 模式（Bank Conflict 优化）
│       ├── swizzle_layout.hpp  # Swizzle Layout 组合
│       ├── int_tuple.hpp       # 层级化 IntTuple（CuTe 的数学基础）
│       ├── pointer*.hpp        # 指针抽象（flagged, swizzle, sparse）
│       └── underscore.hpp      # 切片操作符 _
├── examples/                   # SDK 示例程序（90+ C++ 示例 + CuTe tutorial + Python 示例）
│   ├── 00~94_*                 # 按架构和功能递进的 C++ 示例
│   ├── 111/112_*_ssd           # State Space Decomposition (Mamba) Kernel
│   ├── cute/tutorial/          # CuTe 独立教程（sgemm_1~sm80, tiled_copy, hopper/, blackwell/）
│   └── python/CuTeDSL/         # CuTe DSL Python 示例（ampere/hopper/blackwell/distributed/notebooks）
├── python/                     # Python 工具链
│   ├── CuTeDSL/                # CuTe DSL 核心（base_dsl + cute + cutlass_dsl + pipeline + jax + torch）
│   ├── cutlass_cppgen/         # C++ 代码生成器（Python 接口，backend/emit/epilogue/op）
│   ├── cutlass_library/        # Kernel 库管理（generator + manifest + heuristics）
│   └── pycute/                 # PyCuTe（Layout 代数 Python 纯实现，学习利器）
├── tools/                      # 工具链
│   ├── library/                # CUTLASS Instance Library（所有 Kernel 实例化模板）
│   ├── profiler/               # CUTLASS Profiler（性能分析命令行工具）
│   └── util/                   # 辅助工具（HostTensor、reference 实现、random init、比较工具）
├── test/unit/                  # Google Test 单元测试（cute/gemm/conv/pipeline/core）
├── docs/                       # Doxygen 文档
└── media/                      # 图片和附加 Markdown 文档
```

### 1.3 核心实现原理

#### 1.3.1 CuTe —— CUTLASS 的数学基石

CuTe (CUDA Tensors) 是 CUTLASS 3.x+ 的**核心抽象层**。理解 CuTe 是掌握 CUTLASS 的关键。

**三大核心概念：**

```
┌──────────────────────────────────────────────────────────┐
│  Layout = (Shape, Stride)                                │
│  → 一个从坐标空间到索引空间的映射函数                       │
│  → 例: Layout<(2,3), (3,1)> 表示 2×3 列主序矩阵           │
│  → 所有 GEMM 的 Tile、Thread 分配都通过 Layout 描述        │
├──────────────────────────────────────────────────────────┤
│  Tensor = Engine + Layout                                │
│  → Engine: 数据存储（指针 + 内存空间）                     │
│  → Layout: 数据组织方式                                   │
│  → make_tensor(pointer, layout) 创建                     │
├──────────────────────────────────────────────────────────┤
│  Atom = 硬件指令的元信息描述                               │
│  → MMA_Atom: 描述一条 MMA 指令的 Shape + 线程布局          │
│  → Copy_Atom: 描述一次内存搬运的 Shape + 线程布局          │
│  → TiledMma / TiledCopy: 将 Atom 在空间上扩展到 Tile 级    │
└──────────────────────────────────────────────────────────┘
```

**Layout 代数操作（关键！）：**

| 操作 | 含义 | 用途 |
|------|------|------|
| `composition(A, B)` | 函数组合 A∘B | 将逻辑 Layout 映射到物理 Layout |
| `complement(A, size)` | 求补布局 | 找到 A 未覆盖的索引空间 |
| `logical_divide(L, T)` | 逻辑分割 | 将 Layout L 按 Tile T 分块 |
| `zipped_divide(L, T)` | 压缩分割 | 分块后将 Tile 内外维度重组 |
| `logical_product(L, T)` | 逻辑乘积 | 在 Layout 上复制 Tile 模式 |
| `make_layout(s, d)` | 创建布局 | 基础构造 |

> **参考**: `include/cute/layout.hpp` (63KB), [CuTe Layout Algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html)

#### 1.3.2 GEMM 五层架构

CUTLASS 的 GEMM 实现遵循**五层架构**，从硬件指令到宿主端接口逐层抽象：

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 5: Device    (gemm/device/)                           │
│  → 宿主端接口：参数校验、workspace 分配、Kernel Launch       │
│  → GemmUniversalAdapter: 统一适配器                          │
├──────────────────────────────────────────────────────────────┤
│  Layer 4: Kernel    (gemm/kernel/)                           │
│  → 设备端主函数：组合 Mainloop + Epilogue + TileScheduler    │
│  → GemmUniversal: 3.x 统一入口                              │
│  → sm90_gemm_tma_warpspecialized*.hpp: Hopper Kernel 实现    │
│  → sm100_gemm_tma_warpspecialized*.hpp: Blackwell 实现       │
│  → Tile Scheduler: 负责 CTA 到问题空间的映射和调度           │
│    - static_tile_scheduler: 传统数据并行                     │
│    - sm90_tile_scheduler_stream_k: Stream-K 负载均衡         │
│    - sm100_tile_scheduler: Blackwell 专用调度器              │
├──────────────────────────────────────────────────────────────┤
│  Layer 3: Collective (gemm/collective/)                      │
│  → 时序编排：Producer/Consumer Warp 协作 + Pipeline 管理     │
│  → sm80_mma_multistage: Ampere 多阶段流水线                 │
│  → sm90_mma_tma_gmma_ss_warpspecialized: Hopper WarpSpec    │
│  → sm100_mma_warpspecialized: Blackwell UMMA                │
│  → 使用 CollectiveBuilder 自动选择最优 Collective            │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: Tiled MMA/Copy  (cute/atom/)                       │
│  → 空间微核：将 Atom 扩展为 Tile 级操作                     │
│  → TiledMma: 多个 MMA Atom 的空间组合                       │
│  → TiledCopy: 多个 Copy Atom 的空间组合                     │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: Atom      (cute/arch/ + cute/atom/)                │
│  → 硬件指令封装：MMA Atom、Copy Atom                        │
│  → SM70 HMMA / SM80 MMA / SM90 WGMMA / SM100 UMMA           │
│  → SM80 cp.async / SM90 TMA / SM100 TMA + TMEM              │
└──────────────────────────────────────────────────────────────┘
```

#### 1.3.3 Epilogue 融合框架（EVT）

Epilogue Visitor Tree (EVT) 允许在 GEMM 输出后**编译期组合任意后处理操作**：

```
EVT 节点类型（include/cutlass/epilogue/fusion/）：
├── Load 节点: 从 Global Memory 加载额外数据（bias、scale 等）
├── Compute 节点: 元素级运算（ReLU、GELU、Clamp、TopK、Softmax）
├── Store 节点: 写回结果（含可选的类型转换和 absmax 计算）
└── Reduce 节点: 规约操作（partial reduce 用于 Split-K）

文件索引:
├── operations.hpp                    # EVT 节点定义
├── sm90_visitor_*.hpp                # Hopper EVT 实现
├── sm100_visitor_*.hpp               # Blackwell EVT 实现
├── sm90_visitor_topk_softmax.hpp     # TopK + Softmax 融合节点
└── sm90_callbacks_tma_warpspecialized.hpp  # Hopper EVT 回调（109KB）
```

> **参考论文**: [EVT: Accelerating Deep Learning Training with Epilogue Visitor Tree](https://dl.acm.org/doi/10.1145/3620666.3651369), ASPLOS 2024

#### 1.3.4 架构支持矩阵

| GPU 架构 | CC | 代表 GPU | CUTLASS 关键特性 | CuTe Arch 文件 |
|---------|-----|---------|-----------------|----------------|
| Volta | 7.0 | V100 | HMMA, FP16 Tensor Core | `mma_sm70.hpp` |
| Turing | 7.5 | T4, RTX 20x0 | INT8/INT4 Tensor Core | `mma_sm75.hpp` |
| Ampere | 8.0 | A100 | TF32, Sparse TC, cp.async | `mma_sm80.hpp` (70KB) |
| Ada | 8.9 | L40, RTX 40x0 | FP8 (e4m3/e5m2) TC | `mma_sm89.hpp` |
| Hopper | 9.0a | H100/H200 | WGMMA, TMA, Cluster, WarpSpec | `mma_sm90_gmma.hpp` (946KB!) |
| Blackwell DC | 10.0a | B200/B300 | UMMA, 2-SM MMA, TMEM, NVFP4 | `mma_sm100_umma.hpp` (80KB) |
| Blackwell GF | 12.0 | RTX 50x0 | GeForce TC, Sparse MMA | `mma_sm120.hpp` (111KB) |

#### 1.3.5 关键调度策略

| 策略 | 实现文件 | 核心思想 |
|------|---------|---------|
| Data-Parallel | `static_tile_scheduler.hpp` | 每个 CTA 处理固定 Tile，简单但有 Wave Quantization 问题 |
| Split-K | `gemm_splitk_parallel.h` | K 维度分割，多次 partial GEMM 后 reduce，适合 K 远大于 M×N |
| Stream-K | `sm90_tile_scheduler_stream_k.hpp` | 按工作量均匀分配到 SM，消除 Tile 粒度负载不均 |
| Persistent | `sm90_gemm_tma_warpspecialized_pingpong.hpp` | Kernel 常驻不退出，减少 launch overhead |
| Warp Specialization | `sm90_mma_tma_gmma_ss_warpspecialized.hpp` | Producer Warp 负责数据搬运，Consumer Warp 负责计算 |
| Ping-Pong | `sm90_gemm_tma_warpspecialized_pingpong.hpp` | 两组 Warpgroup 交替执行，最大化流水线利用率 |
| PDL (Dependent Launch) | `arch/grid_dependency_control.h` | Hopper+ 支持依赖 Kernel 在同一 Stream 重叠执行 |

---

### 1.4 Collective 文件命名规则解读

仓库中 collective 目录有 50+ 个文件，命名遵循固定模式：

```
sm{arch}_{type}_[modifier]_{warpspecialized|cooperative|pingpong}.hpp

示例解读：
sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp
│    │   │    │    │   │                │    │
│    │   │    │    │   │                │    └── blockwise scaling (块级缩放)
│    │   │    │    │   │                └── FP8 数据类型
│    │   │    │    │   └── warp specialization 调度
│    │   │    │    └── ss = A,B 都在 Shared Memory
│    │   │    └── gmma = 使用 WGMMA 指令
│    │   └── tma = 使用 TMA 搬运数据
│    └── mma = Matrix Multiply Accumulate
└── SM90 = Hopper 架构

rs = A 在 Register, B 在 Shared Memory
ss = A, B 都在 Shared Memory
array = Grouped/Batched GEMM（Ptr Array 模式）
```

### 1.5 Python 工具链架构

```
python/
├── pycute/                 # Layout 代数的纯 Python 实现（学习 CuTe 概念的最佳入口）
│   ├── layout.py          # Layout 运算（composition, complement, product 等）
│   ├── int_tuple.py       # 层级 IntTuple
│   └── swizzle.py         # Swizzle 实现
├── cutlass_cppgen/        # Python → C++ CUTLASS Kernel 生成
│   ├── op/gemm.py         # GEMM 操作封装
│   ├── epilogue/          # Epilogue/EVT Python 接口
│   └── backend/           # JIT 编译 + Runtime 管理
├── cutlass_library/       # Kernel 库管理（支撑 Profiler 的 Kernel 枚举 + 实例化）
│   ├── generator.py       # 全量 Kernel 生成器（497KB！）
│   ├── sm90_utils.py      # Hopper Kernel 配置
│   └── sm100_utils.py     # Blackwell Kernel 配置
└── CuTeDSL/               # CuTe DSL（CUTLASS 4.x 重磅特性）
    ├── cutlass/base_dsl/  # DSL 基础设施（AST → MLIR → PTX）
    ├── cutlass/cute/      # CuTe Python 绑定（Layout/Tensor/Atom/Algorithm）
    └── cutlass/torch.py   # PyTorch 集成接口
```

---

## 第二部分：示例索引（按学习阶段排序）

### 2.1 入门阶段示例

| 编号 | 目录 | 核心内容 | 架构 |
|------|------|---------|------|
| 00 | `00_basic_gemm` | 基础 SGEMM（SIMT），CUTLASS 第一个程序 | SM50+ |
| 01 | `01_cutlass_utilities` | CUTLASS 工具类：HostTensor、随机初始化、比较 | ALL |
| 02 | `02_dump_reg_shmem` | 调试工具：打印寄存器和共享内存 | ALL |
| 03 | `03_visualize_layout` | Layout 可视化工具 | ALL |
| 04 | `04_tile_iterator` | Tile Iterator 概念演示 | ALL |

### 2.2 基础阶段示例

| 编号 | 目录 | 核心内容 | 架构 |
|------|------|---------|------|
| 05 | `05_batched_gemm` | Batched Strided GEMM | SM50+ |
| 06 | `06_splitK_gemm` | Split-K 并行规约 | SM50+ |
| 07 | `07_volta_tensorop_gemm` | Volta Tensor Core 混合精度 GEMM | SM70 |
| 08 | `08_turing_tensorop_gemm` | Turing INT8 Tensor Core GEMM | SM75 |
| 12 | `12_gemm_bias_relu` | GEMM + Bias + ReLU 融合 | SM75+ |
| 14 | `14_ampere_tf32_tensorop_gemm` | Ampere TF32 隐式转换 GEMM | SM80 |
| 20 | `20_simt_canonical` | Canonical SIMT GEMM（理解基础分层） | SM50+ |

### 2.3 进阶阶段示例

| 编号 | 目录 | 核心内容 | 架构 |
|------|------|---------|------|
| 15 | `15_ampere_sparse_tensorop_gemm` | 2:4 结构化稀疏 Tensor Core | SM80 |
| 24 | `24_gemm_grouped` | Grouped GEMM（MoE 场景基础） | SM80 |
| 35 | `35_gemm_softmax` | GEMM + Softmax 融合 | SM80 |
| 37 | `37_gemm_layernorm_gemm_fusion` | GEMM → LayerNorm → GEMM 融合 | SM80 |
| 41 | `41_fused_multi_head_attention` | Fused MHA（变长序列） | SM80 |
| 47 | `47_ampere_gemm_universal_streamk` | Stream-K 调度 vs Data-Parallel vs Split-K | SM80 |

### 2.4 高级阶段示例（Hopper）

| 编号 | 目录 | 核心内容 | 架构 |
|------|------|---------|------|
| 48 | `48_hopper_warp_specialized_gemm` | Hopper Warp Specialization GEMM | SM90a |
| 49 | `49_hopper_gemm_with_collective_builder` | CollectiveBuilder API 使用 | SM90a |
| 54 | `54_hopper_fp8_warp_specialized_gemm` | FP8 GEMM | SM90a |
| 55 | `55_hopper_mixed_dtype_gemm` | 混合精度 GEMM（不同 A/B 类型） | SM90a |
| 57 | `57_hopper_grouped_gemm` | Hopper Grouped GEMM | SM90a |
| 61 | `61_hopper_gemm_with_topk_and_softmax` | GEMM + TopK + Softmax 融合 | SM90a |
| 62 | `62_hopper_sparse_gemm` | Hopper 稀疏 GEMM | SM90a |
| 67 | `67_hopper_fp8_..._blockwise_scaling` | FP8 Block-wise Scaling GEMM | SM90a |
| 88 | `88_hopper_fmha` | Hopper Flash MHA | SM90a |
| 111 | `111_hopper_ssd` | Hopper State Space Decomposition (Mamba) | SM90a |

### 2.5 前沿阶段示例（Blackwell）

| 编号 | 目录 | 核心内容 | 架构 |
|------|------|---------|------|
| 70 | `70_blackwell_gemm` | Blackwell UMMA 基础 GEMM | SM100a |
| 71 | `71_blackwell_gemm_with_collective_builder` | Blackwell CollectiveBuilder + EVT | SM100a |
| 72 | `72_blackwell_narrow_precision_gemm` | Block-scaled NVFP4/MXFP4/6/8 GEMM | SM100a |
| 74 | `74_blackwell_gemm_streamk` | Blackwell Stream-K | SM100a |
| 75 | `75_blackwell_grouped_gemm` | Blackwell Grouped GEMM | SM100a |
| 77 | `77_blackwell_fmha` | Blackwell FMHA | SM100a |
| 81 | `81_blackwell_gemm_blockwise` | Blackwell Blockwise GEMM | SM100a |
| 82 | `82_blackwell_distributed_gemm` | 跨 GPU 分布式 GEMM | SM100a |
| 83/84 | `83/84_blackwell_sparse_gemm` | Blackwell Sparse GEMM | SM100a |
| 92 | `92_blackwell_moe_gemm` | Blackwell MoE GEMM | SM100a |
| 93 | `93_blackwell_low_latency_gqa` | Blackwell 低延迟 GQA (Flash Decoding) | SM100a |
| 112 | `112_blackwell_ssd` | Blackwell Mamba SSD Kernel | SM100a |

### 2.6 CuTe 独立教程

| 文件 | 内容 | 难度 |
|------|------|------|
| `cute/tutorial/sgemm_1.cu` | 用 CuTe 从零写 SGEMM（基础版） | ⭐⭐ |
| `cute/tutorial/sgemm_2.cu` | 使用 TiledMma 的 SGEMM | ⭐⭐⭐ |
| `cute/tutorial/sgemm_sm70.cu` | Volta Tensor Core SGEMM | ⭐⭐⭐ |
| `cute/tutorial/sgemm_sm80.cu` | Ampere SGEMM (cp.async + multistage) | ⭐⭐⭐⭐ |
| `cute/tutorial/tiled_copy.cu` | TiledCopy 教程 | ⭐⭐⭐ |
| `cute/tutorial/hopper/wgmma_sm90.cu` | Hopper WGMMA 教程 | ⭐⭐⭐⭐ |
| `cute/tutorial/hopper/wgmma_tma_sm90.cu` | Hopper WGMMA + TMA | ⭐⭐⭐⭐⭐ |
| `cute/tutorial/blackwell/01~05_*.cu` | Blackwell 5 步渐进教程 | ⭐⭐⭐⭐⭐ |

### 2.7 CuTe DSL Python 示例（CUTLASS 4.x 重点方向）

| 目录 | 内容 |
|------|------|
| `python/CuTeDSL/ampere/sgemm.py` | Ampere SGEMM（Python 版） |
| `python/CuTeDSL/ampere/flash_attention_v2.py` | Ampere FlashAttention v2 |
| `python/CuTeDSL/hopper/dense_gemm_persistent.py` | Hopper Persistent GEMM |
| `python/CuTeDSL/blackwell/dense_gemm.py` | Blackwell Dense GEMM |
| `python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py` | Blackwell Block-scaled GEMM |
| `python/CuTeDSL/blackwell/fmha.py` | Blackwell FMHA |
| `python/CuTeDSL/blackwell/grouped_gemm.py` | Blackwell Grouped GEMM |
| `python/CuTeDSL/blackwell/mla/mla_decode_fp8.py` | **Blackwell MLA Decode FP8**（DeepSeek-V3 风格） |
| `python/CuTeDSL/blackwell/mamba2_ssd/` | Blackwell Mamba2 SSD |
| `python/CuTeDSL/blackwell/mixed_input_fmha/` | Mixed-input FMHA（INT4 KV cache） |
| `python/CuTeDSL/blackwell/epilogue/` | Epilogue Fusion Configuration (EFC) |
| `python/CuTeDSL/distributed/` | All-Reduce / AG-GEMM / RS-GEMM |
| `python/CuTeDSL/notebooks/` | **交互式 Jupyter 教程**（layout_algebra, async_pipeline, tour_to_sol_gemm 等） |

---

## 第三部分：分阶段学习计划

### 阶段一：基础概念掌握（4~6 周）

**目标**: 
- 理解 GPU 硬件层次结构（Grid → Block → Warp → Thread）与 GEMM 映射关系
- 掌握 CUTLASS 基础类型系统、Layout 概念和构建流程
- 能够编译运行并理解基础 GEMM 示例

**Week 1-2：CUDA GEMM 原理与 CUTLASS 入门**

| 任务 | 参考资料 |
|------|---------|
| 重温 CUDA 内存层级（Global → Shared → Register）和 coalesced access | [CUDA C Programming Guide - Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy) |
| 学习 GEMM 分块策略：ThreadBlock Tile → Warp Tile → Thread Tile | [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html) ⭐⭐⭐ |
| 理解 Tensor Core 工作原理（MMA 指令、数据布局要求） | [GTC 2018: CUTLASS Primitives](http://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf) |
| 搭建编译环境，运行 Example 00/01/03 | [Quick Start Guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html) |

```bash
# 编译示例
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=80  # 或 90a/100a
make 00_basic_gemm 01_cutlass_utilities 03_visualize_layout -j16
```

**实践**: 修改 `examples/00_basic_gemm/basic_gemm.cu` 的矩阵大小和数据类型（FP32 → FP16），观察性能变化。

**Week 3-4：CUTLASS 类型系统与 Layout**

| 任务 | 具体内容 | 参考资料 |
|------|---------|---------|
| 2.1 | 掌握基础数值类型：`half_t`, `bfloat16_t`, `tfloat32_t`, `float_e4m3_t/float_e5m2_t` | [官方文档：Fundamental Types](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html) |
| 2.2 | 理解 Layout 系统：`RowMajor`, `ColumnMajor`, `TensorNHWC` 及其 Stride 表示 | [官方文档：Layouts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/layout.html) |
| 2.3 | 学习 CUTLASS Utilities：`HostTensor`, `DeviceAllocation`, 随机初始化与比较 | `examples/01_cutlass_utilities/cutlass_utilities.cu` |
| 2.4 | 理解 Tile Iterator 概念和数据搬运 | `examples/04_tile_iterator/tile_iterator.cu` |
| 2.5 | 了解 CUTLASS 2.x vs 3.x API 设计差异 | [官方文档：GEMM API 2.x](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html) vs [GEMM API 3.x](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html) |

**实践项目**：使用 `examples/03_visualize_layout` 可视化不同 Layout 的内存映射；手动创建 HostTensor 并进行 GEMM 验证。
**Week 5-6：CuTe 基础**

| 任务 | 参考资料 |
|------|---------|
| 理解 CuTe Layout 代数：Shape + Stride + Layout 三元组 | [CuTe Quickstart](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html) ⭐⭐⭐ |
| 掌握 `make_layout/make_shape/make_stride` 基本操作 | [CuTe Layout](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html) |
| 学习 Layout 代数：composition, complement, logical_divide, zipped_divide | [CuTe Layout Algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html) ⭐⭐⭐ |
| 理解 CuTe Tensor：Engine + Layout，make_tensor | [CuTe Tensor](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html) |
| 运行 PyCuTe 理解 Layout 运算 | `python/pycute/` 下测试文件 |
| 用 CuTe DSL Notebook 交互式学习 | `examples/python/CuTeDSL/notebooks/cute_layout_algebra.ipynb` |

**实践**: 阅读 `examples/cute/tutorial/sgemm_1.cu`，用 `cute::print_layout()` 可视化多种 Layout 组合。

**阶段一检验标准**:
- [ ] 能独立编译运行 CUTLASS 示例并解释输出
- [ ] 能画出 GEMM 的 ThreadBlock → Warp → Thread 分块图
- [ ] 能解释 CuTe Layout `((2,4),(3,2)):((1,8),(2,16))` 的含义并手算索引
- [ ] 能修改基础示例的数据类型和问题规模并验证正确性

---

### 阶段二：核心功能实践（6~8 周）

**目标**: 
- 精通 CuTe 的 MMA Atom / Copy Atom 体系
- 理解 CUTLASS 2.x 的 Threadblock → Warp → Thread 三层 GEMM
- 掌握 CUTLASS 3.x 的 Collective → Kernel → Device 三层架构
- 能使用 CUTLASS Profiler 进行性能分析

**Week 7-8：CuTe 进阶——Atom 与 TiledMma/TiledCopy**

| 任务 | 参考资料 |
|------|---------|
| 理解 MMA_Atom：硬件 MMA 指令的元信息封装 | `include/cute/atom/mma_atom.hpp`, `mma_traits_sm80.hpp` |
| 理解 Copy_Atom：数据搬运的元信息封装 | `include/cute/atom/copy_atom.hpp`, `copy_traits_sm80.hpp` |
| 掌握 TiledMma：Atom → Tile 级 MMA 的空间扩展 | `include/cute/atom/mma_atom.hpp` 中的 `TiledMma` 定义 |
| 掌握 TiledCopy：Atom → Tile 级 Copy 的空间扩展 | `examples/cute/tutorial/tiled_copy.cu` |
| 理解 Swizzle 机制及其在避免 Bank Conflict 中的作用 | `include/cute/swizzle.hpp`, `include/cute/swizzle_layout.hpp` |

**实践**: 
- 运行 `examples/cute/tutorial/sgemm_2.cu`（使用 TiledMma 的 SGEMM）
- 对比 `sgemm_1.cu` 和 `sgemm_2.cu` 的性能差异，分析原因

**Week 9-10：CUTLASS 2.x GEMM 深入**

| 任务 | 参考资料 |
|------|---------|
| 理解 2.x 四级层次：Device → Kernel → Threadblock → Warp | [GEMM API 2.x](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html) |
| 分析 Volta/Ampere Tensor Core GEMM | `examples/07_volta_tensorop_gemm/`, `examples/14_ampere_tf32_tensorop_gemm/` |
| 理解 Epilogue 融合：LinearCombination/ReLU/Clamp | `include/cutlass/epilogue/thread/` |
| 学习 Batched GEMM 与 Split-K | `examples/05_batched_gemm/`, `examples/06_splitK_gemm/` |

**实践**: 
- 分析 `examples/12_gemm_bias_relu/` 的 Epilogue 融合实现
- 尝试自定义一个简单的 Epilogue（如 GELU 激活函数）

**Week 11-12：CUTLASS 3.x Collective 架构**

| 任务 | 参考资料 |
|------|---------|
| 理解 3.x 设计哲学：正交性、可组合性 | [NVIDIA Blog: CUTLASS 3.x Design](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/) ⭐⭐⭐ |
| 学习 CollectiveBuilder API | `examples/49_hopper_gemm_with_collective_builder/` |
| 理解 Dispatch Policy 与 Schedule | `include/cutlass/gemm/dispatch_policy.hpp` (72KB) |
| 分析 SM80 Multistage Mainloop | `include/cutlass/gemm/collective/sm80_mma_multistage.hpp` |
| 学习 GemmUniversal 统一入口 | `include/cutlass/gemm/kernel/gemm_universal.hpp` |

**Week 13-14：性能分析工具链**

| 任务 | 参考资料 |
|------|---------|
| 掌握 CUTLASS Profiler | [Profiler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html) |
| 学习 NVIDIA Nsight Compute 分析 Kernel | [Nsight Compute Docs](https://docs.nvidia.com/nsight-compute/) |
| 理解性能指标：峰值利用率、Memory BW Utilization | CUTLASS 性能图 `media/images/` |

```bash
# Profiler 构建与运行
cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_*gemm_f16_*
make cutlass_profiler -j16
./tools/profiler/cutlass_profiler --operation=gemm --m=4096 --n=4096 --k=4096 \
    --A=f16:column --B=f16:row --C=f32:column
```

**阶段二检验标准**:
- [ ] 能画出 MMA Atom → TiledMma 的扩展过程
- [ ] 能解释 CUTLASS 2.x 和 3.x GEMM 的执行流程差异
- [ ] 能使用 Profiler 生成性能报告并分析瓶颈
- [ ] 能通过修改 Collective Builder 参数调优 GEMM 性能

---

### 阶段三：高级特性深入（8~10 周）

**目标**: 
- 精通 Hopper 架构特性（TMA、WGMMA、Warp Specialization、Cluster）
- 掌握 Persistent Kernel、Stream-K 等高级调度策略
- 理解 Epilogue Visitor Tree (EVT) 融合框架
- 能处理 FP8/Block-wise Scaling、Sparse、Mixed-Precision 等高级场景

**Week 15-17：Hopper 架构精通**

| 任务 | 参考资料 |
|------|---------|
| 理解 TMA (Tensor Memory Accelerator) | [Colfax: CUTLASS Tutorial WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper) ⭐⭐⭐ |
| 理解 WGMMA：Warpgroup 级异步矩阵运算 | `include/cute/arch/mma_sm90.hpp`, `mma_sm90_gmma.hpp` |
| 掌握 Warp Specialization：Producer/Consumer 分工 | `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp` |
| 理解 Thread Block Cluster 与 Distributed Shared Memory | `include/cute/arch/cluster_sm90.hpp` |
| 分析 Hopper FP8 GEMM | `examples/54_hopper_fp8_warp_specialized_gemm/` |
| 运行 CuTe Hopper Tutorial | `examples/cute/tutorial/hopper/wgmma_sm90.cu` |

> **关键论文**: [A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper using CUTLASS](https://arxiv.org/abs/2312.11918) — Bikshandi, Shah, 2023

**实践项目**：
- 对比 `examples/48_hopper_warp_specialized_gemm/` 与 `examples/14_ampere_tf32_tensorop_gemm/` 的性能
- 使用 Nsight Compute 分析 TMA 命中率和 WGMMA 利用率
**Week 18-19：高级调度策略**

| 任务 | 参考资料 |
|------|---------|
| 理解 Persistent Kernel 设计理念与 Wave Quantization 问题 | [Colfax: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) |
| 掌握 Stream-K 并行分解：消除 Tile 粒度的负载不均 | `include/cutlass/gemm/kernel/gemm_universal_streamk.h` |
| 理解 Dependent Kernel Launch（Hopper PDL） | [官方文档：Dependent Kernel Launch](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/dependent_kernel_launch.html) |
| 分析 Ping-Pong Kernel 设计 | [PyTorch Blog: CUTLASS Ping-Pong GEMM Kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/) |

> **关键论文**: [Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication](https://arxiv.org/abs/2301.03598) — Osama et al., 2023

**Week 20-21：EVT Epilogue 融合**

| 任务 | 参考资料 |
|------|---------|
| 理解 EVT 框架 | [EVT Paper](https://dl.acm.org/doi/10.1145/3620666.3651369), ASPLOS 2024 |
| 分析 EVT 节点类型 | `include/cutlass/epilogue/fusion/operations.hpp` |
| 使用 CollectiveBuilder 构建 EVT GEMM | `examples/71_blackwell_gemm_with_collective_builder/` |
| 自定义 Epilogue Fusion | `python/cutlass_cppgen/epilogue/` |

**实践**: 
- 实现 GEMM + LayerNorm 融合（参考 `examples/37_gemm_layernorm_gemm_fusion/`）
- 实现 GEMM + TopK + Softmax 融合（参考 `examples/61_hopper_gemm_with_topk_and_softmax/`）

**Week 22-24：高级数据类型与稀疏**

| 任务 | 参考资料 |
|------|---------|
| FP8 Block-wise Scaling GEMM 实现 | `examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/` |
| Mixed-Dtype GEMM（如 FP8 × FP16 → FP32） | `examples/55_hopper_mixed_dtype_gemm/` |
| Structured Sparsity (2:4) | `examples/15_ampere_sparse_tensorop_gemm/`, `examples/62_hopper_sparse_gemm/` |
| Grouped GEMM（MoE 场景核心） | `examples/57_hopper_grouped_gemm/` |
| Block-Scaled Narrow Precision（NVFP4、MXFP4/6/8） | `examples/72_blackwell_narrow_precision_gemm/` |

**关键论文**：
> [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Tri Dao, 2023
>
> [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841) — Gale et al., MLSys 2023
>
> [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024（使用 CUTLASS 的 FP8 训练实践）
**阶段三检验标准**:
- [ ] 能解释 Hopper Warp Specialization 中 Producer/Consumer 的协作流程（含 Barrier 同步）
- [ ] 能分析 Stream-K 与传统 Data-Parallel 的性能差异及适用场景
- [ ] 能使用 EVT 框架实现自定义 Epilogue 融合
- [ ] 能实现 FP8 Block-wise Scaling GEMM 并验证数值精度

---

### 阶段四：实际项目应用（8~12 周）

**目标**: 独立开发生产级 Kernel、掌握 Blackwell 前沿、CuTe DSL + 框架集成
- 能独立开发生产级 CUTLASS Kernel
- 掌握 Blackwell 最新架构特性
- 能将 CUTLASS 集成到 LLM Serving / Training 流水线
- 掌握 CuTe DSL Python 编程范式

**Week 25-28：Blackwell 架构与前沿特性**

| 具体内容 | 参考资料 |
|---------|---------|
| Blackwell UMMA（Unified Matrix Multiply-Accumulate） | `examples/70_blackwell_gemm/` |
| 2-SM MMA 与 TMEM（Tensor Memory） | `examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu` |
| Blackwell Stream-K 调度 | `examples/74_blackwell_gemm_streamk/` |
| Blackwell MoE GEMM | `examples/92_blackwell_moe_gemm/` |
| Blackwell Low-Latency GQA（Decode 阶段） | `examples/93_blackwell_low_latency_gqa/` |
| Blackwell FMHA | `examples/77_blackwell_fmha/` |
| 渐进学习 CuTe Blackwell Tutorial（5 步） | `examples/cute/tutorial/blackwell/01~05_*.cu` |

**关键资源**：
> [Colfax: CUTLASS Tutorial GEMM with Thread Block Clusters on Blackwell GPUs](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/)

**Week 29-32：CuTe DSL（Python 原生 Kernel 开发）**

| 具体内容 | 参考资料 |
|---------|---------|
| CuTe DSL 环境搭建与基础概念 | [CuTe DSL Quick Start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html) |
| 用 Python 写 Dense GEMM Kernel | `examples/python/CuTeDSL/` |
| CuTe DSL 与 PyTorch 集成 | `python/CuTeDSL/cutlass/torch.py` |
| CuTe DSL 与 JAX 集成 | `examples/python/CuTeDSL/jax/` |
| AOT 编译与性能调优 | `examples/python/CuTeDSL/cute/export/` |
| 高级：DSL Epilogue Fusion Configuration (EFC) | CUTLASS 4.4 Release Notes |

**实践项目**：用 CuTe DSL 实现一个 Blackwell Dense Blockscaled GEMM，与 C++ 版本对比性能。

**Week 29-36：生产级项目实践**

| 项目 | 描述 | 参考示例 |
|------|------|---------|
| P1: FP8 MoE GEMM | Block-wise Scaling Grouped GEMM 适配 MoE 推理 | `68_*`, `92_*` |
| P2: Flash Attention | 基于 CUTLASS 实现 FlashAttn/FlashDecoding | `77_*`, `88_*`, `93_*` |
| P3: GEMM + Quant/Dequant 融合 | 量化反量化融入 Epilogue | EVT + `71_*` |
| P4: 分布式 GEMM | 跨 GPU GEMM（NCCL + CUTLASS） | `65_*`, `82_*` |
| P5: SSD/Mamba Kernel | State Space Decomposition | `111_*`, `112_*` |
| P6: MLA Decode Kernel | DeepSeek-V3 风格 Multi-head Latent Attention | `python/CuTeDSL/blackwell/mla/` |
**关键论文**：
> [FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion](https://arxiv.org/abs/2406.06858) — Chang et al., 2024
>
> [Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](https://arxiv.org/abs/2502.19811) — Zhang et al., 2025

**阶段四检验标准**:
- [ ] 能独立开发生产级 Kernel，性能达到理论峰值 85%+
- [ ] 能使用 CuTe DSL 快速原型化并验证 Kernel 设计
- [ ] 能分析真实 LLM 推理/训练场景中的性能瓶颈
- [ ] 能理解 CUTLASS CI、代码规范，具备向上游贡献代码的能力

---

## 第四部分：完整参考资源索引

### 4.1 官方文档

| 文档 | 链接 | 优先级 |
|------|------|-------|
| Quick Start Guide | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html | ⭐⭐⭐ |
| Efficient GEMM in CUDA | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html | ⭐⭐⭐ |
| CUTLASS 3.x Design | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_design.html | ⭐⭐⭐ |
| GEMM API 3.x | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html | ⭐⭐⭐ |
| CuTe Quickstart | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html | ⭐⭐⭐ |
| CuTe Layout | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html | ⭐⭐⭐ |
| CuTe Layout Algebra | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html | ⭐⭐⭐ |
| CuTe Tensor | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html | ⭐⭐⭐ |
| CuTe DSL Quick Start | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html | ⭐⭐⭐ |
| Code Organization | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/code_organization.html | ⭐⭐ |
| Programming Guidelines | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html | ⭐⭐ |
| Terminology | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/terminology.html | ⭐⭐ |
| Profiler | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html | ⭐⭐ |
| CuTe DSL Quick Start | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html | ⭐⭐ |
| Dependent Kernel Launch | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/dependent_kernel_launch.html | ⭐ |
| Functionality | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html | ⭐ |

### 4.2 权威技术博客

| 来源 | 文章 | 主题 |
|------|------|------|
| NVIDIA Blog | [Principled Abstractions for Multidimensional Data](https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels) | CuTe 设计哲学 |
| NVIDIA Blog | [CUTLASS 3.x: Orthogonal, Reusable Abstractions](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/) | 3.x 架构总览 |
| Colfax Research | [WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper) | Hopper WGMMA 深入 |
| Colfax Research | [Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) | 调度策略 |
| Colfax Research | [GEMM on Blackwell](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/) | Blackwell TBC |
| Kapil Sharma | [Learn CUTLASS the Hard Way](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way-2/) | 源码逐行解析 |
| 博客园 | [深入 CUTLASS 之 CuTe 详解](https://www.cnblogs.com/naonao-scorpio/p/18951577) | CuTe 中文详解 |
| 博客园 | [细读 CuTe Layout Algebra](https://www.cnblogs.com/duzhenblog/p/19679252) | Layout 代数中文详解 |
| PyTorch Blog | [Ping-Pong GEMM Kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/) | Ping-Pong 设计 |
| Albresky's Blog | [CuTe DSL vs CuTe C++](https://www.albresky.cn/cutlass-cutedsl-vs-cutecpp) | DSL 对比 |
| mlai.blog | [Understanding Basics of CuTe/CUTLASS](https://mlai.blog/2025-05-10-cute-basics) | CuTe 入门 |

### 4.3 GTC 演讲

| 年份 | 主题 | 链接 |
|------|------|------|
| 2018 | CUTLASS: Software Primitives for Dense Linear Algebra | [Slides](http://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf) |
| 2020 | Push Tensor Cores to the Absolute Limit on A100 | [Video](https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21745/) |
| 2021 | Accelerating Convolution with Tensor Cores | [Video](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31883/) |
| 2022 | CUTLASS: Python API + NVIDIA Hopper | [Video](https://www.nvidia.com/en-us/on-demand/session/gtcfall22-a41131/) |

### 4.4 核心论文

| 论文 | 年份 | 关联 |
|------|------|------|
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) (Tri Dao) | 2023 | Hopper Kernel 优化标杆 |
| [FlashAttention on Hopper using CUTLASS](https://arxiv.org/abs/2312.11918) (Bikshandi, Shah) | 2023 | **CUTLASS FA2 Case Study** |
| [EVT: Epilogue Visitor Tree](https://dl.acm.org/doi/10.1145/3620666.3651369) (Chen et al.) | 2024 | CUTLASS EVT 框架论文 |
| [Stream-K](https://arxiv.org/abs/2301.03598) (Osama et al.) | 2023 | Stream-K 理论基础 |
| [MegaBlocks: Efficient Sparse Training with MoE](https://arxiv.org/abs/2211.15841) (Gale et al.) | 2023 | Grouped GEMM 在 MoE 的应用 |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 2024 | FP8 训练 + CUTLASS 实践 |
| [FLUX: Communication Overlap via Kernel Fusion](https://arxiv.org/abs/2406.06858) | 2024 | CUTLASS + 通信重叠 |
| [Comet: Computation-communication Overlapping for MoE](https://arxiv.org/abs/2502.19811) | 2025 | MoE + CUTLASS 最新实践 |
| [Generalized Neighborhood Attention](https://arxiv.org/abs/2504.16922) (Hassani et al.) | 2025 | CUTLASS 稀疏注意力 |
| [Graphene: IR for Optimized Tensor Computations](https://dl.acm.org/doi/pdf/10.1145/3582016.3582018) | 2023 | GPU 张量计算 IR 设计 |
| [Benchmarking GPU Tensor Cores through CUTLASS](https://www.mdpi.com/2076-3417/13/24/13022) | 2023 | CUTLASS 性能基准测试方法论 |
| [Generalized Neighborhood Attention](https://arxiv.org/abs/2504.16922) (Hassani et al.) | 2025 | 使用 CUTLASS 实现稀疏注意力 |

### 4.5 优质开源项目参考

| 项目 | 描述 | 链接 |
|------|------|------|
| FlashAttention | 高性能 Attention（大量使用 CUTLASS） | https://github.com/Dao-AILab/flash-attention |
| vLLM | LLM 推理引擎（集成 CUTLASS Kernel） | https://github.com/vllm-project/vllm |
| TensorRT-LLM | NVIDIA LLM 推理框架 | https://github.com/NVIDIA/TensorRT-LLM |
| FasterTransformer | 高性能 Transformer 推理库 | https://github.com/NVIDIA/FasterTransformer |
| Triton | OpenAI GPU 编程语言（与 CUTLASS 互补） | https://github.com/triton-lang/triton |
| NATTEN | 邻域注意力（使用 CUTLASS） | https://github.com/SHI-Labs/NATTEN |
| DeepSeek | 大模型（FP8 训练使用 CUTLASS） | https://github.com/deepseek-ai |
| ByteTransformer | 变长输入高性能 Transformer | https://github.com/bytedance/ByteTransformer |

---

## 第五部分：学习建议

### 5.1 策略

1. **CuTe 优先**: CUTLASS 3.x/4.x 全面基于 CuTe，Layout 代数是一切的基础
2. **从示例入手**: 每个阶段以对应示例为切入点，配合 `cute::print_layout()` 和 CUDA-GDB 调试
3. **由浅入深架构线**: SM80 (Ampere) → SM90 (Hopper) → SM100 (Blackwell)，不要跳级
4. **C++ 与 Python 双线并进**: C++ 掌握底层原理，CuTe DSL 提升开发效率
5. **性能数据驱动**: 每个实验都用 Profiler/Nsight Compute 收集数据，建立性能直觉
6. **善用 PyCuTe**: `python/pycute/` 是理解 Layout 代数的最快方式
7. **善用 DSL Notebooks**: `examples/python/CuTeDSL/notebooks/` 中的 11 个 Jupyter 笔记本是交互式学习的利器

### 5.2 常见陷阱

| 陷阱 | 解决方案 |
|------|---------|
| 编译时间极长（数十分钟） | 指定 `-DCUTLASS_NVCC_ARCHS=80` 只编译目标架构 |
| 模板错误信息难以阅读 | 从最内层错误开始看；善用 `static_assert` |
| `sm_90a` vs `sm_90` 混淆 | 带 `a` 的是 architecture-accelerated features，Hopper TC 必须用 `90a` |
| Layout 索引映射困惑 | 使用 `cute::print_layout()` 或 PyCuTe 可视化 |
| Shared Memory Bank Conflict | 理解 Swizzle 原理，使用 `SmemLayoutAtom` |
| FP8 精度问题 | 理解 Block-wise Scaling 的粒度选择 |
| SM100 vs SM120 不兼容 | Blackwell DC (B200) 和 GeForce (RTX 50x0) 是不同架构！ |

### 5.3 时间总结

| 阶段 | 时长 | 核心产出 |
|------|------|---------|
| 阶段一：基础概念 | 4~6 周 | 编译运行示例、理解 GEMM 分层、掌握 CuTe Layout |
| 阶段二：核心功能 | 6~8 周 | 精通 2.x/3.x 架构、Atom 体系、Profiler |
| 阶段三：高级特性 | 8~10 周 | Hopper 精通、EVT/Stream-K/Sparse/FP8/MoE |
| 阶段四：项目实践 | 8~12 周 | 生产级 Kernel、Blackwell + DSL + 框架集成 |
| **总计** | **26~36 周（~6-9 个月）** | **CUTLASS 专家** |

---

> **最后提醒**: CUTLASS 是快速迭代的项目（已到 4.4.1），建议定期 `git pull` 关注 CHANGELOG，重点跟踪 Blackwell 和 CuTe DSL 的最新进展。积极参与 [GitHub Issues](https://github.com/NVIDIA/cutlass/issues) 和 Discussion 是深入理解设计意图的最佳途径。
