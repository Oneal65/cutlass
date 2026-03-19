# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**CUTLASS 4.4.2** — NVIDIA's header-only C++ template library for high-performance GEMM and linear algebra. This repo runs on an **H20 8-GPU machine** (SM90a, Hopper architecture, CUDA 12.8).

The learning roadmap is in `dive/claude_gen_cutlass_plan260316.md`. The working notes file is `learn.md`.

## Build Commands

```bash
# Standard H20/Hopper build (SM90a)
export CUDACXX=/usr/local/cuda/bin/nvcc
mkdir -p build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCMAKE_BUILD_TYPE=Release
make <target> -j16

# Build a single example
make 48_hopper_warp_specialized_gemm -j16

# Build all Hopper examples
cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_ENABLE_EXAMPLES=ON
make -j$(nproc)

# Build with multiple arch targets
cmake .. -DCUTLASS_NVCC_ARCHS="80;90a"

# Build the CUTLASS Profiler (needs kernel library)
cmake .. -DCUTLASS_NVCC_ARCHS=90a \
         -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_*gemm_f16_*
make cutlass_profiler -j16

# Run profiler
./tools/profiler/cutlass_profiler --operation=gemm \
    --m=4096 --n=4096 --k=4096 \
    --A=f16:column --B=f16:row --C=f32:column

# Build CuTe tutorial examples
make sgemm_1 sgemm_2 tiled_copy -j8
```

### CuTe DSL Python Setup

```bash
# Install CuTe DSL (Python kernel writing interface)
cd python/CuTeDSL
bash setup.sh               # For CUDA 12.x
# bash setup.sh --cu13      # For CUDA 13.1+

# Run a DSL example
python examples/python/CuTeDSL/hopper/dense_gemm_persistent.py

# Launch Jupyter notebooks for interactive learning
jupyter notebook examples/python/CuTeDSL/notebooks/
```

### Running Tests

```bash
cd build
# Run CuTe unit tests
make cutlass_test_unit_cute -j8
./test/unit/cute/cutlass_test_unit_cute

# Run GEMM unit tests
make cutlass_test_unit_gemm_device_sm90 -j8
ctest -R cutlass_test_unit_gemm_device_sm90

# Run a specific example to validate correctness
./examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm
```

## Hardware Context (H20 x8)

- **Architecture**: Hopper (SM90a), Compute Capability 9.0
- **Memory**: 97871 MiB per GPU
- **Key features available**: WGMMA, TMA, Warp Specialization, Persistent Kernels, Cluster, PDL
- **Use `CUDA_VISIBLE_DEVICES=0`** to target a specific GPU

## Architecture Overview

### Directory Structure

```
include/cutlass/     # CUTLASS C++ template library (Device→Kernel→Collective→Atom layers)
  gemm/              # GEMM five-layer stack
  epilogue/          # Epilogue Visitor Tree (EVT) fusion framework
  pipeline/          # SM90/SM100 async pipeline + barrier abstractions
  arch/              # Architecture feature exposure

include/cute/        # CuTe DSL — mathematical foundation of CUTLASS 3.x+
  layout.hpp         # Layout = (Shape, Stride) algebra (63KB, core concept)
  tensor.hpp         # Tensor = Engine + Layout
  atom/              # MMA Atom + Copy Atom + TiledMma/TiledCopy
  arch/              # PTX instruction wrappers (sm70→sm120)
  algorithm/         # copy(), gemm(), fill() — high-level algorithms

examples/            # 90+ C++ examples (numbered by complexity/arch)
  cute/tutorial/     # CuTe standalone tutorials: sgemm_1~4, tiled_copy, hopper/, blackwell/
  python/CuTeDSL/    # Python kernel examples (ampere/hopper/blackwell/distributed/notebooks)

python/
  pycute/            # Pure Python CuTe Layout algebra — best for learning CuTe concepts
  CuTeDSL/           # CuTe DSL: Python-native GPU kernel writing (AST→MLIR→PTX)
  cutlass_library/   # Kernel instance generator/manifest (supports profiler)

tools/
  profiler/          # CUTLASS Profiler CLI for benchmarking
  util/              # HostTensor, reference implementations, random init

test/unit/           # Google Test unit tests (cute/gemm/conv/pipeline)
dive/                # Personal learning notes and planning docs
```

### GEMM Five-Layer Architecture

```
Device    (gemm/device/)        — Host API: param validation, workspace alloc, launch
Kernel    (gemm/kernel/)        — Device main: combines Mainloop + Epilogue + TileScheduler
Collective (gemm/collective/)   — Timing orchestration: producer/consumer warp cooperation
Tiled MMA/Copy (cute/atom/)     — Spatial microkernel: extend Atom to Tile level
Atom      (cute/arch/ + atom/)  — Hardware instruction wrappers (MMA, Copy)
```

### CuTe Core Concepts

**Layout** = `(Shape, Stride)` — a function from coordinate space to index space. All thread/data partitioning in CUTLASS uses Layouts.

Key Layout algebra operations:
- `composition(A, B)` — function composition A∘B (map logical to physical layout)
- `logical_divide(L, T)` — partition Layout L by Tile T
- `zipped_divide(L, T)` — partition and reorder tile inner/outer dims
- `logical_product(L, T)` — replicate Tile T pattern across Layout L

**Tensor** = `Engine + Layout` — created with `make_tensor(pointer, layout)`.

**Atom** — hardware instruction metadata:
- `MMA_Atom`: MMA instruction shape + thread layout (e.g., SM90 WGMMA)
- `Copy_Atom`: memory transfer shape + thread layout (e.g., SM90 TMA)
- `TiledMma` / `TiledCopy`: spatial extension of Atom to tile level

### Collective Naming Convention

```
sm{arch}_{type}_{transport}_{instruction}_{smem_layout}_{schedule}[_{modifier}].hpp

Example: sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp
  sm90           = Hopper architecture
  tma            = TMA data transport
  gmma           = WGMMA instruction
  ss             = A and B both in Shared Memory (rs = A in Register, B in Shared)
  warpspecialized = Producer/Consumer warp separation scheduling
  fp8            = FP8 data type
  blockwise_scaling = block-wise quantization scaling
```

### Scheduling Strategies

| Strategy | Key File | Use Case |
|----------|----------|----------|
| Data-Parallel | `static_tile_scheduler.hpp` | Default; wave quantization issue on small problems |
| Split-K | `gemm_splitk_parallel.h` | K >> M×N; reduces to partial GEMMs |
| Stream-K | `sm90_tile_scheduler_stream_k.hpp` | Load-balanced across SMs; best for irregular shapes |
| Persistent | `sm90_gemm_tma_warpspecialized_pingpong.hpp` | Kernel stays resident; low launch overhead |
| Warp Spec. | `sm90_mma_tma_gmma_ss_warpspecialized.hpp` | Producer warps do TMA, consumer warps do WGMMA |
| Ping-Pong | `sm90_gemm_tma_warpspecialized_pingpong.hpp` | Two warpgroups alternate; maximizes pipeline utilization |

### EVT (Epilogue Visitor Tree) Fusion

Composable epilogue fusion at compile time. Node types in `include/cutlass/epilogue/fusion/`:
- `Load` — load extra data from global memory (bias, scale)
- `Compute` — element-wise ops (ReLU, GELU, Clamp, TopK, Softmax)
- `Store` — write back results (with optional type conversion or absmax)
- `Reduce` — reduction ops (partial reduce for Split-K)

Key files: `operations.hpp`, `sm90_visitor_*.hpp`, `sm90_callbacks_tma_warpspecialized.hpp` (109KB)

## Learning Path for H20 (SM90a Focus)

### Stage 1: CuTe Fundamentals
Start with PyCuTe (`python/pycute/`) and Jupyter notebooks before touching C++:
```bash
python -c "from pycute import *; L = make_layout((2,3),(3,1)); print(L)"
jupyter notebook examples/python/CuTeDSL/notebooks/cute_layout_algebra.ipynb
```

Then C++ tutorials in order:
1. `examples/cute/tutorial/sgemm_1.cu` — basic CuTe SGEMM
2. `examples/cute/tutorial/sgemm_2.cu` — TiledMma SGEMM
3. `examples/cute/tutorial/tiled_copy.cu` — TiledCopy
4. `examples/cute/tutorial/sgemm_sm80.cu` — Ampere multistage pipeline

### Stage 2: Hopper-specific (SM90a) — Relevant to H20
Key examples (compile with `-DCUTLASS_NVCC_ARCHS=90a`):
- `examples/48_hopper_warp_specialized_gemm/` — Hopper Warp Specialization GEMM
- `examples/49_hopper_gemm_with_collective_builder/` — CollectiveBuilder API
- `examples/54_hopper_fp8_warp_specialized_gemm/` — FP8 GEMM
- `examples/57_hopper_grouped_gemm/` — Grouped GEMM (MoE)
- `examples/61_hopper_gemm_with_topk_and_softmax/` — GEMM+TopK+Softmax EVT fusion
- `examples/67_hopper_fp8_..._blockwise_scaling/` — FP8 Block-wise Scaling GEMM
- `examples/88_hopper_fmha/` — Flash MHA

CuTe tutorials for Hopper:
- `examples/cute/tutorial/hopper/wgmma_sm90.cu` — WGMMA instruction
- `examples/cute/tutorial/hopper/wgmma_tma_sm90.cu` — WGMMA + TMA

### Stage 3: CuTe DSL Python (CUTLASS 4.x)
Key examples for H20 (Hopper):
```bash
python examples/python/CuTeDSL/hopper/dense_gemm_persistent.py
python examples/python/CuTeDSL/hopper/fmha.py
```

## Important Files for Core Concepts

| Concept | Key File |
|---------|----------|
| Layout algebra | `include/cute/layout.hpp` (63KB) |
| Tensor abstraction | `include/cute/tensor.hpp` |
| SM90 MMA atoms | `include/cute/arch/mma_sm90_gmma.hpp` (946KB) |
| SM90 TMA copy atoms | `include/cute/arch/copy_sm90_tma.hpp` |
| Swizzle (bank conflict avoidance) | `include/cute/swizzle.hpp` |
| Hopper collective mainloop | `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp` |
| GEMM dispatch policy | `include/cutlass/gemm/dispatch_policy.hpp` (72KB) |
| EVT operations | `include/cutlass/epilogue/fusion/operations.hpp` |
| Hopper EVT callbacks | `include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp` (109KB) |

## Common Pitfalls

- **Compilation time**: Single CUTLASS example can take 2–5 minutes. Use `-j16` and build only the needed target.
- **Architecture mismatch**: Always use `90a` (not `90`) for Hopper features like WGMMA/TMA.
- **Template error messages**: CUTLASS template errors are extremely verbose. Look for the first error in the chain.
- **`_` operator in CuTe**: `cute::_` is the slice/underscore, not a variable name.
- **Layout vs Stride**: CuTe Layout `(shape:stride)` notation; the stride is per-element, not per-byte.
- **sm90 vs sm90a**: Regular SM90 kernels work on Hopper; `SM90a` uses asynchronous warp-specialized pipeline (WGMMA requires `sm90a`).
