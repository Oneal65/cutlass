# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**CUTLASS 4.4.2** — NVIDIA's header-only C++ template library for high-performance GEMM and linear algebra. This repo runs on an **H20 8-GPU machine** (SM90a, Hopper architecture, CUDA 12.8).

**Purpose**: This repo is a hands-on training ground for GPU programming and CUTLASS, with the goal of becoming a kernel expert. All study plans, weekly plans, and learning notes live in `dive/`.

### Study Materials (`dive/`)

| File/Dir | Purpose |
|----------|---------|
| `dive/claude_gen_cutlass_plan260316.md` | Master learning roadmap — phased plan from CuTe basics to expert-level kernel dev |
| `dive/week01_plan.md` | Week 1 detailed plan (CUDA GEMM theory + environment setup + basic examples) |
| `dive/Notes/week01_notes.md` | Week 1 study notes (memory hierarchy, GEMM tiling, etc.) |
| `dive/logs/` | Benchmark and experiment logs (e.g., `day2_baseline.log`) |
| `dive/tools/gemm_timer.cu` | Custom GEMM profiling tool |
| `dive/prompt/prompt_log.md` | Prompt interaction log — every user prompt is recorded here with timestamp |
| `learn.md` | Working notes file (root-level) |

When helping with study tasks, check the current week plan and notes in `dive/` for context on where the user is in their learning journey.

### Prompt Logging (MANDATORY)

**Every conversation turn**, append the user's prompt to `dive/prompt/prompt_log.md` using the format:

```
YYYY年MM月DD号HH:MM  "用户的prompt内容"
```

- Use the current date/time from the system
- Quote the user's original prompt verbatim (truncate to first 200 chars if extremely long)
- This is a silent background task — do it first, do not mention it to the user unless asked

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

### Useful CMake Options

| Option | Description |
|--------|-------------|
| `CUTLASS_NVCC_ARCHS` | Target SM architectures (use `90a` for H20 Hopper) |
| `CUTLASS_ENABLE_EXAMPLES` | Build examples (ON by default when not header-only) |
| `CUTLASS_ENABLE_TESTS` | Build tests (ON by default when CUDA toolkit found) |
| `CUTLASS_ENABLE_LIBRARY` | Build the CUTLASS instance library |
| `CUTLASS_ENABLE_PROFILER` | Build the profiler (requires library) |
| `CUTLASS_LIBRARY_KERNELS` | Comma-delimited kernel name filters with wildcards. `all` for everything (huge build), omit for largest-tile-only defaults |
| `CUTLASS_TEST_LEVEL` | Test verbosity level (default `0`) |

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

# Run GEMM unit tests for SM90
make cutlass_test_unit_gemm_device_sm90 -j8
ctest -R cutlass_test_unit_gemm_device_sm90

# Run all unit tests
make test_unit -j8

# Run a specific example to validate correctness
./examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm

# Available test subdirectories under test/unit/:
#   cute, gemm, conv, epilogue, pipeline, layout, core, reduction, transform, cluster_launch, substrate
```

## Hardware Context (H20 x8)

- **Architecture**: Hopper (SM90a), Compute Capability 9.0
- **Memory**: 97871 MiB per GPU
- **Key features available**: WGMMA, TMA, Warp Specialization, Persistent Kernels, Cluster, PDL
- **Use `CUDA_VISIBLE_DEVICES=0`** to target a specific GPU

## Architecture Overview

### Directory Structure

```
include/cutlass/     # CUTLASS C++ template library (Device->Kernel->Collective->Atom layers)
  gemm/              # GEMM five-layer stack
    device/          # Host API: GemmUniversalAdapter (param validation, workspace, launch)
    kernel/          # Device main: combines Mainloop + Epilogue + TileScheduler
    collective/      # Timing orchestration: producer/consumer warp cooperation
      builders/      # CollectiveBuilder specializations per arch (sm90, sm100, sm120)
  epilogue/          # Epilogue Visitor Tree (EVT) fusion framework
    collective/      # Epilogue collective builders (sm90_builder.inl, etc.)
    fusion/          # EVT node types (operations.hpp, sm90_visitor_*.hpp)
  pipeline/          # SM90/SM100 async pipeline + barrier abstractions
  conv/              # Convolution (im2col-based, reuses GEMM collective)
  arch/              # Architecture feature exposure

include/cute/        # CuTe DSL -- mathematical foundation of CUTLASS 3.x+
  layout.hpp         # Layout = (Shape, Stride) algebra (63KB, core concept)
  tensor.hpp         # Tensor = Engine + Layout
  swizzle.hpp        # Swizzle functions for bank conflict avoidance
  atom/              # MMA Atom + Copy Atom + TiledMma/TiledCopy
  arch/              # PTX instruction wrappers (sm70->sm120)
  algorithm/         # copy(), gemm(), fill() -- high-level algorithms

examples/            # 90+ C++ examples (numbered by complexity/arch)
  cute/tutorial/     # CuTe standalone tutorials: sgemm_1~4, tiled_copy, hopper/, blackwell/
  python/CuTeDSL/    # Python kernel examples (ampere/hopper/blackwell/distributed/notebooks)

python/
  pycute/            # Pure Python CuTe Layout algebra -- best for learning CuTe concepts
  CuTeDSL/           # CuTe DSL: Python-native GPU kernel writing (AST->MLIR->PTX)
  cutlass_library/   # Kernel instance generator/manifest (supports profiler)

tools/
  profiler/          # CUTLASS Profiler CLI for benchmarking
  util/              # HostTensor, reference implementations, random init

test/unit/           # Google Test unit tests (cute/gemm/conv/pipeline)
dive/                # Personal learning notes and planning docs
```

### GEMM Five-Layer Architecture

```
Device    (gemm/device/)        -- Host API: param validation, workspace alloc, launch
  Key type: GemmUniversalAdapter -- wraps any Kernel into a host-callable interface
Kernel    (gemm/kernel/)        -- Device main: combines Mainloop + Epilogue + TileScheduler
  sm90_gemm_tma_warpspecialized*.hpp -- three warp-spec variants (basic, cooperative, pingpong)
Collective (gemm/collective/)   -- Timing orchestration: producer/consumer warp cooperation
  CollectiveBuilder -- high-level API that auto-selects collective impl from (arch, dtype, layout, schedule)
  CollectiveMma -- low-level direct parameterization of the collective mainloop
Tiled MMA/Copy (cute/atom/)     -- Spatial microkernel: extend Atom to Tile level
Atom      (cute/arch/ + atom/)  -- Hardware instruction wrappers (MMA, Copy)
```

**Typical CUTLASS 3.x kernel assembly pattern** (see example 49):
1. Use `CollectiveBuilder` to create a `CollectiveMainloop` from (arch, opclass, dtypes, layouts, tile_shape, schedule)
2. Use `EpilogueBuilder` to create a `CollectiveEpilogue` (with optional EVT fusion)
3. Combine into a `GemmKernel` (mainloop + epilogue + tile_scheduler)
4. Wrap in `GemmUniversalAdapter` for host launch

### CuTe Core Concepts

**Layout** = `(Shape, Stride)` -- a function from coordinate space to index space. All thread/data partitioning in CUTLASS uses Layouts.

Key Layout algebra operations:
- `composition(A, B)` -- function composition A.B (map logical to physical layout)
- `logical_divide(L, T)` -- partition Layout L by Tile T
- `zipped_divide(L, T)` -- partition and reorder tile inner/outer dims
- `logical_product(L, T)` -- replicate Tile T pattern across Layout L

**Tensor** = `Engine + Layout` -- created with `make_tensor(pointer, layout)`.

**Atom** -- hardware instruction metadata:
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
| Split-K | `gemm_splitk_parallel.h` | K >> MxN; reduces to partial GEMMs |
| Stream-K | `sm90_tile_scheduler_stream_k.hpp` | Load-balanced across SMs; best for irregular shapes |
| Persistent | `sm90_gemm_tma_warpspecialized_pingpong.hpp` | Kernel stays resident; low launch overhead |
| Warp Spec. | `sm90_mma_tma_gmma_ss_warpspecialized.hpp` | Producer warps do TMA, consumer warps do WGMMA |
| Ping-Pong | `sm90_gemm_tma_warpspecialized_pingpong.hpp` | Two warpgroups alternate; maximizes pipeline utilization |
| Cooperative | `sm90_gemm_tma_warpspecialized_cooperative.hpp` | Multiple warpgroups collaborate on same tile |

Schedule tags are defined in `dispatch_policy.hpp` (1566 lines) and control which collective specialization the builder selects. `KernelScheduleAuto` / `EpilogueScheduleAuto` let the builder pick automatically.

### EVT (Epilogue Visitor Tree) Fusion

Composable epilogue fusion at compile time. Node types in `include/cutlass/epilogue/fusion/`:
- `Load` -- load extra data from global memory (bias, scale)
- `Compute` -- element-wise ops (ReLU, GELU, Clamp, TopK, Softmax)
- `Store` -- write back results (with optional type conversion or absmax)
- `Reduce` -- reduction ops (partial reduce for Split-K)

Key files: `operations.hpp`, `sm90_visitor_*.hpp`, `sm90_callbacks_tma_warpspecialized.hpp` (109KB)

### Async Pipeline (SM90)

`include/cutlass/pipeline/sm90_pipeline.hpp` provides `PipelineTmaAsync` and `PipelineAsync` abstractions for producer-consumer coordination:
- Producer warps issue TMA loads and signal arrival barriers
- Consumer warps wait on barriers, compute (WGMMA), then release
- Pipeline depth (number of stages) trades SMEM capacity for latency hiding

## Learning Path for H20 (SM90a Focus)

### Stage 1: CuTe Fundamentals
Start with PyCuTe (`python/pycute/`) and Jupyter notebooks before touching C++:
```bash
python -c "from pycute import *; L = make_layout((2,3),(3,1)); print(L)"
jupyter notebook examples/python/CuTeDSL/notebooks/cute_layout_algebra.ipynb
```

Then C++ tutorials in order:
1. `examples/cute/tutorial/sgemm_1.cu` -- basic CuTe SGEMM
2. `examples/cute/tutorial/sgemm_2.cu` -- TiledMma SGEMM
3. `examples/cute/tutorial/tiled_copy.cu` -- TiledCopy
4. `examples/cute/tutorial/sgemm_sm80.cu` -- Ampere multistage pipeline

### Stage 2: Hopper-specific (SM90a) -- Relevant to H20
Key examples (compile with `-DCUTLASS_NVCC_ARCHS=90a`):
- `examples/48_hopper_warp_specialized_gemm/` -- Hopper Warp Specialization GEMM
- `examples/49_hopper_gemm_with_collective_builder/` -- CollectiveBuilder API (recommended starting point for 3.x kernels)
- `examples/54_hopper_fp8_warp_specialized_gemm/` -- FP8 GEMM
- `examples/55_hopper_mixed_dtype_gemm/` -- Mixed dtype GEMM (e.g., e2m1 * TF32)
- `examples/57_hopper_grouped_gemm/` -- Grouped GEMM (MoE)
- `examples/61_hopper_gemm_with_topk_and_softmax/` -- GEMM+TopK+Softmax EVT fusion
- `examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/` -- FP8 Block-wise Scaling GEMM
- `examples/88_hopper_fmha/` -- Flash MHA
- `examples/111_hopper_ssd/` -- State Space Decomposition (Mamba-like)

CuTe tutorials for Hopper:
- `examples/cute/tutorial/hopper/wgmma_sm90.cu` -- WGMMA instruction
- `examples/cute/tutorial/hopper/wgmma_tma_sm90.cu` -- WGMMA + TMA

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
| CollectiveBuilder declaration | `include/cutlass/gemm/collective/collective_builder_decl.hpp` |
| SM90 builder specialization | `include/cutlass/gemm/collective/builders/sm90_common.inl` |
| GEMM dispatch policy / schedule tags | `include/cutlass/gemm/dispatch_policy.hpp` (1566 lines) |
| SM90 async pipeline | `include/cutlass/pipeline/sm90_pipeline.hpp` |
| SM90 tile schedulers | `include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp`, `sm90_tile_scheduler_stream_k.hpp` |
| EVT operations | `include/cutlass/epilogue/fusion/operations.hpp` |
| Hopper EVT callbacks | `include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp` (109KB) |
| Epilogue builder (SM90) | `include/cutlass/epilogue/collective/builders/sm90_builder.inl` |
| GemmUniversalAdapter (host API) | `include/cutlass/gemm/device/gemm_universal_adapter.h` |

## Common Pitfalls

- **Compilation time**: Single CUTLASS example can take 2-5 minutes. Use `-j16` and build only the needed target.
- **Architecture mismatch**: Always use `90a` (not `90`) for Hopper features like WGMMA/TMA. `sm90a` means architecture-accelerated features (async warp-specialized pipeline); plain `sm90` PTX is forward-compatible but misses WGMMA.
- **Template error messages**: CUTLASS template errors are extremely verbose. Look for the first error in the chain.
- **`_` operator in CuTe**: `cute::_` is the slice/underscore, not a variable name.
- **Layout vs Stride**: CuTe Layout `(shape:stride)` notation; the stride is per-element, not per-byte.
- **CollectiveBuilder Auto schedules**: When using `KernelScheduleAuto` for the mainloop, you must also use `EpilogueScheduleAuto` for the epilogue (and vice versa) to ensure compatibility.
- **GCC 8.5.0**: Has known regressions with fold expressions and overloaded operators. Use GCC 7.5 or >= 9.
- **Windows**: CUTLASS 4.x builds are known to be broken on Windows for all CUDA toolkits.
