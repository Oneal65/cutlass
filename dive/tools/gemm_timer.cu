/**
 * gemm_timer.cu
 *
 * 用 CUDA Event 精确计时的 CUTLASS GEMM wrapper
 * 包含: warmup runs + 多次取平均 + 理论FLOPS计算 + 算术强度
 *
 * 编译:
 *   nvcc -std=c++17 \
 *     -I${CUTLASS_ROOT}/include \
 *     -I${CUTLASS_ROOT}/tools/util/include \
 *     -arch=sm_90a -O3 \
 *     gemm_timer.cu -o gemm_timer
 *
 * 用法:
 *   ./gemm_timer <M> <N> <K>
 */

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__           \
                << " - " << cudaGetErrorString(err) << std::endl;            \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

// CUTLASS SIMT SGEMM (ColumnMajor, works on any SM)
using Gemm = cutlass::gemm::device::Gemm<
  float, cutlass::layout::ColumnMajor,
  float, cutlass::layout::ColumnMajor,
  float, cutlass::layout::ColumnMajor
>;

double benchmark_gemm(int M, int N, int K, int warmup = 3, int repeats = 10) {
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

  cutlass::Status status = gemm_op.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "GEMM cannot implement (status=" << (int)status << ")\n";
    return -1.0;
  }

  // Warmup
  for (int i = 0; i < warmup; ++i) gemm_op(args);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed runs
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeats; ++i) gemm_op(args);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return static_cast<double>(elapsed_ms) / repeats;
}

int main(int argc, char* argv[]) {
  int M = (argc > 1) ? atoi(argv[1]) : 4096;
  int N = (argc > 2) ? atoi(argv[2]) : 4096;
  int K = (argc > 3) ? atoi(argv[3]) : 4096;

  std::string sep(62, '=');
  std::cout << sep << "\n";
  std::cout << "  CUTLASS GEMM Timer  (SIMT SGEMM, ColumnMajor)\n";
  std::cout << "  Problem : M=" << M << "  N=" << N << "  K=" << K << "\n";
  std::cout << sep << "\n";

  double avg_ms = benchmark_gemm(M, N, K);
  if (avg_ms < 0) return 1;

  double flops   = 2.0 * M * N * K;
  double gflops  = flops / (avg_ms * 1e-3) / 1e9;
  double bytes   = (1.0 * M * K + 1.0 * K * N + 1.0 * M * N) * sizeof(float);
  double gbps    = bytes / (avg_ms * 1e-3) / 1e9;
  double ai      = flops / bytes;   // Arithmetic Intensity (FLOP/byte)

  // H20 reference peaks
  double h20_fp32_simt_peak = 67.0 * 1024;  // ~67 TFLOPS tensor core; SIMT << this
  double h20_hbm_bw         = 4096.0;        // GB/s

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  Avg Time         : " << avg_ms  << " ms\n";
  std::cout << "  GFLOP/s          : " << gflops  << "  GFLOPS\n";
  std::cout << "  Arith Intensity  : " << ai      << "  FLOP/byte\n";
  std::cout << "  Est. BW used     : " << gbps    << "  GB/s\n";
  std::cout << "  H20 HBM3 peak    : " << h20_hbm_bw << "  GB/s\n";
  std::cout << "  Roofline bound   : "
            << (ai > h20_fp32_simt_peak / h20_hbm_bw ? "Compute" : "Memory")
            << "-bound (crossover AI = "
            << std::setprecision(1) << (h20_fp32_simt_peak / h20_hbm_bw) << ")\n";
  std::cout << sep << "\n";
  return 0;
}
