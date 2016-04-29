#ifndef OHNN_COMMON_H
#define OHNN_COMMON_H

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;

inline int DIV_CEIL(const int N, const int m) {
  return (N + m - 1) / m;
}

#define DBG_OUTPUT_LVL 0

#if (DBG_OUTPUT_LVL > 0)
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif

#endif
