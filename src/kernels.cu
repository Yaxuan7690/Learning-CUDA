#include <vector>
#include <cuda_fp16.h>
#include <cmath>

#include "../tester/utils.h"

namespace {

constexpr int kBlockSize = 256;
__host__ __device__ float to_float(float x) { return x; }
__host__ __device__ float to_float(half x) { return __half2float(x); }

template <typename T>
__host__ __device__ T from_float(float x);

template <>
__host__ __device__ float from_float<float>(float x) {
  return x;
}

template <>
__host__ __device__ half from_float<half>(float x) {
  return __float2half(x);
}

__device__ float rcp_rn(float x) {
  float y;
  asm("rcp.rn.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

template <typename T>
__global__ void rms_norm_kernel(const T* input, const T* weight, T* output,
                                size_t rows, size_t hidden_dim, float eps) {
  extern __shared__ float scratch[];
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  float local_sum = 0.0f;
  const size_t row_offset = row * hidden_dim;
  for (size_t col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    const float x = to_float(input[row_offset + col]);
    local_sum += x * x;
  }
  scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float scale =
      rsqrtf(scratch[0] / static_cast<float>(hidden_dim) + eps);
  for (size_t col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    const float x = to_float(input[row_offset + col]);
    const float w = to_float(weight[col]);
    output[row_offset + col] = from_float<T>(x * scale * w);
  }
}

template <typename T>
__global__ void attention_kernel(const T* q, const T* k, const T* v, T* o,
                                 int batch_size, int target_seq_len,
                                 int src_seq_len, int query_heads,
                                 int kv_heads, int head_dim, bool is_causal) {
  float q_local[2048];
  float out_local[2048];

  const int vec = blockIdx.x * blockDim.x + threadIdx.x;
  const int qh = vec % query_heads;
  const int t = (vec / query_heads) % target_seq_len;
  const int b = vec / (query_heads * target_seq_len);
  if (b >= batch_size) {
    return;
  }

  const int kvh = qh / (query_heads / kv_heads);
  const float scale = rcp_rn(sqrtf(static_cast<float>(head_dim)));

  for (int d = 0; d < head_dim; ++d) {
    const size_t q_idx =
        (((static_cast<size_t>(b) * target_seq_len + t) * query_heads + qh) *
         head_dim) +
        d;
    q_local[d] = to_float(q[q_idx]);
  }

  float max_score = -INFINITY;
  for (int s = 0; s < src_seq_len; ++s) {
    const bool masked = is_causal && s > t;
    if (!masked) {
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        const size_t k_idx =
            (((static_cast<size_t>(b) * src_seq_len + s) * kv_heads + kvh) *
             head_dim) +
            d;
        dot = fmaf(q_local[d], to_float(k[k_idx]), dot);
      }
      max_score = fmaxf(max_score, dot * scale);
    }
  }

  float denom = 0.0f;
  for (int s = 0; s < src_seq_len; ++s) {
    const bool masked = is_causal && s > t;
    if (!masked) {
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        const size_t k_idx =
            (((static_cast<size_t>(b) * src_seq_len + s) * kv_heads + kvh) *
             head_dim) +
            d;
        dot = fmaf(q_local[d], to_float(k[k_idx]), dot);
      }
      denom += expf(dot * scale - max_score);
    }
  }

  for (int d = 0; d < head_dim; ++d) {
    out_local[d] = 0.0f;
  }

  const float inv_denom = rcp_rn(denom);
  for (int s = 0; s < src_seq_len; ++s) {
    const bool masked = is_causal && s > t;
    if (!masked) {
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        const size_t k_idx =
            (((static_cast<size_t>(b) * src_seq_len + s) * kv_heads + kvh) *
             head_dim) +
            d;
        dot = fmaf(q_local[d], to_float(k[k_idx]), dot);
      }
      const float p = expf(dot * scale - max_score) * inv_denom;
      for (int d = 0; d < head_dim; ++d) {
      const size_t v_idx =
          (((static_cast<size_t>(b) * src_seq_len + s) * kv_heads + kvh) *
           head_dim) +
          d;
      out_local[d] = fmaf(p, to_float(v[v_idx]), out_local[d]);
      }
    }
  }

  for (int d = 0; d < head_dim; ++d) {
    const size_t o_idx =
        (((static_cast<size_t>(b) * target_seq_len + t) * query_heads + qh) *
         head_dim) +
        d;
    o[o_idx] = from_float<T>(out_local[d]);
  }
}

template <typename T>
void copy_to_device(T** dst, const std::vector<T>& src) {
  RUNTIME_CHECK(cudaMalloc(dst, src.size() * sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(*dst, src.data(), src.size() * sizeof(T),
                           cudaMemcpyHostToDevice));
}

template <typename T>
void copy_from_device(std::vector<T>& dst, const T* src) {
  RUNTIME_CHECK(cudaMemcpy(dst.data(), src, dst.size() * sizeof(T),
                           cudaMemcpyDeviceToHost));
}

}
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  h_output.resize(rows * hidden_dim);
  T *d_input = nullptr, *d_weight = nullptr, *d_output = nullptr;
  copy_to_device(&d_input, h_input);
  copy_to_device(&d_weight, h_weight);
  RUNTIME_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(T)));

  rms_norm_kernel<T><<<static_cast<unsigned>(rows), kBlockSize,
                       kBlockSize * sizeof(float)>>>(d_input, d_weight,
                                                     d_output, rows, hidden_dim,
                                                     eps);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());
  copy_from_device(h_output, d_output);

  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_weight));
  RUNTIME_CHECK(cudaFree(d_output));
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  h_o.resize(static_cast<size_t>(batch_size) * target_seq_len * query_heads *
             head_dim);
  T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  copy_to_device(&d_q, h_q);
  copy_to_device(&d_k, h_k);
  copy_to_device(&d_v, h_v);
  RUNTIME_CHECK(cudaMalloc(&d_o, h_o.size() * sizeof(T)));

  const int total = batch_size * target_seq_len * query_heads;
  const int blocks = (total + kBlockSize - 1) / kBlockSize;
  RUNTIME_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 65536));
  attention_kernel<T><<<blocks, kBlockSize>>>(
      d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len, query_heads,
      kv_heads, head_dim, is_causal);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());
  copy_from_device(h_o, d_o);

  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));
}

template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
