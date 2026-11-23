template <typename T, typename StateT, int k_bucket_size, bool FusedNorm>
struct PreparedRecurrent {
 public:
  static constexpr int group_size = FusedNorm ? 1024 : 256;
  static constexpr int sg_per_group = group_size / sub_group_size;
  static constexpr int v_dim_per_sg = 4;
  static constexpr int v_dim_per_group = v_dim_per_sg * sg_per_group;
  static constexpr float eps = 0.000001;

  PreparedRecurrent(
      T* core_attn_out, const T* mixed, const T* norm, float norm_eps, sycl::local_accessor<float,1> scratch,
      const float* q,
      const float* k,
      const T* v,
      const float* b,
      const float* a,
      const float* A_log,
      const T* dt_bias,
      StateT* ssm_state,
      const int ssm_state_stride_0,
      const int* query_start_loc,
      const int* token_indx,
      const int* cache_indices,
      const int cache_indices_stride_0,
      const int* num_accepted_tokens,
      const int num_spec_decodes,
      const int num_spec_tokens,
      const int num_k_heads,
      const int head_k_dim,
      const int num_v_heads,
      const int head_v_dim)
      : core_attn_out(core_attn_out), mixed(mixed), norm(norm), norm_eps(norm_eps), scratch(scratch),
        q(q),
        k(k),
        v(v),
        b(b),
        a(a),
        A_log(A_log),
        dt_bias(dt_bias),
        ssm_state(ssm_state),
        ssm_state_stride_0(ssm_state_stride_0),
        query_start_loc(query_start_loc),
        token_indx(token_indx),
        cache_indices(cache_indices),
        cache_indices_stride_0(cache_indices_stride_0),
        num_accepted_tokens(num_accepted_tokens),
        num_spec_decodes(num_spec_decodes),
        num_spec_tokens(num_spec_tokens),
        num_k_heads(num_k_heads),
        head_k_dim(head_k_dim),
        num_v_heads(num_v_heads),
        head_v_dim(head_v_dim) {}

  static inline sycl::nd_range<3> get_nd_range(
      const int num_spec_decodes, const int num_v_heads, const int head_v_dim) {
    int num_v_bucket = (head_v_dim + v_dim_per_group - 1) / v_dim_per_group;
    sycl::range<3> local(1, 1, group_size);
    sycl::range<3> global(num_spec_decodes, num_v_heads, num_v_bucket);
    return sycl::nd_range<3>(global * local, local);
  }

  static inline float act_sigmoid(float& x) {
    return 1.0f / (1.0f + sycl::exp(-x));
  }

  static inline float
  act_softplus(float& x, float beta = 1.0f, float threshold = 20.0f) {
    if (beta * x < threshold) {
      return sycl::log(1.0f + sycl::exp(beta * x)) / beta;
    } else
      return x;
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<3> item) const {
    int batch_id = item.get_group(0);
    int num_v_heads_id = item.get_group(1);
    int v_bucket_id = item.get_group(2);

    auto sg = item.get_sub_group();
    int sg_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();

    int kv_ratio = num_v_heads / num_k_heads;
    int head_v_dim_id = v_bucket_id * v_dim_per_group + sg_id * v_dim_per_sg;
    if (head_v_dim_id >= head_v_dim) {
      return;
    }

    const float scale = 1.0f / sycl::sqrt(float(head_k_dim));
    float A_log_local = A_log[num_v_heads_id];
    float dt_bias_local = dt_bias[num_v_heads_id];
    A_log_local = -sycl::exp(A_log_local);

    float state_local[v_dim_per_sg * k_bucket_size];
    float q_local[k_bucket_size];
    float k_local[k_bucket_size];
    float v_local[v_dim_per_sg];

    // -- Load initial state ---------------------------------------------------
    // init_col = num_accepted_tokens[batch_id] - 1 (clamped to >= 0).
    int init_col = num_accepted_tokens[batch_id] - 1;
    if (init_col < 0) init_col = 0;
    const int init_state_idx =
        cache_indices[batch_id * cache_indices_stride_0 + init_col];
    const StateT* init_state_ptr =
        ssm_state + static_cast<int64_t>(init_state_idx) * ssm_state_stride_0;
#pragma unroll
    for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
        state_local[j * k_bucket_size + i] =
            static_cast<float>(init_state_ptr
                                   [num_v_heads_id * head_k_dim * head_v_dim +
                                    (k_bucket_size * sg_local_id + i) +
                                    (head_v_dim_id + j) * head_k_dim]);
      }
    }

    // query_start_loc describes actual compact tokens; num_spec_tokens only
    // describes the fixed cache-index row width used for rollback state.
    const int token_start = query_start_loc[batch_id];
    const int token_end = query_start_loc[batch_id + 1];
    for (int t = token_start, t_local = 0; t < token_end; ++t, ++t_local) {
      float b_local = b[t * num_v_heads + num_v_heads_id];
      float beta = b_local;
      float g = a[t * num_v_heads + num_v_heads_id];

      float q_sum = 0.0f;
      float k_sum = 0.0f;
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
        q_local[i] =
            q[t * num_k_heads * head_k_dim +
              (num_v_heads_id / kv_ratio) * head_k_dim +
              (k_bucket_size * sg_local_id + i)];
        k_local[i] =
            k[t * num_k_heads * head_k_dim +
              (num_v_heads_id / kv_ratio) * head_k_dim +
              (k_bucket_size * sg_local_id + i)];
        q_sum += q_local[i] * q_local[i];
        k_sum += k_local[i] * k_local[i];
      }
      float kv_mem[v_dim_per_sg];
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        kv_mem[i] = 0.0f;
      }
#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < k_bucket_size; ++i) {
          state_local[j * k_bucket_size + i] *= g;
          kv_mem[j] += state_local[j * k_bucket_size + i] * k_local[i];
        }
      }
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        kv_mem[i] = sycl::reduce_over_group(sg, kv_mem[i], sycl::plus<>());
      }

#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        v_local[i] =
            v[t * num_v_heads * head_v_dim + num_v_heads_id * head_v_dim +
              head_v_dim_id + i];
      }
      float delta[v_dim_per_sg];
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        delta[i] = (v_local[i] - kv_mem[i]) * beta;
      }

      float res[v_dim_per_sg];
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        res[i] = 0.0f;
      }
#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < k_bucket_size; ++i) {
          state_local[j * k_bucket_size + i] += k_local[i] * delta[j];
          res[j] += state_local[j * k_bucket_size + i] * q_local[i];
        }
      }
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        res[i] = sycl::reduce_over_group(sg, res[i], sycl::plus<>());
      }

      // Write O(t) to GLOBAL core_attn_out via token_indx remap.
      if (sg_local_id == 0) {
        const int global_t = (token_indx != nullptr) ? token_indx[t] : t;
#pragma unroll
        for (int i = 0; i < v_dim_per_sg; ++i) {
          if constexpr (FusedNorm) scratch[t_local * 128 + head_v_dim_id + i] = float(T(res[i]));
          else core_attn_out[global_t * num_v_heads * head_v_dim + num_v_heads_id * head_v_dim + head_v_dim_id + i] = res[i];
        }
      }

      // Write the new SSM state back to this token's dedicated cache slot,
      // so that the next forward can pick the right rollback column based on
      // num_accepted_tokens.
      const int final_state_idx =
          cache_indices[batch_id * cache_indices_stride_0 + t_local];
      StateT* final_state_ptr =
          ssm_state +
          static_cast<int64_t>(final_state_idx) * ssm_state_stride_0;
#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < k_bucket_size; ++i) {
          final_state_ptr
              [num_v_heads_id * head_k_dim * head_v_dim +
               (k_bucket_size * sg_local_id + i) +
               (head_v_dim_id + j) * head_k_dim] =
                  static_cast<StateT>(state_local[j * k_bucket_size + i]);
        }
      }
    }
    if constexpr (FusedNorm) {
      item.barrier(sycl::access::fence_space::local_space);
      if (sg_id < token_end-token_start) {
        float vals[4]; float sum=0.f;
#pragma unroll
        for (int e=0;e<4;++e) { vals[e]=scratch[sg_id*128+sg_local_id*4+e]; sum+=vals[e]*vals[e]; }
        sum=sycl::reduce_over_group(sg,sum,sycl::plus<>());
        float inv=sycl::rsqrt(sum/128.f+norm_eps);
        int glob=token_indx[token_start+sg_id];
#pragma unroll
        for (int e=0;e<4;++e) {
          int d=sg_local_id*4+e;
          float z=float(mixed[glob*4096+2560+num_v_heads_id*128+d]);
          float gate=z/(1.f+sycl::exp(-z));
          core_attn_out[glob*1536+num_v_heads_id*128+d]=vals[e]*inv*float(norm[d])*gate;
        }
      }
    }
  }

 private:
  T* core_attn_out; const T* mixed; const T* norm; float norm_eps; sycl::local_accessor<float,1> scratch;
  const float* q;
  const float* k;
  const T* v;
  const float* b;
  const float* a;
  const float* A_log;
  const T* dt_bias;
  StateT* ssm_state;
  const int ssm_state_stride_0;
  const int* query_start_loc;
  const int* token_indx;
  const int* cache_indices;
  const int cache_indices_stride_0;
  const int* num_accepted_tokens;
  const int num_spec_decodes;
  const int num_spec_tokens;
  const int num_k_heads;
  const int head_k_dim;
  const int num_v_heads;
  const int head_v_dim;
};
