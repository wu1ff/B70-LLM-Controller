// SPDX-License-Identifier: Apache-2.0
// B70 DFlash2 v0.33 experiment: exact-order skinny BF16 GEMMs.
//
// out[M,N] = BF16( ((c_0 + c_1) + c_2) + ... + c_{S-1} ), where c_s is a single
// chained-DPAS FP32 accumulation over the s-th of S equal contiguous K slices,
// starting from zero. S=1 reproduces the image's Triton _ordered_hc_gemm and
// oneDNN at K<=1536 decode widths; S=4/8 reproduce oneDNN's K=2560 decode
// GEMMs (measured: dev/onednn_order.json, dev/onednn_splitk.json).
// One work-group per (8-row, 16-column) output tile; one sub-group per slice.
#include <torch/all.h>
#include "csrc/utils.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "gemm_xe2_policy.hpp"
#include "dense_fragment.hpp"

namespace B70ExactV1 {
using T = cutlass::bfloat16_t;
using MMA = typename MoE::TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, T>>,
    Layout<Shape<_8, _16, _32>>,
    Layout<Shape<_1, _1, _1>, Stride<_1, _1, _0>>>::TiledMMA;
template<int S, int PF> class SplitChained;

template<int S, int PF>
void split_chained(const at::Tensor& x, const at::Tensor& weight, at::Tensor& output) {
  auto& q = at::xpu::getCurrentXPUStream(x.device().index()).queue();
  const int rows = x.size(0), K = x.size(1), N = weight.size(0);
  const int ntiles = (N + 15) / 16, mtiles = (rows + 7) / 8, KS = K / S;
  auto xp = reinterpret_cast<T*>(x.data_ptr());
  auto wp = reinterpret_cast<T*>(weight.data_ptr());
  auto yp = reinterpret_cast<T*>(output.data_ptr());
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> partial(S * 8 * 16, h);
    namespace sx = sycl::ext::oneapi::experimental;
    namespace ix = sycl::ext::intel::experimental;
    sx::properties props{sx::sub_group_size<16>, ix::grf_size<128>};
    h.parallel_for<SplitChained<S, PF>>(
        sycl::nd_range<3>({1, size_t(ntiles * mtiles), size_t(S * 16)},
                          {1, 1, size_t(S * 16)}), props,
        [=](sycl::nd_item<3> it) {
      const int sg = it.get_local_id(2) / 16, lane = it.get_local_id(2) % 16;
      const int nt = it.get_group(1) % ntiles, mt = it.get_group(1) / ntiles;
      const int remaining = rows - mt * 8;
      const int active = remaining < 8 ? remaining : 8;
      auto a = make_tensor(make_gmem_ptr(xp + mt * 8 * K + sg * KS),
          make_layout(make_shape(active, KS), make_stride(K, _1{})));
      auto b = make_tensor(make_gmem_ptr(wp + sg * KS),
          make_layout(make_shape(N, KS), make_stride(K, _1{})));
      auto d = MoE::make_moe_tensor<T, 'R'>(yp + mt * 8 * N, active, N);
      auto acc = dense_fragment<PF, void, void, void>(a, b,
          static_cast<T*>(nullptr), static_cast<T*>(nullptr), d,
          make_coord(0, nt, _, 0), MMA{});
      const int col = nt * 16 + lane;
      if constexpr (S == 1) {
        for (int row = 0; row < active; ++row)
          if (col < N) yp[(mt * 8 + row) * N + col] = T(acc(row));
      } else {
        for (int row = 0; row < 8; ++row)
          partial[(sg * 8 + row) * 16 + lane] = acc(row);
        it.barrier(sycl::access::fence_space::local_space);
        for (int row = sg; row < active; row += S) {
          float sum = partial[row * 16 + lane];
          for (int s = 1; s < S; ++s) sum += partial[(s * 8 + row) * 16 + lane];
          if (col < N) yp[(mt * 8 + row) * N + col] = T(sum);
        }
      }
    });
  });
}

at::Tensor forward(at::Tensor x, at::Tensor weight, int64_t split, int64_t prefetch) {
  TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && x.size(1) == weight.size(1), "geometry");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 32, "rows 1..32");
  TORCH_CHECK(x.is_xpu() && x.is_contiguous() && weight.is_contiguous() &&
              weight.device() == x.device() && x.scalar_type() == at::kBFloat16 &&
              weight.scalar_type() == at::kBFloat16, "tensor contract");
  const int64_t K = x.size(1);
  TORCH_CHECK(K % (split * 32) == 0, "K must split into 32-multiples");
  TORCH_CHECK((K * 2) % 64 == 0, "row pitch alignment");
  auto out = at::empty({x.size(0), weight.size(0)}, x.options());
#define B70_CASE(S_, PF_) if (split == S_ && prefetch == PF_) { split_chained<S_, PF_>(x, weight, out); return out; }
  B70_CASE(1, 3) B70_CASE(1, 8) B70_CASE(1, 16)
  B70_CASE(4, 3) B70_CASE(4, 8) B70_CASE(4, 16)
  B70_CASE(8, 3) B70_CASE(8, 8) B70_CASE(8, 16)
#undef B70_CASE
  TORCH_CHECK(false, "unsupported split/prefetch");
}
}  // namespace B70ExactV1

TORCH_LIBRARY(b70_exact_v1, m) {
  m.def("gemm(Tensor x, Tensor weight, int split, int prefetch) -> Tensor");
}
TORCH_LIBRARY_IMPL(b70_exact_v1, XPU, m) {
  m.impl("gemm", &B70ExactV1::forward);
}
