// SPDX-License-Identifier: Apache-2.0
// B70 DFlash2 v0.33 experiment: exact fused MoE "glue" for decode widths (M<=32).
//
// moe_front(x, Wr[512,2560], Wgu[320,2560], We[1,2560]) -> (logits[M,512], act[M,160], egate[M,1])
//   logits/gate_up: oneDNN-equivalent K=2560 arithmetic = 4 contiguous K slices, each a
//   chained FP32 DPAS accumulation from zero, summed left to right, rounded to BF16.
//   egate: one chained accumulation over K (oneDNN N=1 arithmetic), rounded to BF16.
//   act = BF16( float(BF16(silu(g))) * float(u) ), silu(g) = g / (1 + exp(-g)) in FP32,
//   exactly the vllm silu_and_mul element formula (g,u are the BF16 gate_up halves).
// moe_back(act[M,160], Wd[2560,160], egate[M,1], routed[M,2560]) -> out[M,2560]
//   d = BF16(chained DPAS over K=160); s = BF16(1/(1+exp(-egate))) (torch.sigmoid on BF16);
//   shared = BF16(float(s)*float(d)); out = BF16(float(routed) + float(shared)).
#include <torch/all.h>
#include "csrc/utils.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "gemm_xe2_policy.hpp"
#include "dense_fragment.hpp"

namespace B70ExactMoE {
using namespace B70ExactV1;
using T = cutlass::bfloat16_t;
using MMA = typename MoE::TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, T>>,
    Layout<Shape<_8, _16, _32>>,
    Layout<Shape<_1, _1, _1>, Stride<_1, _1, _0>>>::TiledMMA;

inline T silu_bf16(T g) {
  float x = static_cast<float>(g);
  return T(x / (1.0f + sycl::exp(-x)));
}
inline T sigmoid_bf16(T g) {
  float x = static_cast<float>(g);
  return T(1.0f / (1.0f + sycl::exp(-x)));
}

class Front;
class Back;
class SiluTest;
class SigmoidTest;

// Work-groups per 8-row tile: 16 router (2 column tiles each), 10 gate/up pairs, 1 egate.
std::vector<at::Tensor> moe_front(at::Tensor x, at::Tensor wr, at::Tensor wgu, at::Tensor we) {
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 2560 && x.size(0) >= 1 && x.size(0) <= 32, "x");
  TORCH_CHECK(wr.sizes() == at::IntArrayRef({512, 2560}) && wgu.sizes() == at::IntArrayRef({320, 2560}) &&
              we.sizes() == at::IntArrayRef({1, 2560}), "weights");
  for (auto* t : {&x, &wr, &wgu, &we})
    TORCH_CHECK(t->is_xpu() && t->is_contiguous() && t->scalar_type() == at::kBFloat16, "contract");
  const int rows = x.size(0), mtiles = (rows + 7) / 8;
  auto logits = at::empty({rows, 512}, x.options());
  auto act = at::empty({rows, 160}, x.options());
  auto egate = at::empty({rows, 1}, x.options());
  auto& q = at::xpu::getCurrentXPUStream(x.device().index()).queue();
  auto xp = reinterpret_cast<T*>(x.data_ptr());
  auto rp = reinterpret_cast<T*>(wr.data_ptr());
  auto gp = reinterpret_cast<T*>(wgu.data_ptr());
  auto ep = reinterpret_cast<T*>(we.data_ptr());
  auto lp = reinterpret_cast<T*>(logits.data_ptr());
  auto ap = reinterpret_cast<T*>(act.data_ptr());
  auto gop = reinterpret_cast<T*>(egate.data_ptr());
  constexpr int K = 2560, S = 4, KS = K / S, WGS = 16 + 10 + 1;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> partial(8 * 8 * 16, h);
    namespace sx = sycl::ext::oneapi::experimental;
    namespace ix = sycl::ext::intel::experimental;
    sx::properties props{sx::sub_group_size<16>, ix::grf_size<128>};
    h.parallel_for<Front>(
        sycl::nd_range<3>({1, size_t(WGS * mtiles), 128}, {1, 1, 128}), props,
        [=](sycl::nd_item<3> it) {
      const int sg = it.get_local_id(2) / 16, lane = it.get_local_id(2) % 16;
      const int w = it.get_group(1) % WGS, mt = it.get_group(1) / WGS;
      const int remaining = rows - mt * 8;
      const int active = remaining < 8 ? remaining : 8;
      const int sel = sg / 4, slice = sg % 4;
      if (w < 26) {
        const bool router = w < 16;
        T* wp = router ? rp : gp;
        const int N = router ? 512 : 320;
        const int nt = router ? (2 * w + sel) : (sel == 0 ? (w - 16) : (10 + w - 16));
        auto a = make_tensor(make_gmem_ptr(xp + mt * 8 * K + slice * KS),
            make_layout(make_shape(active, KS), make_stride(K, _1{})));
        auto b = make_tensor(make_gmem_ptr(wp + slice * KS),
            make_layout(make_shape(N, KS), make_stride(K, _1{})));
        auto d = MoE::make_moe_tensor<T, 'R'>(lp, active, N);
        auto acc = dense_fragment<3, void, void, void>(a, b,
            static_cast<T*>(nullptr), static_cast<T*>(nullptr), d,
            make_coord(0, nt, _, 0), MMA{});
        for (int row = 0; row < 8; ++row) partial[(sg * 8 + row) * 16 + lane] = acc(row);
        it.barrier(sycl::access::fence_space::local_space);
        if (router) {
          // sg -> (tile sg/4, rows sg%4 and sg%4+4)
          for (int row = sg % 4; row < active; row += 4) {
            const int base = (sg / 4) * 4;
            float sum = partial[((base + 0) * 8 + row) * 16 + lane];
            for (int s = 1; s < S; ++s) sum += partial[((base + s) * 8 + row) * 16 + lane];
            const int col = (2 * w + sg / 4) * 16 + lane;
            lp[(mt * 8 + row) * 512 + col] = T(sum);
          }
        } else {
          const int row = sg;
          if (row < active) {
            float gs = partial[(0 * 8 + row) * 16 + lane];
            for (int s = 1; s < S; ++s) gs += partial[(s * 8 + row) * 16 + lane];
            float us = partial[(4 * 8 + row) * 16 + lane];
            for (int s = 1; s < S; ++s) us += partial[((4 + s) * 8 + row) * 16 + lane];
            const T g = T(gs), u = T(us);
            const T sl = silu_bf16(g);
            const int col = (w - 16) * 16 + lane;
            ap[(mt * 8 + row) * 160 + col] = T(static_cast<float>(sl) * static_cast<float>(u));
          }
        }
      } else {
        if (sg == 0) {
          auto a = make_tensor(make_gmem_ptr(xp + mt * 8 * K),
              make_layout(make_shape(active, K), make_stride(K, _1{})));
          auto b = make_tensor(make_gmem_ptr(ep),
              make_layout(make_shape(1, K), make_stride(K, _1{})));
          auto d = MoE::make_moe_tensor<T, 'R'>(gop, active, 1);
          auto acc = dense_fragment<3, void, void, void>(a, b,
              static_cast<T*>(nullptr), static_cast<T*>(nullptr), d,
              make_coord(0, 0, _, 0), MMA{});
          if (lane == 0)
            for (int row = 0; row < active; ++row) gop[mt * 8 + row] = T(acc(row));
        }
      }
    });
  });
  return {logits, act, egate};
}

at::Tensor moe_back(at::Tensor act, at::Tensor wd, at::Tensor egate, at::Tensor routed, bool add_routed) {
  const int rows = act.size(0);
  TORCH_CHECK(act.dim() == 2 && act.size(1) == 160 && rows >= 1 && rows <= 32, "act");
  TORCH_CHECK(wd.sizes() == at::IntArrayRef({2560, 160}), "wd");
  TORCH_CHECK(egate.numel() == rows && routed.sizes() == at::IntArrayRef({rows, 2560}), "egate/routed");
  for (auto* t : {&act, &wd, &egate, &routed})
    TORCH_CHECK(t->is_xpu() && t->is_contiguous() && t->scalar_type() == at::kBFloat16, "contract");
  auto out = at::empty({rows, 2560}, act.options());
  const int mtiles = (rows + 7) / 8;
  auto& q = at::xpu::getCurrentXPUStream(act.device().index()).queue();
  auto ap = reinterpret_cast<T*>(act.data_ptr());
  auto wp = reinterpret_cast<T*>(wd.data_ptr());
  auto ep = reinterpret_cast<T*>(egate.data_ptr());
  auto rp = reinterpret_cast<T*>(routed.data_ptr());
  auto op = reinterpret_cast<T*>(out.data_ptr());
  q.submit([&](sycl::handler& h) {
    namespace sx = sycl::ext::oneapi::experimental;
    namespace ix = sycl::ext::intel::experimental;
    sx::properties props{sx::sub_group_size<16>, ix::grf_size<128>};
    h.parallel_for<Back>(
        sycl::nd_range<3>({1, size_t(160 * mtiles), 16}, {1, 1, 16}), props,
        [=](sycl::nd_item<3> it) {
      const int lane = it.get_local_id(2);
      const int nt = it.get_group(1) % 160, mt = it.get_group(1) / 160;
      const int remaining = rows - mt * 8;
      const int active = remaining < 8 ? remaining : 8;
      auto a = make_tensor(make_gmem_ptr(ap + mt * 8 * 160),
          make_layout(make_shape(active, 160), make_stride(160, _1{})));
      auto b = make_tensor(make_gmem_ptr(wp),
          make_layout(make_shape(2560, 160), make_stride(160, _1{})));
      auto d = MoE::make_moe_tensor<T, 'R'>(op + mt * 8 * 2560, active, 2560);
      auto acc = dense_fragment<3, void, void, void>(a, b,
          static_cast<T*>(nullptr), static_cast<T*>(nullptr), d,
          make_coord(0, nt, _, 0), MMA{});
      const int col = nt * 16 + lane;
      for (int row = 0; row < active; ++row) {
        const int r = mt * 8 + row;
        const T dn = T(acc(row));
        const T s = sigmoid_bf16(ep[r]);
        const T sh = T(static_cast<float>(s) * static_cast<float>(dn));
        op[r * 2560 + col] = add_routed ? T(static_cast<float>(rp[r * 2560 + col]) + static_cast<float>(sh)) : sh;
      }
    });
  });
  return out;
}

// Exhaustive-domain element tests: out[i] = silu_bf16(g[i]) * u[i] (rounded), sigmoid_bf16(x[i]).
at::Tensor silu_mul_test(at::Tensor g, at::Tensor u) {
  auto out = at::empty_like(g);
  const int64_t n = g.numel();
  auto gp = reinterpret_cast<T*>(g.data_ptr()); auto up = reinterpret_cast<T*>(u.data_ptr());
  auto op = reinterpret_cast<T*>(out.data_ptr());
  auto& q = at::xpu::getCurrentXPUStream(g.device().index()).queue();
  q.parallel_for<SiluTest>(sycl::range<1>(n), [=](sycl::id<1> i) {
    op[i] = T(static_cast<float>(silu_bf16(gp[i])) * static_cast<float>(up[i]));
  });
  return out;
}
at::Tensor sigmoid_test(at::Tensor x) {
  auto out = at::empty_like(x);
  const int64_t n = x.numel();
  auto xp = reinterpret_cast<T*>(x.data_ptr()); auto op = reinterpret_cast<T*>(out.data_ptr());
  auto& q = at::xpu::getCurrentXPUStream(x.device().index()).queue();
  q.parallel_for<SigmoidTest>(sycl::range<1>(n), [=](sycl::id<1> i) { op[i] = sigmoid_bf16(xp[i]); });
  return out;
}
}  // namespace B70ExactMoE

TORCH_LIBRARY(b70_exact_moe_v1, m) {
  m.def("front(Tensor x, Tensor wr, Tensor wgu, Tensor we) -> Tensor[]");
  m.def("back(Tensor act, Tensor wd, Tensor egate, Tensor routed, bool add_routed) -> Tensor");
  m.def("silu_mul_test(Tensor g, Tensor u) -> Tensor");
  m.def("sigmoid_test(Tensor x) -> Tensor");
}
TORCH_LIBRARY_IMPL(b70_exact_moe_v1, XPU, m) {
  m.impl("front", &B70ExactMoE::moe_front);
  m.impl("back", &B70ExactMoE::moe_back);
  m.impl("silu_mul_test", &B70ExactMoE::silu_mul_test);
  m.impl("sigmoid_test", &B70ExactMoE::sigmoid_test);
}
