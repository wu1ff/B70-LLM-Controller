// SPDX-License-Identifier: Apache-2.0
// Exact retained K128/K64 partials; ordered phases bound SLM to 40 KiB.
#include <torch/all.h>
#include "csrc/utils.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "gemm_xe2_policy.hpp"
#include "dense_fragment.hpp"

namespace B70OrderedHCNativeV2 {
using T = cutlass::bfloat16_t;
using MMA = typename MoE::TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, T>>,
    Layout<Shape<_8, _16, _32>>,
    Layout<Shape<_1, _1, _1>, Stride<_1, _1, _0>>>::TiledMMA;
template<int Split, int BK, int Phases> class Projection;

template<int Split, int BK, int Phases>
void project(at::Tensor x, at::Tensor weight, at::Tensor output, int valid) {
  auto& q = at::xpu::getCurrentXPUStream(x.device().index()).queue();
  int rows = x.size(0), columns = weight.size(0), tiles = columns / 16;
  auto xp = reinterpret_cast<T*>(x.data_ptr());
  auto wp = reinterpret_cast<T*>(weight.data_ptr());
  auto yp = reinterpret_cast<T*>(output.data_ptr());
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> partial(80 * 8 * 16, h);
    namespace sx = sycl::ext::oneapi::experimental;
    namespace ix = sycl::ext::intel::experimental;
    sx::properties props{sx::sub_group_size<16>, ix::grf_size<128>};
    h.parallel_for<Projection<Split, BK, Phases>>(
        sycl::nd_range<3>({1, size_t(tiles * ((rows + 7) / 8)), size_t(Split * 16)},
                          {1, 1, size_t(Split * 16)}), props,
        [=](sycl::nd_item<3> it) {
      int sg = it.get_local_id(2) / 16, lane = it.get_local_id(2) % 16;
      int nt = it.get_group(1) % tiles, mt = it.get_group(1) / tiles;
      int col = nt * 16 + lane, remaining = rows - mt * 8;
      int active = remaining < 8 ? remaining : 8;
      float sum = 0;
      for (int phase = 0; phase < Phases; ++phase) {
        for (int tile = sg; tile < 80; tile += Split) {
          int offset = (phase * 80 + tile) * BK;
          auto a = make_tensor(make_gmem_ptr(xp + mt * 8 * 10240 + offset),
              make_layout(make_shape(active, BK), make_stride(10240, _1{})));
          auto b = make_tensor(make_gmem_ptr(wp + offset),
              make_layout(make_shape(columns, BK), make_stride(10240, _1{})));
          auto d = MoE::make_moe_tensor<T, 'R'>(yp + mt * 8 * columns, active, columns);
          auto acc = dense_fragment<void, void, void>(a, b,
              static_cast<T*>(nullptr), static_cast<T*>(nullptr), d,
              make_coord(0, nt, _, 0), MMA{});
          for (int row = 0; row < 8; ++row)
            partial[(tile * 8 + row) * 16 + lane] = acc(row);
        }
        it.barrier(sycl::access::fence_space::local_space);
        if (sg < active)
          for (int tile = 0; tile < 80; ++tile)
            sum += partial[(tile * 8 + sg) * 16 + lane];
        it.barrier(sycl::access::fence_space::local_space);
      }
      if (sg < active)
        yp[(mt * 8 + sg) * columns + col] = col < valid ? T(sum) : T(0.f);
    });
  });
}

template<int Split>
void dispatch(at::Tensor x, at::Tensor weight, at::Tensor output, int valid) {
  if (x.size(0) <= 8) project<Split, 128, 1>(x, weight, output, valid);
  else project<Split, 64, 2>(x, weight, output, valid);
}

at::Tensor forward(at::Tensor x, at::Tensor weight, int64_t valid, int64_t split) {
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(0) <= 32 && x.size(1) == 10240,
              "HC input geometry");
  TORCH_CHECK(weight.dim() == 2 && weight.size(1) == 10240 &&
              (weight.size(0) == 320 || weight.size(0) == 336), "HC weight geometry");
  TORCH_CHECK(valid == (weight.size(0) == 336 ? 324 : 320), "HC valid columns");
  TORCH_CHECK(x.is_xpu() && x.is_contiguous() && weight.is_contiguous() &&
              weight.device() == x.device() && x.scalar_type() == at::kBFloat16 &&
              weight.scalar_type() == at::kBFloat16, "HC tensor contract");
  auto output = at::empty({x.size(0), weight.size(0)}, x.options());
  switch (split) {
    case 10: dispatch<10>(x, weight, output, valid); break;
    case 20: dispatch<20>(x, weight, output, valid); break;
    case 40: dispatch<40>(x, weight, output, valid); break;
    default: TORCH_CHECK(false, "HC split");
  }
  return output;
}
}
TORCH_LIBRARY(b70_ordered_hc_v2, m) {
  m.def("forward(Tensor x, Tensor weight, int valid, int split) -> Tensor");
}
TORCH_LIBRARY_IMPL(b70_ordered_hc_v2, XPU, m) {
  m.impl("forward", &B70OrderedHCNativeV2::forward);
}
