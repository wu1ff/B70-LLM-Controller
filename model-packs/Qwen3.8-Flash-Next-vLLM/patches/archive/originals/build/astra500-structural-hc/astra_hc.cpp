// SPDX-License-Identifier: Apache-2.0
#include <torch/all.h>
#include "csrc/utils.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "gemm_xe2_policy.hpp"
#include "fused_hc.hpp"
namespace AstraHC {
class FusedHC;
void up(at::Tensor a, at::Tensor w, at::Tensor x, at::Tensor out) {
  TORCH_CHECK(a.scalar_type()==at::kBFloat16 && w.scalar_type()==at::kBFloat16 &&
      x.scalar_type()==at::kBFloat16 && out.scalar_type()==at::kBFloat16,"BF16 required");
  TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && x.is_contiguous() && out.is_contiguous(),"contiguous required");
  const int m=a.size(0);
  TORCH_CHECK(a.dim()==2 && a.size(1)==320 && w.sizes()==at::IntArrayRef({10240,320}) &&
      x.sizes()==at::IntArrayRef({m,10240}) && out.sizes()==at::IntArrayRef({m,2560}),"shape mismatch");
  using T=cutlass::bfloat16_t;
  using P=MoE::w16a16_policy_m_8;
  using MMA=typename MoE::TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8,float,T>>,Layout<typename P::WGTile>,typename P::SGLayout>::TiledMMA;
  auto mma=MMA{};
  auto A=MoE::make_moe_tensor<T,'R'>(reinterpret_cast<T*>(a.data_ptr()),m,320);
  auto B=MoE::make_moe_tensor<T,'R'>(reinterpret_cast<T*>(w.data_ptr()),10240,320);
  // C supplies only coordinates; the epilogue writes the reduced [M,2560] output.
  auto C=MoE::make_moe_tensor<T,'R'>(reinterpret_cast<T*>(out.data_ptr()),m,10240);
  auto* xn=reinterpret_cast<T*>(x.data_ptr());auto* y=reinterpret_cast<T*>(out.data_ptr());
  auto& q=at::xpu::getCurrentXPUStream(a.device().index()).queue();
  namespace sx=sycl::ext::oneapi::experimental; namespace ix=sycl::ext::intel::experimental;
  sx::properties props{sx::sub_group_size<16>,ix::grf_size<256>};
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float,1> slm(sycl::range<1>(512),cgh);
    cgh.parallel_for<FusedHC>(sycl::nd_range<3>(sycl::range<3>((m+7)/8,160,64),sycl::range<3>(1,1,64)),props,
      [=](sycl::nd_item<3> it) {
        auto coord=make_coord(int(it.get_group(0)),int(it.get_group(1)),_,0);
        auto localC=C;
        fused_hc_up<void,void,void>(A,B,static_cast<T*>(nullptr),static_cast<T*>(nullptr),localC,coord,mma,xn,y,
          slm.template get_multi_ptr<sycl::access::decorated::no>().get(),m);
      });
  });
}
}
TORCH_LIBRARY(astra_hc,m){m.def("up(Tensor a, Tensor w, Tensor x, Tensor(a!) out) -> ()");}
TORCH_LIBRARY_IMPL(astra_hc,XPU,m){m.impl("up",&AstraHC::up);}
