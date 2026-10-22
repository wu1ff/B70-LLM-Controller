// SPDX-License-Identifier: Apache-2.0
// Full-row BF16/FP32 HC projection using the checkpoint's existing row layout.
#include <torch/all.h>
#include "csrc/utils.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "gemm_xe2_policy.hpp"
#include "hc_tile.hpp"
namespace LongHCV2 {
using T=cutlass::bfloat16_t;
template<int BM,int BN,int BK,int AM,int AN>struct Policy:MoE::xe_gemm_policy_base {
 using WGTile=Shape<Int<BM>,Int<BN>,Int<BK>>;
 using SGLayout=Layout<Shape<Int<AM>,Int<AN>,_1>,Stride<Int<AN>,_1,_0>>;
};
template<int BM,int BN,int BK,int AM,int AN>class Projection;
template<int BM,int BN,int BK=64,int AM=1,int AN=BN/16>
void launch(sycl::queue& q,const T* x,const T* w,T* y,int M,int N,int K,int valid) {
 using P=Policy<BM,BN,BK,AM,AN>;
 using MMA=typename MoE::TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8,float,T>>,Layout<typename P::WGTile>,typename P::SGLayout>::TiledMMA;
 auto mma=MMA{};
 namespace sx=sycl::ext::oneapi::experimental;namespace ix=sycl::ext::intel::experimental;
 sx::properties props{sx::sub_group_size<16>,ix::grf_size<256>};
 q.parallel_for<Projection<BM,BN,BK,AM,AN>>(sycl::nd_range<3>(sycl::range<3>((M+BM-1)/BM,(N+BN-1)/BN,16*AM*AN),sycl::range<3>(1,1,16*AM*AN)),props,[=](sycl::nd_item<3> it){
  int mt=it.get_group(0),nt=it.get_group(1),local=it.get_local_id(2),sg=local/16,lane=local%16;
  constexpr int SM=BM/AM,SN=BN/AN;
  auto A=MoE::make_moe_tensor<const T,'R'>(x,M,K);
  auto B=MoE::make_moe_tensor<const T,'R'>(w,valid,K);
  auto C=MoE::make_moe_tensor<T,'R'>(y,M,N);
  auto accum=down_tile<void,void,void>(A,B,static_cast<const T*>(nullptr),static_cast<const T*>(nullptr),C,make_coord(mt,nt,_,0),mma);
  #pragma unroll
  for(int sn=0;sn<SN/16;++sn){
   int col=nt*BN+(sg%AN)*SN+sn*16+lane;
   #pragma unroll
   for(int sm=0;sm<SM;++sm){
    int row=mt*BM+(sg/AN)*SM+sm;
    if(row<M && col<N)y[row*N+col]=col<valid?T(accum(sn*SM+sm)):T(0.f);
   }
  }
 });
}
at::Tensor forward(at::Tensor x,at::Tensor w,int64_t valid,int64_t mode) {
 TORCH_CHECK(x.dim()==2 && w.dim()==2 && x.is_contiguous() && w.is_contiguous(),"contiguous matrices required");
 TORCH_CHECK(x.scalar_type()==at::kBFloat16 && w.scalar_type()==at::kBFloat16 && x.device()==w.device(),"BF16 checkpoint projection required");
 int M=x.size(0),N=w.size(0),K=x.size(1);
 TORCH_CHECK(M>0 && M<=1024 && K==10240 && w.size(1)==K && (N==320 || N==336) && valid==(N==336?324:320),"unsupported HC shape");
 auto out=at::empty({M,N},x.options());
 auto& q=at::xpu::getCurrentXPUStream(x.device().index()).queue();
 auto xp=reinterpret_cast<const T*>(x.data_ptr());auto wp=reinterpret_cast<const T*>(w.data_ptr());auto yp=reinterpret_cast<T*>(out.data_ptr());
 switch(mode){
 case 0:launch<32,64>(q,xp,wp,yp,M,N,K,valid);break;
 case 1:launch<32,128>(q,xp,wp,yp,M,N,K,valid);break;
 case 2:launch<64,32>(q,xp,wp,yp,M,N,K,valid);break;
 case 3:launch<64,64>(q,xp,wp,yp,M,N,K,valid);break;
 case 4:launch<16,64>(q,xp,wp,yp,M,N,K,valid);break;
 case 5:launch<32,64,128>(q,xp,wp,yp,M,N,K,valid);break;
 case 6:launch<32,128,128>(q,xp,wp,yp,M,N,K,valid);break;
 case 7:launch<128,128,64,4,2>(q,xp,wp,yp,M,N,K,valid);break;
 case 8:launch<128,128,128,4,2>(q,xp,wp,yp,M,N,K,valid);break;
 case 9:launch<128,64,128,4,1>(q,xp,wp,yp,M,N,K,valid);break;
 case 10:launch<64,128,128,2,2>(q,xp,wp,yp,M,N,K,valid);break;
 default:TORCH_CHECK(false,"unsupported HC tile");
 }
 return out;
}
}
TORCH_LIBRARY(long_hc_v2,m){m.def("forward(Tensor x, Tensor w, int valid, int mode) -> Tensor");}
TORCH_LIBRARY_IMPL(long_hc_v2,XPU,m){m.impl("forward",&LongHCV2::forward);}
