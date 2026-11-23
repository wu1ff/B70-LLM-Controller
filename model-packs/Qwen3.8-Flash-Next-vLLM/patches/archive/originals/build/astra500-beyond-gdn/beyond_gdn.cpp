// SPDX-License-Identifier: Apache-2.0
// BF16 projection -> convolution: consume each projected channel in registers.
#include <torch/all.h>
#include "csrc/utils.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "gemm_xe2_policy.hpp"
#include "project.hpp"
namespace BeyondGDN {
using T=cutlass::bfloat16_t;
template<int BK,int SG>struct Policy:MoE::xe_gemm_policy_base {
 using WGTile=Shape<_8,Int<16*SG>,Int<BK>>;
 using SGLayout=Layout<Shape<_1,Int<SG>,_1>,Stride<Int<SG>,_1,_0>>;
};
template<int BK,int SG>class ProjectionConv;
template<int BK,int SG>
void launch(sycl::queue& queue,const T* x,const T* w,const T* bw,const T* cw,T* cs,int64_t css,
            const int* ids,const int* acc,const int* map,T* qkv,T* ba,T* z,T* raw) {
 using P=Policy<BK,SG>;
 using MMA=typename MoE::TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8,float,T>>,Layout<typename P::WGTile>,typename P::SGLayout>::TiledMMA;
 auto mma=MMA{};
 namespace sx=sycl::ext::oneapi::experimental;namespace ix=sycl::ext::intel::experimental;
 sx::properties props{sx::sub_group_size<16>,ix::grf_size<256>};
 constexpr int NT=16*SG,QT=4096/NT,BT=(24+NT-1)/NT;
 queue.parallel_for<ProjectionConv<BK,SG>>(sycl::nd_range<3>(sycl::range<3>(1,QT+BT,16*SG),sycl::range<3>(1,1,16*SG)),props,[=](sycl::nd_item<3> it){
  int tile=it.get_group(1),lane=it.get_local_id(2),col=(tile<QT?tile:tile-QT)*NT+lane;
  bool gate=tile>=QT;
  auto A=MoE::make_moe_tensor<const T,'R'>(x,8,2560);
  auto B=MoE::make_moe_tensor<const T,'R'>(gate?bw:w,gate?24:4096,2560);
  auto C=MoE::make_moe_tensor<T,'R'>(gate?ba:qkv,8,gate?24:4096);
  auto frag=project<void,void,void>(A,B,static_cast<T*>(nullptr),static_cast<T*>(nullptr),C,make_coord(0,gate?tile-QT:tile,_,0),mma);
  T values[8];
  for(int m=0;m<8;++m)values[m]=T(frag(m));
  if(gate){if(col<24)for(int m=0;m<8;++m)ba[(col/12)*96+m*12+col%12]=values[m];return;}
  for(int m=0;m<8;++m)if(raw)raw[m*4096+col]=values[m];
  if(col>=2560){for(int m=0;m<8;++m)z[m*1536+col-2560]=values[m];return;}
  int state=ids[0],init=sycl::max(0,acc[0]-1);
  T hist[3]={T(0),T(0),T(0)},weights[4];
  for(int i=0;i<4;++i)weights[i]=cw[col*4+i];
  if(state>=0)for(int i=0;i<3;++i)hist[i]=cs[int64_t(state)*css+(init+i)*2560+col];
  for(int m=0;m<8;++m){
   T next=values[map[m]];float result=0;
   for(int i=0;i<3;++i)result+=float(hist[i])*float(weights[i]);
   result+=float(next)*float(weights[3]);
   result=result/(1.f+sycl::exp(-result));
   qkv[col<512?m*512+col:(col<1024?4096+m*512+col-512:8192+m*1536+col-1024)]=T(result);
   hist[0]=hist[1];hist[1]=hist[2];hist[2]=next;
  }
  if(state>=0){
   T h0=cs[int64_t(state)*css+(init+1)*2560+col];
   T h1=cs[int64_t(state)*css+(init+2)*2560+col];
   cs[int64_t(state)*css+col]=h0;cs[int64_t(state)*css+2560+col]=h1;
   for(int m=0;m<8;++m)cs[int64_t(state)*css+(m+2)*2560+col]=values[map[m]];
  }
 });
}
std::vector<at::Tensor> forward(at::Tensor x,at::Tensor w,at::Tensor bw,at::Tensor cw,
 at::Tensor cs,at::Tensor ids,at::Tensor acc,at::Tensor map,int64_t variant,bool debug){
 TORCH_CHECK(x.sizes()==at::IntArrayRef({8,2560}) && w.sizes()==at::IntArrayRef({4096,2560}) && bw.sizes()==at::IntArrayRef({24,2560}));
 TORCH_CHECK(cw.sizes()==at::IntArrayRef({2560,4}) && cs.size(1)>=10 && cs.size(2)==2560 && cs.stride(1)==2560 && cs.stride(2)==1);
 TORCH_CHECK(ids.sizes()==at::IntArrayRef({1,8}) && acc.numel()==1 && map.numel()==8);
 for(auto t:{x,w,bw,cw,cs})TORCH_CHECK(t.scalar_type()==at::kBFloat16 && t.device()==x.device());
 for(auto t:{x,w,bw,cw,ids,acc,map})TORCH_CHECK(t.is_contiguous());
 for(auto t:{ids,acc,map})TORCH_CHECK(t.scalar_type()==at::kInt && t.device()==x.device());
 auto qkv=at::empty({8,2560},x.options()),ba=at::empty({8,24},x.options()),z=at::empty({8,12,128},x.options());
 auto raw=at::empty({debug?8:0,4096},x.options());
 auto& queue=at::xpu::getCurrentXPUStream(x.device().index()).queue();
 auto f=[&]<int BK,int SG>(){launch<BK,SG>(queue,(T*)x.data_ptr(),(T*)w.data_ptr(),(T*)bw.data_ptr(),(T*)cw.data_ptr(),(T*)cs.data_ptr(),cs.stride(0),ids.data_ptr<int>(),acc.data_ptr<int>(),map.data_ptr<int>(),(T*)qkv.data_ptr(),(T*)ba.data_ptr(),(T*)z.data_ptr(),debug?(T*)raw.data_ptr():nullptr);};
 if(variant==0)f.template operator()<32,4>();
 else if(variant==1)f.template operator()<64,1>();
 else if(variant==2)f.template operator()<64,4>();
 else {TORCH_CHECK(variant==3);f.template operator()<128,1>();}
 return {qkv,ba,z,raw};
}
}
TORCH_LIBRARY(astra_beyond_gdn,m){m.def("forward(Tensor x, Tensor w, Tensor bw, Tensor cw, Tensor(a!) cs, Tensor ids, Tensor acc, Tensor map, int variant, bool debug) -> Tensor[]");}
TORCH_LIBRARY_IMPL(astra_beyond_gdn,XPU,m){m.impl("forward",&BeyondGDN::forward);}
