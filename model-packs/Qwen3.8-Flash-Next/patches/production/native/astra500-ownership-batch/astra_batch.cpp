// SPDX-License-Identifier: Apache-2.0
// Compact real token rows before grouped expert execution; no atomic tile queue.
#include <torch/all.h>
#include "csrc/utils.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "gemm_xe2_policy.hpp"
#include "direct_gemm.hpp"
namespace AstraBatch {
using T=cutlass::bfloat16_t;
constexpr int TILES=2, GOFF=642, XOFF=1282, MASK=1794, ROWS=2306;
struct UpPolicy:MoE::xe_gemm_policy_base {
 using WGTile=Shape<_8,_16,_64>;
 using SGLayout=Layout<Shape<_1,_1,_1>,Stride<_1,_1,_0>>;
};
struct DownPolicy:MoE::xe_gemm_policy_base {
 using WGTile=Shape<_8,_64,_64>;
 using SGLayout=Layout<Shape<_1,_4,_1>,Stride<_4,_1,_0>>;
};
template<class Id>class Routes;
class Pack;
class Up;
class Down;
template<class Id>class Gather;

template<class Id>
void routes(sycl::queue& q,const Id* ids,int rank,int M,int* meta) {
 q.parallel_for<Routes<Id>>(sycl::nd_range<1>(512,512),[=](sycl::nd_item<1> it){
  int e=it.get_local_id(0);uint32_t mask=0;
  for(int i=0;i<M*10;++i)if(int(ids[i])==e)mask|=uint32_t(1)<<(i/10);
  int count=sycl::popcount(mask),first=(rank-e%4+4)%4,groups=1+(first==0);
  int tiles=(count+7)/8;
  int xoff=sycl::exclusive_scan_over_group(it.get_group(),count,sycl::plus<int>());
  int goff=sycl::exclusive_scan_over_group(it.get_group(),count*groups,sycl::plus<int>());
  int toff=sycl::exclusive_scan_over_group(it.get_group(),tiles*groups,sycl::plus<int>());
  int all_rows=sycl::reduce_over_group(it.get_group(),count,sycl::plus<int>());
  int all_tiles=sycl::reduce_over_group(it.get_group(),tiles*groups,sycl::plus<int>());
  meta[XOFF+e]=xoff;meta[MASK+e]=int(mask);
  int j=0;
  for(int r=0;r<M;++r)if(mask&(uint32_t(1)<<r))meta[ROWS+xoff+j++]=r;
  for(int g=0;g<groups;++g){
   int slot=(5*e+first+4*g)/4;
   meta[GOFF+slot]=goff+g*count;
   for(int t=0;t<tiles;++t)meta[TILES+toff+g*tiles+t]=slot*4+t;
  }
  if(e==0){meta[0]=all_tiles;meta[1]=all_rows;}
 });
}
void pack(sycl::queue& q,const T* x,T* packed,const int* meta,int M) {
 q.parallel_for<Pack>(sycl::nd_range<1>(M*10*2560/8,256),[=](sycl::nd_item<1> it){
  int i=it.get_global_id(0)*8,r=i/2560,c=i%2560;
  if(r<meta[1]) {
   using V=sycl::vec<uint16_t,8>;
   *reinterpret_cast<V*>(packed+i)=*reinterpret_cast<const V*>(x+meta[ROWS+r]*2560+c);
  }
 });
}
void up(sycl::queue& q,const T* x,const int8_t* w,const T* s,T* act,const int* meta,int rank) {
 using P=UpPolicy;
 using MMA=typename MoE::TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8,float,T>>,Layout<typename P::WGTile>,typename P::SGLayout>::TiledMMA;
 auto mma=MMA{};
 namespace sx=sycl::ext::oneapi::experimental;namespace ix=sycl::ext::intel::experimental;
 sx::properties props{sx::sub_group_size<16>,ix::grf_size<256>};
 q.parallel_for<Up>(sycl::nd_range<3>(sycl::range<3>(1,1024,16),sycl::range<3>(1,1,16)),props,[=](sycl::nd_item<3> it){
  for(int task=it.get_group(1);task<meta[0]*8;task+=1024) {
   int code=meta[TILES+task/8],g=code/4,mt=code%4,nt=task%8,e=(4*g+rank)/5;
   int count=sycl::popcount(uint32_t(meta[MASK+e])),goff=meta[GOFF+g];
   auto A=MoE::make_moe_tensor<const T,'R'>(x+meta[XOFF+e]*2560,count,2560);
   auto B0=MoE::make_moe_tensor<int4_t,'R'>(reinterpret_cast<int4_t*>(const_cast<int8_t*>(w)+int64_t(g)*256*1280),128,2560);
   auto B1=MoE::make_moe_tensor<int4_t,'R'>(reinterpret_cast<int4_t*>(const_cast<int8_t*>(w)+int64_t(g)*256*1280+128*1280),128,2560);
   auto D=MoE::make_moe_tensor<T,'R'>(act+goff*128,count,128);
   auto coord=make_coord(mt,nt,_,0);
   auto gate=direct_gemm<void,void,void,MoE::B_DTYPE::INT4,128>(A,B0,s+int64_t(g)*256*20,static_cast<T*>(nullptr),D,coord,mma);
   auto value=direct_gemm<void,void,void,MoE::B_DTYPE::INT4,128>(A,B1,s+int64_t(g)*256*20+128*20,static_cast<T*>(nullptr),D,coord,mma);
   for(int m=0;m<8;++m)if(mt*8+m<count) {
    float gate_bf=float(T(gate(m))),v=float(T(value(m)));
    float silu=float(T(gate_bf/(1.0f+sycl::exp(-gate_bf))));
    act[(goff+mt*8+m)*128+nt*16+it.get_local_id(2)]=T(silu*v);
   }
  }
 });
}
void down(sycl::queue& q,const T* act,const int8_t* w,const T* s,T* stage,const int* meta,int rank) {
 using P=DownPolicy;
 using MMA=typename MoE::TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8,float,T>>,Layout<typename P::WGTile>,typename P::SGLayout>::TiledMMA;
 auto mma=MMA{};
 namespace sx=sycl::ext::oneapi::experimental;namespace ix=sycl::ext::intel::experimental;
 sx::properties props{sx::sub_group_size<16>,ix::grf_size<256>};
 q.parallel_for<Down>(sycl::nd_range<3>(sycl::range<3>(1,256,64),sycl::range<3>(1,1,64)),props,[=](sycl::nd_item<3> it){
  for(int task=it.get_group(1);task<meta[0]*40;task+=256) {
   int code=meta[TILES+task/40],g=code/4,mt=code%4,nt=task%40,e=(4*g+rank)/5;
   int count=sycl::popcount(uint32_t(meta[MASK+e])),off=meta[GOFF+g];
   auto A=MoE::make_moe_tensor<const T,'R'>(act+off*128,count,128);
   auto B=MoE::make_moe_tensor<int4_t,'R'>(reinterpret_cast<int4_t*>(const_cast<int8_t*>(w)+int64_t(g)*2560*64),2560,128);
   auto D=MoE::make_moe_tensor<T,'R'>(stage+off*2560,count,2560);
   MoE::xe_gemm_4bits<void,void,void,MoE::B_DTYPE::INT4,128>(A,B,s+int64_t(g)*2560,static_cast<T*>(nullptr),D,make_coord(mt,nt,_,0),mma);
  }
 });
}
template<class Id>
void gather(sycl::queue& q,const T* stage,const Id* ids,const float* weights,const int* meta,int rank,int M,T* out) {
 q.parallel_for<Gather<Id>>(sycl::nd_range<1>(M*2560,256),[=](sycl::nd_item<1> it){
  int i=it.get_global_id(0),r=i/2560,c=i%2560;float sum=0;
  for(int k=0;k<10;++k) {
   int e=int(ids[r*10+k]);
   if(e>=0 && e<512) {
    int row=sycl::popcount(uint32_t(meta[MASK+e])&((uint32_t(1)<<r)-1));
    for(int j=0;j<5;++j)if((5*e+j)%4==rank) {
     int g=(5*e+j)/4;
     sum+=float(stage[(meta[GOFF+g]+row)*2560+c])*weights[r*10+k];
    }
   }
  }
  out[i]=T(sum);
 });
}
void forward(at::Tensor x,at::Tensor w1,at::Tensor s1,at::Tensor w2,at::Tensor s2,
             at::Tensor ids,at::Tensor weights,int64_t rank,at::Tensor output) {
 int M=x.size(0);
 TORCH_CHECK(x.dim()==2 && x.size(1)==2560 && M>=16 && M<=32 && M%8==0 && output.sizes()==x.sizes());
 TORCH_CHECK(x.scalar_type()==at::kBFloat16 && output.scalar_type()==at::kBFloat16 && s1.scalar_type()==at::kBFloat16 && s2.scalar_type()==at::kBFloat16 && w1.scalar_type()==at::kChar && w2.scalar_type()==at::kChar && weights.scalar_type()==at::kFloat);
 TORCH_CHECK(w1.sizes()==at::IntArrayRef({640,256,1280}) && s1.sizes()==at::IntArrayRef({640,256,20}) && w2.sizes()==at::IntArrayRef({640,2560,64}) && s2.sizes()==at::IntArrayRef({640,2560,1}) && ids.sizes()==at::IntArrayRef({M,10}) && weights.sizes()==ids.sizes() && rank>=0 && rank<4);
 for(const auto& t:{x,w1,s1,w2,s2,ids,weights,output})TORCH_CHECK(t.is_contiguous() && t.device()==x.device());
 auto meta=at::empty({ROWS+M*10},x.options().dtype(at::kInt));
 auto packed=at::empty({M*10,2560},x.options());
 auto act=at::empty({M*20,128},x.options());
 auto stage=at::empty({M*20,2560},x.options());
 auto& q=at::xpu::getCurrentXPUStream(x.device().index()).queue();
 if(ids.scalar_type()==at::kInt)routes(q,ids.data_ptr<int32_t>(),rank,M,meta.data_ptr<int>());
 else {TORCH_CHECK(ids.scalar_type()==at::kLong);routes(q,ids.data_ptr<int64_t>(),rank,M,meta.data_ptr<int>());}
 pack(q,reinterpret_cast<T*>(x.data_ptr()),reinterpret_cast<T*>(packed.data_ptr()),meta.data_ptr<int>(),M);
 up(q,reinterpret_cast<T*>(packed.data_ptr()),w1.data_ptr<int8_t>(),reinterpret_cast<T*>(s1.data_ptr()),reinterpret_cast<T*>(act.data_ptr()),meta.data_ptr<int>(),rank);
 down(q,reinterpret_cast<T*>(act.data_ptr()),w2.data_ptr<int8_t>(),reinterpret_cast<T*>(s2.data_ptr()),reinterpret_cast<T*>(stage.data_ptr()),meta.data_ptr<int>(),rank);
 if(ids.scalar_type()==at::kInt)gather(q,reinterpret_cast<T*>(stage.data_ptr()),ids.data_ptr<int32_t>(),weights.data_ptr<float>(),meta.data_ptr<int>(),rank,M,reinterpret_cast<T*>(output.data_ptr()));
 else gather(q,reinterpret_cast<T*>(stage.data_ptr()),ids.data_ptr<int64_t>(),weights.data_ptr<float>(),meta.data_ptr<int>(),rank,M,reinterpret_cast<T*>(output.data_ptr()));
}
}
TORCH_LIBRARY(astra_batch,m){m.def("forward(Tensor x, Tensor w1, Tensor s1, Tensor w2, Tensor s2, Tensor ids, Tensor weights, int rank, Tensor(a!) output) -> ()");}
TORCH_LIBRARY_IMPL(astra_batch,XPU,m){m.impl("forward",&AstraBatch::forward);}
