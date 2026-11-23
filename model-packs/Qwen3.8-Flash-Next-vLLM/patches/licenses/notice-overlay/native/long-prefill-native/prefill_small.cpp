// Modified for the B70 Flash-Next DFlash2 campaign; notice export 2026-10-22.
// DFlash2 campaign native kernel/residency integration; original upstream licensing and attribution remain below. Exact accepted inputs and build commands are in production/manifests/native-builds.json.
// Implementation attribution: Wulf integration and Astra campaign records; see licenses/NOTICE.md.
// This notice adds no license grant and does not replace upstream notices.
// SPDX-License-Identifier: Apache-2.0
// Compact real token rows before grouped expert execution; no atomic tile queue.
#include <torch/all.h>
#include "csrc/utils.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "gemm_xe2_policy.hpp"
#include "direct_gemm.hpp"
namespace LongPrefillV3 {
using T=cutlass::bfloat16_t;
constexpr int XOFF=2, GOFF=515, TILES=1156;
inline int max_tiles(int M){return (M*20+31*640)/32;}
struct UpPolicy:MoE::xe_gemm_policy_base {
 using WGTile=Shape<_32,_256,_64>;
 using SGLayout=Layout<Shape<_1,_16,_1>,Stride<_16,_1,_0>>;
};
template<int BM,int BN>struct DownPolicy:MoE::xe_gemm_policy_base {
 using WGTile=Shape<Int<BM>,Int<BN>,_64>;
 using SGLayout=Layout<Shape<_1,Int<BN/16>,_1>,Stride<Int<BN/16>,_1,_0>>;
};
template<class Id>class Routes;
class Prefix;
class Pack;
class Up;
template<int BM,int BN>class Down;
template<class Id>class GatherReference;
template<class Id>class Gather;

template<class Id>
void routes(sycl::queue& q,const Id* ids,int M,int* counts,int* inverse,int* rows) {
 q.parallel_for<Routes<Id>>(sycl::nd_range<2>({512,256},{1,256}),[=](sycl::nd_item<2> it){
  int e=it.get_group(0),lane=it.get_local_id(1),total=0;
  for(int start=0;start<M;start+=256){
   int r=start+lane,hit=0;
   if(r<M)for(int k=0;k<10;++k)hit|=int(ids[r*10+k])==e;
   int pos=sycl::exclusive_scan_over_group(it.get_group(),hit,sycl::plus<int>());
   if(r<M)inverse[e*M+r]=hit?total+pos:-1;
   if(hit)rows[e*M+total+pos]=r;
   total+=sycl::reduce_over_group(it.get_group(),hit,sycl::plus<int>());
  }
  if(lane==0)counts[e]=total;
 });
}
void prefix(sycl::queue& q,const int* counts,const int* rows,int* meta,int M,int rank,int ROFF){
 q.parallel_for<Prefix>(sycl::nd_range<1>(640,640),[=](sycl::nd_item<1> it){
  int g=it.get_local_id(0),e=(4*g+rank)/5,n=counts[e];
  int t=(n+31)/32;
  int goff=sycl::exclusive_scan_over_group(it.get_group(),n,sycl::plus<int>());
  int toff=sycl::exclusive_scan_over_group(it.get_group(),t,sycl::plus<int>());
  int xcount=g<512?counts[g]:0;
  int xoff=sycl::exclusive_scan_over_group(it.get_group(),xcount,sycl::plus<int>());
  int allx=sycl::reduce_over_group(it.get_group(),xcount,sycl::plus<int>());
  int allt=sycl::reduce_over_group(it.get_group(),t,sycl::plus<int>());
  meta[GOFF+g]=goff;
  for(int i=0;i<t;++i)meta[TILES+toff+i]=g*((M+31)/32)+i;
  if(g<512){
   meta[XOFF+g]=xoff;
   for(int i=0;i<xcount;++i)meta[ROFF+xoff+i]=rows[g*M+i];
  }
  if(g==0){meta[0]=allt;meta[1]=allx;}
 });
}
void pack(sycl::queue& q,const T* x,T* packed,const int* meta,int M,int ROFF) {
 q.parallel_for<Pack>(sycl::nd_range<1>(((M*10*2560/8+255)/256)*256,256),[=](sycl::nd_item<1> it){
  int i=it.get_global_id(0)*8,r=i/2560,c=i%2560;
  if(r<meta[1]) {
   using V=sycl::vec<uint16_t,8>;
   *reinterpret_cast<V*>(packed+i)=*reinterpret_cast<const V*>(x+meta[ROFF+r]*2560+c);
  }
 });
}
void up(sycl::queue& q,const T* x,const int8_t* w,const T* s,T* act,const int* meta,const int* counts,int rank,int M) {
 using P=UpPolicy;
 using MMA=typename MoE::TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8,float,T>>,Layout<typename P::WGTile>,typename P::SGLayout>::TiledMMA;
 auto mma=MMA{};
 namespace sx=sycl::ext::oneapi::experimental;namespace ix=sycl::ext::intel::experimental;
 sx::properties props{sx::sub_group_size<16>,ix::grf_size<256>};
 q.submit([&](sycl::handler& cgh){
  sycl::local_accessor<uint16_t,1> shared(sycl::range<1>(32*256),cgh);
  cgh.parallel_for<Up>(sycl::nd_range<3>(sycl::range<3>(1,128,256),sycl::range<3>(1,1,256)),props,[=](sycl::nd_item<3> it){
   T* values=reinterpret_cast<T*>(shared.template get_multi_ptr<sycl::access::decorated::no>().get());
   for(int task=it.get_group(1);task<meta[0];task+=128) {
    int code=meta[TILES+task],g=code/((M+31)/32),mt=code%((M+31)/32),e=(4*g+rank)/5;
    int count=counts[e],goff=meta[GOFF+g];
    auto A=MoE::make_moe_tensor<const T,'R'>(x+meta[XOFF+e]*2560,count,2560);
    auto B=MoE::make_moe_tensor<int4_t,'R'>(reinterpret_cast<int4_t*>(const_cast<int8_t*>(w)+int64_t(g)*256*1280),256,2560);
    auto D=MoE::make_moe_tensor<T,'R'>(act+goff*128,count,256);
    auto both=direct_gemm<void,void,void,MoE::B_DTYPE::INT4,128>(A,B,s+int64_t(g)*256*20,static_cast<T*>(nullptr),D,make_coord(mt,0,_,0),mma);
    int col=it.get_local_id(2);
    for(int m=0;m<32;++m)values[col*32+m]=T(both(m));
    sycl::group_barrier(it.get_group());
    if(col<128)for(int m=0;m<32;++m)if(mt*32+m<count){
      float gate=float(values[col*32+m]),value=float(values[(col+128)*32+m]);
      float silu=float(T(gate/(1.f+sycl::exp(-gate))));
      act[(goff+mt*32+m)*128+col]=T(silu*value);
    }
    sycl::group_barrier(it.get_group());
   }
  });
 });
}
template<int BM,int BN>void down(sycl::queue& q,const T* act,const int8_t* w,const T* s,T* stage,const int* meta,const int* counts,int rank,int M) {
 using P=DownPolicy<BM,BN>;
 using MMA=typename MoE::TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8,float,T>>,Layout<typename P::WGTile>,typename P::SGLayout>::TiledMMA;
 auto mma=MMA{};
 namespace sx=sycl::ext::oneapi::experimental;namespace ix=sycl::ext::intel::experimental;
 sx::properties props{sx::sub_group_size<16>,ix::grf_size<256>};
 constexpr int NCOUNT=2560/BN, SPLIT=32/BM;
 q.parallel_for<Down<BM,BN>>(sycl::nd_range<3>(sycl::range<3>(1,256,BN),sycl::range<3>(1,1,BN)),props,[=](sycl::nd_item<3> it){
  for(int task=it.get_group(1);task<meta[0]*NCOUNT*SPLIT;task+=256) {
   int code=meta[TILES+task/(NCOUNT*SPLIT)],g=code/((M+31)/32),mt=(code%((M+31)/32))*SPLIT+(task/NCOUNT)%SPLIT,nt=task%NCOUNT,e=(4*g+rank)/5;
   int count=counts[e],off=meta[GOFF+g];
   auto A=MoE::make_moe_tensor<const T,'R'>(act+off*128,count,128);
   auto B=MoE::make_moe_tensor<int4_t,'R'>(reinterpret_cast<int4_t*>(const_cast<int8_t*>(w)+int64_t(g)*2560*64),2560,128);
   auto D=MoE::make_moe_tensor<T,'R'>(stage+off*2560,count,2560);
   MoE::xe_gemm_4bits<void,void,void,MoE::B_DTYPE::INT4,128>(A,B,s+int64_t(g)*2560,static_cast<T*>(nullptr),D,make_coord(mt,nt,_,0),mma);
  }
 });
}
template<class Id>
void gather(sycl::queue& q,const T* stage,const Id* ids,const float* weights,const int* meta,const int* inverse,int rank,int M,T* out) {
 // A work item owns eight adjacent output columns. Route and scale loads
 // are shared by those columns; each column retains the exact k/j sum order.
 q.parallel_for<Gather<Id>>(sycl::nd_range<1>(M*320,64),[=](sycl::nd_item<1> it){
  int i=it.get_global_id(0)*8,r=i/2560,c=i%2560;
  float sum[8]={0,0,0,0,0,0,0,0};
  for(int k=0;k<10;++k) {
   int e=int(ids[r*10+k]);
   if(e>=0 && e<512) {
    int row=inverse[e*M+r];
    float weight=weights[r*10+k];
    for(int j=0;j<5;++j)if((5*e+j)%4==rank) {
     int g=(5*e+j)/4;
     const T* values=stage+(meta[GOFF+g]+row)*2560+c;
     #pragma unroll
     for(int v=0;v<8;++v)sum[v]+=float(values[v])*weight;
    }
   }
  }
  #pragma unroll
  for(int v=0;v<8;++v)out[i+v]=T(sum[v]);
 });
}
template<class Id>
void gather_reference(sycl::queue& q,const T* stage,const Id* ids,const float* weights,const int* meta,const int* inverse,int rank,int M,T* out) {
 q.parallel_for<GatherReference<Id>>(sycl::nd_range<1>(M*2560,256),[=](sycl::nd_item<1> it){
  int i=it.get_global_id(0),r=i/2560,c=i%2560;float sum=0;
  for(int k=0;k<10;++k) {
   int e=int(ids[r*10+k]);
   if(e>=0 && e<512) {
    int row=inverse[e*M+r];
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
             at::Tensor ids,at::Tensor weights,int64_t rank,at::Tensor output,int64_t mode) {
 int M=x.size(0);
 TORCH_CHECK(x.dim()==2 && x.size(1)==2560 && M>=1 && M<=1024 && output.sizes()==x.sizes());
 TORCH_CHECK(x.scalar_type()==at::kBFloat16 && output.scalar_type()==at::kBFloat16 && s1.scalar_type()==at::kBFloat16 && s2.scalar_type()==at::kBFloat16 && w1.scalar_type()==at::kChar && w2.scalar_type()==at::kChar && weights.scalar_type()==at::kFloat);
 TORCH_CHECK(w1.sizes()==at::IntArrayRef({640,256,1280}) && s1.sizes()==at::IntArrayRef({640,256,20}) && w2.sizes()==at::IntArrayRef({640,2560,64}) && s2.sizes()==at::IntArrayRef({640,2560,1}) && ids.sizes()==at::IntArrayRef({M,10}) && weights.sizes()==ids.sizes() && rank>=0 && rank<4);
 for(const auto& t:{x,w1,s1,w2,s2,ids,weights,output})TORCH_CHECK(t.is_contiguous() && t.device()==x.device());
 int ROFF=TILES+max_tiles(M);
 auto meta=at::empty({ROFF+M*10},x.options().dtype(at::kInt));
 auto counts=at::empty({512},x.options().dtype(at::kInt));
 auto inverse=at::empty({512,M},x.options().dtype(at::kInt));
 auto rowmap=at::empty_like(inverse);
 auto packed=at::empty({M*10,2560},x.options());
 auto act=at::empty({M*20,128},x.options());
 auto stage=at::empty({M*20,2560},x.options());
 auto& q=at::xpu::getCurrentXPUStream(x.device().index()).queue();
 if(ids.scalar_type()==at::kInt)routes(q,ids.data_ptr<int32_t>(),M,counts.data_ptr<int>(),inverse.data_ptr<int>(),rowmap.data_ptr<int>());
 else {TORCH_CHECK(ids.scalar_type()==at::kLong);routes(q,ids.data_ptr<int64_t>(),M,counts.data_ptr<int>(),inverse.data_ptr<int>(),rowmap.data_ptr<int>());}
 prefix(q,counts.data_ptr<int>(),rowmap.data_ptr<int>(),meta.data_ptr<int>(),M,rank,ROFF);
 pack(q,reinterpret_cast<T*>(x.data_ptr()),reinterpret_cast<T*>(packed.data_ptr()),meta.data_ptr<int>(),M,ROFF);
 up(q,reinterpret_cast<T*>(packed.data_ptr()),w1.data_ptr<int8_t>(),reinterpret_cast<T*>(s1.data_ptr()),reinterpret_cast<T*>(act.data_ptr()),meta.data_ptr<int>(),counts.data_ptr<int>(),rank,M);
  if(mode==0 || mode==5)down<32,64>(q,reinterpret_cast<T*>(act.data_ptr()),w2.data_ptr<int8_t>(),reinterpret_cast<T*>(s2.data_ptr()),reinterpret_cast<T*>(stage.data_ptr()),meta.data_ptr<int>(),counts.data_ptr<int>(),rank,M);
 else if(mode==1)down<16,64>(q,reinterpret_cast<T*>(act.data_ptr()),w2.data_ptr<int8_t>(),reinterpret_cast<T*>(s2.data_ptr()),reinterpret_cast<T*>(stage.data_ptr()),meta.data_ptr<int>(),counts.data_ptr<int>(),rank,M);
 else if(mode==2)down<16,128>(q,reinterpret_cast<T*>(act.data_ptr()),w2.data_ptr<int8_t>(),reinterpret_cast<T*>(s2.data_ptr()),reinterpret_cast<T*>(stage.data_ptr()),meta.data_ptr<int>(),counts.data_ptr<int>(),rank,M);
 else if(mode==3)down<32,128>(q,reinterpret_cast<T*>(act.data_ptr()),w2.data_ptr<int8_t>(),reinterpret_cast<T*>(s2.data_ptr()),reinterpret_cast<T*>(stage.data_ptr()),meta.data_ptr<int>(),counts.data_ptr<int>(),rank,M);
 else if(mode==4)down<8,64>(q,reinterpret_cast<T*>(act.data_ptr()),w2.data_ptr<int8_t>(),reinterpret_cast<T*>(s2.data_ptr()),reinterpret_cast<T*>(stage.data_ptr()),meta.data_ptr<int>(),counts.data_ptr<int>(),rank,M);
 else TORCH_CHECK(false,"Invalid long prefill mode");
 if(mode==5){ if(ids.scalar_type()==at::kInt)gather_reference(q,reinterpret_cast<T*>(stage.data_ptr()),ids.data_ptr<int32_t>(),weights.data_ptr<float>(),meta.data_ptr<int>(),inverse.data_ptr<int>(),rank,M,reinterpret_cast<T*>(output.data_ptr()));
 else gather_reference(q,reinterpret_cast<T*>(stage.data_ptr()),ids.data_ptr<int64_t>(),weights.data_ptr<float>(),meta.data_ptr<int>(),inverse.data_ptr<int>(),rank,M,reinterpret_cast<T*>(output.data_ptr()));
 }else{ if(ids.scalar_type()==at::kInt)gather(q,reinterpret_cast<T*>(stage.data_ptr()),ids.data_ptr<int32_t>(),weights.data_ptr<float>(),meta.data_ptr<int>(),inverse.data_ptr<int>(),rank,M,reinterpret_cast<T*>(output.data_ptr()));
 else gather(q,reinterpret_cast<T*>(stage.data_ptr()),ids.data_ptr<int64_t>(),weights.data_ptr<float>(),meta.data_ptr<int>(),inverse.data_ptr<int>(),rank,M,reinterpret_cast<T*>(output.data_ptr()));
 }
}
}
TORCH_LIBRARY(long_prefill_v3,m){m.def("forward(Tensor x, Tensor w1, Tensor s1, Tensor w2, Tensor s2, Tensor ids, Tensor weights, int rank, Tensor(a!) output, int mode) -> ()");}
TORCH_LIBRARY_IMPL(long_prefill_v3,XPU,m){m.impl("forward",&LongPrefillV3::forward);}
