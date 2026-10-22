from pathlib import Path
base=Path(__file__).resolve().parents[1]/'astra500-ownership-batch'
out=Path(__file__).resolve().parent
s=(base/'astra_batch.cpp').read_text().replace('AstraBatch','BeyondPrefill').replace('astra_batch','astra_prefill')
s=s.replace('constexpr int TILES=2, GOFF=642, XOFF=1282, MASK=1794, ROWS=2306;', 'constexpr int XOFF=2, GOFF=515, TILES=1156;\ninline int max_tiles(int M){return (M*20+31*640)/32;}')
a=s.index('template<class Id>class Routes;');b=s.index('void up(',a)
s=s[:a]+'''template<class Id>class Routes;
class Prefix;
class Pack;
class Up;
class Down;
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
''' +s[b:]
s=s.replace('using WGTile=Shape<_8,_16,_64>','using WGTile=Shape<_32,_16,_64>').replace('using WGTile=Shape<_8,_64,_64>','using WGTile=Shape<_32,_64,_64>')
s=s.replace('T* act,const int* meta,int rank)', 'T* act,const int* meta,const int* counts,int rank,int M)')
s=s.replace('T* stage,const int* meta,int rank)', 'T* stage,const int* meta,const int* counts,int rank,int M)')
s=s.replace('g=code/4,mt=code%4','g=code/((M+31)/32),mt=code%((M+31)/32)')
s=s.replace('sycl::popcount(uint32_t(meta[MASK+e]))','counts[e]')
s=s.replace('for(int m=0;m<8;++m)if(mt*8+m<count)', 'for(int m=0;m<32;++m)if(mt*32+m<count)')
s=s.replace('(goff+mt*8+m)', '(goff+mt*32+m)')
s=s.replace('const int* meta,int rank,int M,T* out)', 'const int* meta,const int* inverse,int rank,int M,T* out)')
s=s.replace('sycl::popcount(uint32_t(meta[MASK+e])&((uint32_t(1)<<r)-1))','inverse[e*M+r]')
s=s.replace('M>=16 && M<=32 && M%8==0','M>=1 && M<=1024')
s=s.replace('auto meta=at::empty({ROWS+M*10},x.options().dtype(at::kInt));','''int ROFF=TILES+max_tiles(M);
 auto meta=at::empty({ROFF+M*10},x.options().dtype(at::kInt));
 auto counts=at::empty({512},x.options().dtype(at::kInt));
 auto inverse=at::empty({512,M},x.options().dtype(at::kInt));
 auto rowmap=at::empty_like(inverse);''')
s=s.replace('rank,M,meta.data_ptr<int>())', 'M,counts.data_ptr<int>(),inverse.data_ptr<int>(),rowmap.data_ptr<int>())')
s=s.replace(' pack(q,', ' prefix(q,counts.data_ptr<int>(),rowmap.data_ptr<int>(),meta.data_ptr<int>(),M,rank,ROFF);\n pack(q,')
s=s.replace('meta.data_ptr<int>(),M);','meta.data_ptr<int>(),M,ROFF);')
s=s.replace('meta.data_ptr<int>(),rank);','meta.data_ptr<int>(),counts.data_ptr<int>(),rank,M);')
s=s.replace('meta.data_ptr<int>(),rank,M,reinterpret_cast<T*>','meta.data_ptr<int>(),inverse.data_ptr<int>(),rank,M,reinterpret_cast<T*>')
(out/'beyond_prefill.cpp').write_text(s)
(out/'direct_gemm.hpp').write_text((base/'direct_gemm.hpp').read_text().replace('AstraBatch','BeyondPrefill'))
(out/'build.sh').write_text((base/'build.sh').read_text().replace('astra_batch','beyond_prefill'))
