#include <level_zero/ze_api.h>
#include <level_zero/layers/zel_tracing_api.h>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <string>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#define CHECK(value, ...) do { if(!(value)) { fprintf(stderr,"ownership trace failed %s\n",#value);abort(); } } while(0)
static std::unordered_map<ze_kernel_handle_t,uint64_t*> destinations;
static std::mutex stamp_mutex;
static zel_tracer_handle_t tracer=nullptr;
static int64_t injected=0;
static void* loader=dlopen("libze_loader.so.1",RTLD_NOW|RTLD_LOCAL);
template<class T>T symbol(const char* name) {
  auto fn=reinterpret_cast<T>(dlsym(loader,name));CHECK(fn,name);return fn;
}
void ZE_APICALL argument(ze_kernel_set_argument_value_params_t* p,ze_result_t result,void*,void**) {
  if(result!=ZE_RESULT_SUCCESS || *p->pargIndex!=0 || *p->pargSize!=sizeof(void*) || !*p->ppArgValue)return;
  auto name_fn=symbol<decltype(&zeKernelGetName)>("zeKernelGetName");
  size_t size=0;
  if(name_fn(*p->phKernel,&size,nullptr)!=ZE_RESULT_SUCCESS)return;
  std::string name(size,'\0');name_fn(*p->phKernel,&size,name.data());
  if(name.find("OwnershipStamp4")==std::string::npos)return;
  uint64_t* address=nullptr;std::memcpy(&address,*p->ppArgValue,sizeof(address));
  std::lock_guard<std::mutex> guard(stamp_mutex);destinations[*p->phKernel]=address;
}
void ZE_APICALL launched(ze_command_list_append_launch_kernel_params_t* p,ze_result_t result,void*,void**) {
  if(result!=ZE_RESULT_SUCCESS)return;
  uint64_t* dest=nullptr;
  {std::lock_guard<std::mutex> guard(stamp_mutex);auto it=destinations.find(*p->phKernel);if(it==destinations.end())return;dest=it->second;}
  auto write=symbol<decltype(&zeCommandListAppendWriteGlobalTimestamp)>("zeCommandListAppendWriteGlobalTimestamp");
  auto barrier=symbol<decltype(&zeCommandListAppendBarrier)>("zeCommandListAppendBarrier");
  CHECK(barrier(*p->phCommandList,nullptr,0,nullptr)==ZE_RESULT_SUCCESS);
  CHECK(write(*p->phCommandList,dest,nullptr,0,nullptr)==ZE_RESULT_SUCCESS);
  CHECK(barrier(*p->phCommandList,nullptr,0,nullptr)==ZE_RESULT_SUCCESS);
  ++injected; if(injected==1)fprintf(stderr,"[ownership-clock] global timestamps active\n");
}
void enable() {
  if(tracer)return;
  zel_tracer_desc_t desc{};desc.stype=ZEL_STRUCTURE_TYPE_TRACER_DESC;
  ze_callbacks_t callbacks{};
  callbacks.Kernel.pfnSetArgumentValueCb=argument;
  callbacks.CommandList.pfnAppendLaunchKernelCb=launched;
  CHECK(symbol<decltype(&zelTracerCreate)>("zelTracerCreate")(&desc,&tracer)==ZE_RESULT_SUCCESS);
  CHECK(symbol<decltype(&zelTracerSetEpilogues)>("zelTracerSetEpilogues")(tracer,&callbacks)==ZE_RESULT_SUCCESS);
  CHECK(symbol<decltype(&zelTracerSetEnabled)>("zelTracerSetEnabled")(tracer,false)==ZE_RESULT_SUCCESS);
}

__attribute__((constructor)) static void setup(){loader=dlopen("libze_loader.so.1",RTLD_NOW|RTLD_LOCAL);setenv("ZE_ENABLE_TRACING_LAYER","1",1);CHECK(symbol<decltype(&zeInit)>("zeInit")(0)==ZE_RESULT_SUCCESS);enable();fprintf(stderr,"[ownership-clock] tracer installed\n");}

extern "C" void ownership_clock_enable(int value){static int prior=0;if(value==prior)return;CHECK(symbol<decltype(&zelTracerSetEnabled)>("zelTracerSetEnabled")(tracer,value!=0)==ZE_RESULT_SUCCESS);prior=value;}
