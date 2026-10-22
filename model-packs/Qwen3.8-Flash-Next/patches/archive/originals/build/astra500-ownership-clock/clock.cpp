// Diagnostic-only graph-safe device timestamps. No weights or outputs change.
#include <torch/all.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <dlfcn.h>
#include <level_zero/layers/zel_tracing_api.h>
#include <unordered_map>
#include <mutex>
#include <cstring>

namespace {
class OwnershipStamp4;
void enable(at::Tensor output,bool value) {
 auto fn=reinterpret_cast<void(*)(int)>(dlsym(RTLD_DEFAULT,"ownership_clock_enable"));
 TORCH_CHECK(fn,"diagnostic preload missing");fn(value?1:0);
}
void stamp(at::Tensor output, int64_t index) {
  TORCH_CHECK(output.scalar_type()==at::kLong && output.is_contiguous());
  TORCH_CHECK(index>=0 && index<output.numel());
  enable(output,true);
  auto p=reinterpret_cast<uint64_t*>(output.data_ptr<int64_t>())+index;
  auto& q=c10::xpu::getCurrentXPUStream(output.device().index()).queue();
  q.single_task<OwnershipStamp4>([=]() { *static_cast<volatile uint64_t*>(p)=1; });
}
std::vector<int64_t> calibrate(at::Tensor output) {
  auto& q=c10::xpu::getCurrentXPUStream(output.device().index()).queue();
  auto device=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
  static void* lib=dlopen("libze_loader.so.1",RTLD_NOW|RTLD_LOCAL);
  TORCH_CHECK(lib,"Level Zero loader unavailable");
  auto timestamps=reinterpret_cast<decltype(&zeDeviceGetGlobalTimestamps)>(dlsym(lib,"zeDeviceGetGlobalTimestamps"));
  auto properties=reinterpret_cast<decltype(&zeDeviceGetProperties)>(dlsym(lib,"zeDeviceGetProperties"));
  TORCH_CHECK(timestamps && properties);
  ze_device_properties_t prop{}; prop.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
  TORCH_CHECK(properties(device,&prop)==ZE_RESULT_SUCCESS);
  std::vector<int64_t> values{int64_t(prop.timerResolution),int64_t(prop.timestampValidBits),int64_t(prop.kernelTimestampValidBits)};
  for(int i=0;i<16;++i) {
    uint64_t host=0,gpu=0;
    TORCH_CHECK(timestamps(device,&host,&gpu)==ZE_RESULT_SUCCESS);
    values.push_back(host); values.push_back(gpu);
  }
  return values;
}
}
TORCH_LIBRARY(astra_clock5,m) {
  m.def("stamp(Tensor(a!) output, int index) -> ()");
  m.def("calibrate(Tensor output) -> int[]");
  m.def("enable(Tensor output, bool value) -> ()");
}
TORCH_LIBRARY_IMPL(astra_clock5,XPU,m) {
  m.impl("enable",&enable);m.impl("stamp",&stamp);m.impl("calibrate",&calibrate);
}
