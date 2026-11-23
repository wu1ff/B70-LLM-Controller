import json, pathlib, subprocess, time
w=pathlib.Path(__file__).resolve().parents[1]
p=pathlib.Path('/home/wulf/Desktop/qwen3.8-flash-next-dflash2-v032/release-prep/publication-1.0.5/final-pack/patches/production')
items=json.loads((p/'manifests/native-builds.json').read_text())
r=[]
for item in items:
    family=item['family'];out=w/'build/native'/family;out.mkdir(exist_ok=True)
    common=['-std=c++20','-O3','-DNDEBUG','-D_GLIBCXX_USE_CXX11_ABI=1','-fPIC','-fvisibility=hidden','-fsycl','-fsycl-targets=spir64','-DVLLM_XPU_ENABLE_XE2','-DCUTLASS_ENABLE_HEADERS_ONLY','-DCUTLASS_ENABLE_SYCL','-DSYCL_INTEL_TARGET','-I/headers','-I/headers/csrc','-I/headers/csrc/xpu/gdn_attn','-I/headers/csrc/xpu/grouped_gemm/xe_2','-I/tla/include','-I/tla/tools/util/include','-I/tla/examples/common','-I/opt/venv/lib/python3.12/site-packages/torch/include','-I/opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include','-I/usr/include/python3.12','-I/toolchain/opt/intel/oneapi/compiler/2026.0/include/syclcompat','-include','/headers/csrc/sycl_first.h']
    link=['-shared','-fsycl','-fsycl-targets=spir64','-fsycl-max-parallel-link-jobs=2','-Xspirv-translator','-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate','-L/opt/venv/lib/python3.12/site-packages/torch/lib','-L/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels','-Wl,-rpath,$ORIGIN/../torch/lib:$ORIGIN','-Wl,-Bsymbolic','-ltorch','-ltorch_cpu','-ltorch_xpu','-lc10','-lc10_xpu']
    import shlex
    q=shlex.join
    commands=['set -euxo pipefail','cd /outputs','export PATH=/toolchain/opt/intel/oneapi/compiler/2026.0/bin:/opt/venv/bin:/usr/bin:/bin','export LD_LIBRARY_PATH=/toolchain/opt/intel/oneapi/compiler/2026.0/lib:/toolchain/opt/intel/oneapi/tbb/2023.0/lib:/opt/venv/lib:/usr/local/lib']
    objects=[]
    if family=='residency-shim':
        commands.append('gcc -std=c11 -O2 -fPIC -shared -Wall -Wextra -Werror /native/residency-shim/l0_peer_residency_shim.c -o libl0_peer_residency_shim.so -lze_loader -ldl -pthread')
    else:
        for tu in item['translation_units']:
            src='/'+tu;obj=pathlib.Path(tu).stem+'.o';objects.append(obj)
            commands.append(q(['icpx',*common,'-MMD','-MF',obj+'.d','-c',src,'-o',obj]))
        if family=='native':link.append('-lgdn_attn_kernels_xe_2')
        commands.append(q(['icpx',*objects,*link,'-o',pathlib.Path(item['installed_library']).name]))
    (out/'build.sh').write_text('\n'.join(commands)+'\n')
    header=item['header_tree'] or 'common-native-headers'
    command=['docker','run','--pull','never','--name','dflash033-build-'+family,'--network','none','--ipc','private','--cpus','8','--entrypoint','/bin/bash','-v',str(w/'source/native')+':/native:ro','-v',str(p/'dependencies'/header)+':/headers:ro','-v','/home/wulf/Desktop/Qfn-Mtp-vllm33/build/phase6/toolchain:/toolchain:ro','-v','/home/wulf/Desktop/Qfn-Mtp-vllm33/build/phase6/sycl-tla:/tla:ro','-v',str(out)+':/outputs:rw','sha256:010bdc441c72ee222f459d3b688bb41317525f1bd7223afe6218490b0c526832','/outputs/build.sh']
    start=time.monotonic()
    with (out/'build.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    entry={'family':family,'exit':result.returncode,'seconds':time.monotonic()-start,'library':item['installed_library']}
    r.append(entry);(w/'evidence/native-builds.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(entry),flush=True)
    if result.returncode:
        print((out/'build.log').read_text()[-3500:],flush=True)
