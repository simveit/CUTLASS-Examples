# IDE Setup

You may use clangd to get correct formatting.

As a first step install clangd extension.

Inside docker container in `~/.config/clangd/config.yaml`

```
CompileFlags:
  Compiler: /usr/local/cuda/bin/nvcc
  Add:
    - --cuda-path=/usr/local/cuda
    - --cuda-gpu-arch=sm_80
    - -I/usr/local/cuda/include
    - "-xcuda"
    # report all errors
    - "-ferror-limit=0"
    - --cuda-gpu-arch=sm_80
    - --std=c++17
    - "-D__INTELLISENSE__"
    - "-D__CLANGD__"
    - "-DCUDA_12_0_SM80_FEATURES_SUPPORTED"
    - "-DCUTLASS_ARCH_MMA_SM80_SUPPORTED=1"
    - "-D_LIBCUDACXX_STD_VER=12"
    - "-D__CUDACC_VER_MAJOR__=12"
    - "-D__CUDACC_VER_MINOR__=3"
    - "-D__CUDA_ARCH__=800"
    - "-D__CUDA_ARCH_FEAT_SM80_ALL"
    - "-Wno-invalid-constexpr"
  Remove:
    # strip CUDA fatbin args
    - "-Xfatbin*"
    # strip CUDA arch flags
    - "-gencode*"
    - "--generate-code*"
    # strip CUDA flags unknown to clang
    - "-ccbin*"
    - "--compiler-options*"
    - "--expt-extended-lambda"
    - "--expt-relaxed-constexpr"
    - "-forward-unknown-to-host-compiler"
    - "-Werror=cross-execution-space-call"
Hover:
  ShowAKA: No
InlayHints:
  Enabled: No
Diagnostics:
  Suppress:
    - "variadic_device_fn"
    - "attributes_not_allowed"
```
Adjust accordingly to your Device.

Than put a `.clangd` file into the repo and fill it with the following:

```
CompileFlags:
  Add:
    - -I/workspaces/CUTLASS-Examples/cutlass/include/
    - -I/workspaces/CUTLASS-Examples/cutlass/tools/util/include/
    - -I/workspaces/CUTLASS-Examples/cutlass/examples/common/ 
```

This will give accurate auto completion.