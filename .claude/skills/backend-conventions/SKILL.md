---
name: backend-conventions
description: How JustRelax.jl handles CPU/CUDA/AMDGPU backends — module structure, package extensions, ParallelStencil initialization rules. Use when adding/modifying kernels or anything device-related.
---

# Backend conventions

## Structure

- `src/JustRelax.jl` — top-level module: types, traits (`BackendTrait`, `CPUBackendTrait`, ...), `CPUBackend`/`CUDABackend`/`AMDGPUBackend` singletons, `PTArray`.
- `src/JustRelax_CPU.jl` — defines `JustRelax2D` and `JustRelax3D` submodules (the CPU implementations). Each calls `@init_parallel_stencil(Threads, Float64, N)` in its `__init__`.
- `ext/JustRelaxCUDAExt.jl`, `ext/JustRelaxAMDGPUExt.jl` — package extensions (weakdeps in Project.toml). They map `PTArray(::Type{CUDABackend}) = CuArray`, define backend traits, and include the device versions of the 2D/3D modules from `src/ext/CUDA/` and `src/ext/AMDGPU/`.
- `ext/JustRelaxMakieExt.jl` — plotting extension.

## Key rules

- **Solver code is written once** in `src/` using ParallelStencil macros; the GPU extensions re-include shared code with a different `@init_parallel_stencil`. A change to solver logic in `src/` must work on all backends — check whether the file is included from `src/ext/CUDA/2D.jl` etc. too.
- ParallelStencil's `@init_parallel_stencil` is **once per module per session**. Switching backend or dimensionality requires a fresh Julia process. Never try to re-init in the same session.
- User-facing scripts/tests select the backend via `ENV["JULIA_JUSTRELAX_BACKEND"]` (`CPU`/`CUDA`/`AMDGPU`) and pass `backend_JR` (e.g. `CPUBackend`) to constructors like `StokesArrays(backend_JR, ni)`. JustPIC has its *own* backend constant (`JustPIC.CPUBackend`) — they are different types; don't mix them.
- Dispatch on device is done via traits (`backend(::CuArray) = CUDABackendTrait()`), not via `isa CuArray` checks. Follow that pattern.
- MPI/halo exchange goes through ImplicitGlobalGrid (`IGG`); CUDA-aware MPI is enabled in CI with `IGG_CUDAAWARE_MPI=1`.

## Local environment

This machine is macOS/Apple Silicon: only the CPU (Threads) backend runs locally. GPU correctness is verified by the CSCS GH200 CUDA pipeline (`ci/cscs-gh200.yml`). Write GPU-safe code (see [kernel-style](../kernel-style/SKILL.md)) and rely on CPU tests + CI.
