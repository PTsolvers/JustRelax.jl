---
paths:
  - src/**/*.jl
  - ext/**/*.jl
---

# Backend Rules

Solver code is written once and compiled for CPU, CUDA and AMDGPU, in 2D and 3D.

## Module structure

- `src/JustRelax.jl` — root module: types, backend traits (`BackendTrait`, `CPUBackendTrait`, …), backend tags (`CPUBackend`, `CUDABackend`, `AMDGPUBackend`), `PTArray`.
- `src/JustRelax_CPU.jl` — the CPU `JustRelax2D` / `JustRelax3D` modules; each `__init__` calls `@init_parallel_stencil(Threads, Float64, N)`.
- `ext/JustRelaxCUDAExt.jl`, `ext/JustRelaxAMDGPUExt.jl` (weakdeps in `Project.toml`) — map `PTArray(::Type{CUDABackend}) = CuArray`, define the traits, and include `src/ext/{CUDA,AMDGPU}/{2D,3D}.jl`. Those files define a second `JustRelax2D`/`JustRelax3D` per vendor, initialize ParallelStencil for the device, and `include` the shared sources.
- `ext/JustRelaxMakieExt.jl` — plotting.

## Six module headers

The shared sources are included by six modules: CPU 2D/3D (`src/JustRelax_CPU.jl`) and CUDA/AMDGPU 2D/3D (`src/ext/…`).

- **Every module has its own copy of the `import JustRelax: …` block.** A name defined in the root module (or imported from JustPIC) and used in shared code must be added to **all six**. A missing one is an `UndefVarError` at run time on that backend only: CPU tests cannot see it, and it costs a full GPU CI round trip. After adding a name, `grep -n <name> src/JustRelax_CPU.jl src/ext/*/*.jl` must hit every file.
- A new source file must be `include`d by every module that needs it. What a module provides is what its `include` lines say — check them instead of assuming symmetry. For example `DYREL/solver_VS.jl` is included by the 2D modules only.
- Never include a 2D kernel file from a 3D module or the reverse.

## ParallelStencil initialization

`@init_parallel_stencil` runs **once per module per Julia session**. Switching backend or 2D↔3D needs a fresh process; never re-initialize in a loaded session. A script or test that defines its own `@parallel` kernels calls it itself, with the same backend token and dimension, before those kernels are parsed.

## Selecting and dispatching on a backend

- Scripts and tests pick the backend with `ENV["JULIA_JUSTRELAX_BACKEND"]` (`CPU`/`CUDA`/`AMDGPU`) and pass the tag (`JustRelax.CPUBackend`, …) to constructors: `StokesArrays(backend, ni)`.
- JustPIC has its own backend types (`JustPIC.CPU`, `CUDA.CUDABackend`, `AMDGPU.ROCBackend`). They are not JustRelax's tags — do not mix them (see the comment in `src/ext/CUDA/2D.jl`).
- Dispatch on the device through traits (`backend(x)`), never `isa CuArray`. In backend-generic code allocate through the backend (`PTArray(backend)`, `@zeros`), never a plain `Array`.
- Load CUDA/AMDGPU **before** JustRelax so the extension activates.
- Halo exchange and MPI go through ImplicitGlobalGrid. CUDA-aware MPI is enabled in CI with `IGG_CUDAAWARE_MPI=1`.

## What can be verified where

- The CPU (Threads) backend works everywhere. Check accelerators at run time (`CUDA.functional()`, `AMDGPU.functional()`); never infer from the OS.
- CI covers the rest: Buildkite CUDA (Julia 1.10 and 1) and AMDGPU jobs (`.buildkite/run_tests.yml`), and the CSCS GH200 CUDA + MPI pipeline (`ci/cscs-gh200.yml`).
- When no GPU is available, still write GPU-safe code ([kernel-rules](kernel-rules.md)) and say plainly that GPU correctness was not verified locally.
