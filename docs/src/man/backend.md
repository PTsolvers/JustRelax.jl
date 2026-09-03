# Selecting the backend

[JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl) supports three backends: the default CPU backend, and two GPU backends for Nvidia and AMD GPUs. The default CPU backend is selected upon loading JustRelax:

```julia
using JustRelax
```

The GPU backends are implemented as extensions, and can be selected upon loading the appropriate GPU package before loading JustRelax. If running on Nvidia or AMD GPUs, use the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) or the [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) package, respectively:

:::code-group

```julia [Nvidia GPUs]
using CUDA, JustRelax
```

```julia [AMD GPUs]
using AMDGPU, JustRelax
```

:::

Two and three dimensional solvers are implemented in different submodules, which also need to be loaded:

:::code-group

```julia [2D module]
using JustRelax.JustRelax2D
```

```julia [3D module]
using JustRelax.JustRelax3D
```

:::

## Particle backend

Particle advection is delegated to [JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl), which dispatches on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) backends. JustPIC binds no vendor tag of its own: `JustPIC.CPU` is KernelAbstractions' `CPU`, and the GPU tags come from CUDA.jl and AMDGPU.jl. Scripts therefore keep the JustRelax and the JustPIC backend in two separate constants:

:::code-group

```julia [CPU]
using JustRelax, JustPIC
const backend_JR = CPUBackend
const backend_JP = JustPIC.CPU
```

```julia [Nvidia GPUs]
using CUDA, JustRelax, JustPIC
const backend_JR = CUDABackend
const backend_JP = CUDA.CUDABackend
```

```julia [AMD GPUs]
using AMDGPU, JustRelax, JustPIC
const backend_JR = AMDGPUBackend
const backend_JP = AMDGPU.ROCBackend
```

:::

`backend_JR` goes to the JustRelax allocators (`StokesArrays`, `ThermalArrays`, ...), `backend_JP` to the JustPIC ones (`init_particles`, `PhaseRatios`, ...). On Nvidia GPUs the two are the same type: JustRelax dispatches on CUDA.jl's `CUDABackend` rather than defining a tag of its own.

## Indexing cell arrays

Particle fields and phase ratios are stored as cell arrays, whose entries are read and written with `@index`:

```julia
phase = @index pPhases[ip, i, j]
@index pPhases[ip, i, j] = 2.0
```

`JustRelax2D` and `JustRelax3D` re-export this macro from [CellArraysIndexing.jl](https://github.com/albert-de-montserrat/CellArraysIndexing.jl). KernelAbstractions exports an unrelated macro of the same name, so a script that loads it directly (`using KernelAbstractions`) has to qualify the one it means, e.g. `JustRelax2D.@index`.
