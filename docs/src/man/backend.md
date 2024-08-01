# Selecting the backend

[JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl) supports three backends: the default CPU backend, and two GPU backends for Nvidia and AMD GPUs. The default CPU backend is selected upon loading JustRelax:

```julia
using JustRelax
```

The GPU backends are implemented as extensions, and can be selected upon loading the appropriate GPU package before loading JustRelax. If running on Nvidia GPUs, use the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) package:
```julia
using CUDA, JustRelax
```
and if running on AMD GPUs, use the [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) package:
```julia
using AMDGPU, JustRelax
```

Two and three dimensional solvers are implemented in different submodules, which also need to be loaded. To access the two-dimensional module:
```julia
using JustRelax.JustRelax2D
```
and for the three-dimensional module:
```julia
using JustRelax.JustRelax3D
```
