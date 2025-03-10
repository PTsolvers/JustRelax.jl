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
