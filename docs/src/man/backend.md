# Selecting the backend

JustRelax supports three backends: CPU, and CUDA and AMD GPU cards. To use the default CPU backend, simply load JustRelax:

```julia
using JustRelax
```

The GPU backends are implemented as extensions, so it is enough to load the appropriate GPU Pkg before loading JustRelax. That is, to use CUDA cards:
```julia
using CUDA, JustRelax
```
and for AMD cards:
```julia
using AMDGPU, JustRelax
```

Two and three dimensional solvers are implemented in different submodules, which also need to be loaded. To use the two-dimensional backend:

```julia
using JustRelax.JustRelax2D
```

and for the three-dimensional backend:

```julia
using JustRelax.JustRelax3D
```