# Thermal

Heat-diffusion state, the pseudo-transient solver, shear heating, WENO5
advection, and the subgrid diffusion time scale.

```@autodocs; canonical=false
Modules = [JustRelax.JustRelax2D, JustRelax.JustRelax3D]
Pages = [
    "types/constructors/heat_diffusion.jl",
    "thermal_diffusion/DiffusionPT.jl",
    "thermal_diffusion/DiffusionPT_GeoParams.jl",
    "thermal_diffusion/DiffusionPT_kernels.jl",
    "thermal_diffusion/DiffusionPT_coefficients.jl",
    "thermal_diffusion/ShearHeating.jl",
    "thermal_diffusion/DiffusionPT_solver.jl",
    "types/constructors/weno.jl",
    "advection/weno5.jl",
    "particles/subgrid_diffusion.jl",
]
```
