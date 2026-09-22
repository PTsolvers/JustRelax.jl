# Boundary conditions

Boundary condition types and the kernels that apply them to velocity,
displacement, and temperature fields. See [Flow boundary conditions](@ref) for
a narrative guide.

```@autodocs; canonical=false
Modules = [JustRelax, JustRelax.JustRelax2D, JustRelax.JustRelax3D]
Pages = [
    "boundaryconditions/types.jl",
    "boundaryconditions/BoundaryConditions.jl",
    "boundaryconditions/free_slip.jl",
    "boundaryconditions/constant_value.jl",
    "boundaryconditions/periodic.jl",
    "boundaryconditions/free_surface.jl",
    "boundaryconditions/no_slip.jl",
    "boundaryconditions/pure_shear.jl",
]
```
