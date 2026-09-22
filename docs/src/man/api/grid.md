# Grid

The staggered-grid `Geometry` and its constructors, and the interpolation
kernels between cell-center, vertex, and staggered velocity locations. See
[Grid generation](@ref) for a narrative guide.

```@autodocs; canonical=false
Modules = [JustRelax, JustRelax.JustRelax2D, JustRelax.JustRelax3D]
Pages = [
    "grid/Grid.jl",
    "grid/Utils.jl",
    "grid/Cartesian.jl",
    "grid/Annulus.jl",
    "Interpolations.jl",
]
```
