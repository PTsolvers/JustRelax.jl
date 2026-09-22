# Stokes

State containers, PT and DYREL damping coefficients, `solve!`, and the stress,
principal-stress, and stress-rotation kernels shared by the standard and
variational Stokes formulations.

```@autodocs; canonical=false
Modules = [JustRelax, JustRelax.JustRelax2D, JustRelax.JustRelax3D]
Pages = [
    "types/constructors/stokes.jl",
    "stokes/Stokes2D.jl",
    "stokes/Stokes3D.jl",
    "DYREL/types.jl",
    "DYREL/solver.jl",
    "variational_stokes/Stokes2D.jl",
    "variational_stokes/Stokes3D.jl",
    "variational_stokes/mask.jl",
    "stokes/StressKernels.jl",
    "stokes/PrincipalStresses.jl",
    "stokes/PressureKernels.jl",
    "stress_rotation/constructors.jl",
    "stress_rotation/types.jl",
    "stress_rotation/stress_rotation_particles.jl",
    "types/displacement.jl",
]
```
