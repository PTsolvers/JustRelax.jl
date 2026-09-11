# Material physics

[JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl) delegates every
material-property computation to
[GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl). A model
describes its materials once, as a tuple of `MaterialParams`, and passes that
tuple — conventionally called `rheology` — to the solvers, which evaluate the
properties they need on the fly.

## The rheology tuple

One `SetMaterialParams` entry per phase, with `Phase` matching the phase index
carried by the particles:

```julia
rheology = (
    SetMaterialParams(;
        Phase             = 1,
        Density           = ConstantDensity(; ρ = 2700),
        HeatCapacity      = ConstantHeatCapacity(; Cp = 1050.0),
        Conductivity      = ConstantConductivity(; k = 3.0),
        ShearHeat         = ConstantShearheating(1.0NoUnits),
        CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e22),)),
        Gravity           = ConstantGravity(; g = 9.81),
    ),
    SetMaterialParams(;
        Phase             = 2,
        Density           = ConstantDensity(; ρ = 3300),
        HeatCapacity      = ConstantHeatCapacity(; Cp = 1050.0),
        Conductivity      = ConstantConductivity(; k = 3.0),
        CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e21),)),
    ),
)
```

Each field accepts any GeoParams law of the corresponding kind, from constant
values to pressure- and temperature-dependent laws; see
[Constitutive equations](@ref) for the deformation laws in
`CompositeRheology`, and the
[GeoParams documentation](https://juliageodynamics.github.io/GeoParams.jl/dev/)
for the full catalogue.

## Evaluating properties

Laws that depend on the model state are evaluated from an `args` named tuple
holding the fields they read — temperature, pressure, and the time step:

```julia
args = (; T = thermal.T, P = stokes.P, dt = dt)
```

The routines that consume `rheology` and `args` are:

| Routine | Property read | Written to |
|---------|---------------|------------|
| `compute_viscosity!` | `CompositeRheology` | `stokes.viscosity.η`, `η_vep` |
| `compute_ρg!` | `Density`, `Gravity` | `ρg` |
| `heatdiffusion_PT!`, `PTThermalCoeffs` | `Conductivity`, `HeatCapacity`, `Density`, `RadioactiveHeat` | `thermal.T`, PT coefficients |
| `compute_shear_heating!` | `ShearHeat` | `thermal.shear_heating` |
| `compute_melt_fraction!` | `Melting` | melt fraction field |

The Stokes and thermal solvers call the first three internally when they are
given `rheology`; calling them directly is useful to initialize a model or to
diagnose a state between time steps.

## Multiple phases per cell

With particles, a cell rarely holds a single phase. `PhaseRatios` stores the
volume fraction of each phase at cell centers, vertices, and velocity nodes,
and every routine above has a method taking it:

```julia
compute_ρg!(ρg, phase_ratios, rheology, args)
compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
```

The property is evaluated per phase and averaged with those fractions as
weights, so a partly filled cell gets an intermediate value rather than the
property of its dominant phase.

Models with a sticky-air layer pass the index of the air phase as `air_phase`.
That phase is then dropped from the average and the remaining fractions are
renormalized, which yields the property of the rock alone — what the
free-surface and variational solvers expect, since they weight the result by a
rock volume fraction of their own.
