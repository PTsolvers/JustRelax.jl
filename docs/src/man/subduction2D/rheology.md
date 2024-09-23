# Using GeoParams.jl to define the rheology of the material phases

We will use the same physical parameters as the ones defined in [Hummel et al 2024](https://doi.org/10.5194/se-15-567-2024).

The thermal expansion coefficient $\alpha$ and heat capacity $C_p$ are the same for all the material phases

```julia
α  = 2.4e-5 # 1 / K
Cp = 750    # J / kg K
```

The density of all the phases is constant, except for the oceanic lithosphere, which uses the pressure and temperature dependent equation of state $\rho = \rho_0 \left(1 - \alpha (T-T_0) - \beta (P-P_0) \right)$, where $\rho_0 = \rho (T=1475 \text{C}^{\circ})=3200 \text{kg/m}^3$.which corresponds to the `PT_Density` object from [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl):

```julia
density_lithosphere = PT_Density(; ρ0=3.2e3, α = α, β = 0e0, T0 = 273+1474)
```

We will run the case where the rheology is given by a combination of dislocation and diffusion creep for wet olivine,

```julia
using GeoParams.Dislocation
using GeoParams.Diffusion
disl_wet_olivine  = SetDislocationCreep(Dislocation.wet_olivine1_Hirth_2003)
diff_wet_olivine  = SetDiffusionCreep(Diffusion.wet_olivine_Hirth_2003)
```

and where plastic failure is given by the formulation from [Duretz et al, 2021](https://doi.org/10.1029/2021GC009675)
```julia
# non-regularized plasticity
ϕ               = asind(0.1)
C               = 1e6        # Pa
plastic_model   = DruckerPrager_regularised(; C = C, ϕ = ϕ, η_vp=η_reg, Ψ=0.0)
```

Finally we define the rheology objects of [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl)
```julia
rheology = (
    SetMaterialParams(;
        Name              = "Asthenoshpere",
        Phase             = 1,
        Density           = ConstantDensity(; ρ=3.2e3),
        HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
        Conductivity      = ConstantConductivity(; k  = 2.5),
        CompositeRheology = CompositeRheology( (LinearViscous(; η=1e20),)),
        Gravity           = ConstantGravity(; g=9.81),
    ),
    SetMaterialParams(;
        Name              = "Oceanic lithosphere",
        Phase             = 2,
        Density           = density_lithosphere,
        HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
        Conductivity      = ConstantConductivity(; k  = 2.5),
        CompositeRheology = CompositeRheology(
            (
                disl_wet_olivine,
                diff_wet_olivine,
                plastic_model,
            )
        ),
    ),
    SetMaterialParams(;
        Name              = "oceanic crust",
        Phase             = 3,
        Density           = ConstantDensity(; ρ=3.2e3),
        HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
        Conductivity      = ConstantConductivity(; k  = 2.5),
        CompositeRheology = CompositeRheology( (LinearViscous(; η=1e20),)),
    ),
    SetMaterialParams(;
        Name              = "StickyAir",
        Phase             = 4,
        Density           = ConstantDensity(; ρ=1), # water density
        HeatCapacity      = ConstantHeatCapacity(; Cp = 1e34),
        Conductivity      = ConstantConductivity(; k  = 3),
        CompositeRheology = CompositeRheology((LinearViscous(; η=1e19),)),
    ),
)
```
