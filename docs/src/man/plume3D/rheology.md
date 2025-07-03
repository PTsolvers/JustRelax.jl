# Using GeoParams.jl to define the rheology of the material phases

In this example, we will define a rheology for a 3D plume model using the `GeoParams.jl` package. The model will consist of for four different rheology layers: upper crust, lower crust, lithospheric mantle, and sublithospheric mantle. We will use a non-Newtoninan visco-elasto-plastic rheology.

We will start by defining the physical parameters of the diffusion and dislocation creep laws of every phase:

```julia
using GeoParams
disl_upper_crust            = DislocationCreep(A = 5.07e-18,  n = 2.3, E = 154.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)
disl_lower_crust            = DislocationCreep(A = 2.08e-23,  n = 3.2, E = 238.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)
disl_lithospheric_mantle    = DislocationCreep(A = 2.51e-17,  n = 3.5, E = 530.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)
disl_sublithospheric_mantle = DislocationCreep(A = 2.51e-17,  n = 3.5, E = 530.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)
diff_lithospheric_mantle    = DislocationCreep(A = 2.51e-17,  n = 1.0, E = 530.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)
diff_sublithospheric_mantle = DislocationCreep(A = 2.51e-17,  n = 1.0, E = 530.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)
```

then their elastic properties
```julia
el_upper_crust            = SetConstantElasticity(; G = 25.0e9, ν = 0.5)
el_lower_crust            = SetConstantElasticity(; G = 25.0e9, ν = 0.5)
el_lithospheric_mantle    = SetConstantElasticity(; G = 67.0e9, ν = 0.5)
el_sublithospheric_mantle = SetConstantElasticity(; G = 67.0e9, ν = 0.5)
β_upper_crust             = inv(get_Kb(el_upper_crust))
β_lower_crust             = inv(get_Kb(el_lower_crust))
β_lithospheric_mantle     = inv(get_Kb(el_lithospheric_mantle))
β_sublithospheric_mantle  = inv(get_Kb(el_sublithospheric_mantle))
```

plastic properties, where we use the regularised Drucker-Prager model:
```julia
η_reg    = 1.0e16
cohesion = 3.0e6
friction = asind(0.2)
pl_crust = DruckerPrager_regularised(; C = cohesion, ϕ = friction, η_vp = η_reg, Ψ = 0.0) 
friction = asind(0.3)
pl       = DruckerPrager_regularised(; C = cohesion, ϕ = friction, η_vp = η_reg, Ψ = 0.0) # 
```

then the pressure- and temperature-dependant thermal conductivities for the crust and mantle:
```julia
K_crust = TP_Conductivity(;
    a = 0.64,
    b = 807.0e0,
    c = 0.77,
    d = 0.00004 * 1.0e-6,
)
K_mantle = TP_Conductivity(;
    a = 0.73,
    b = 1293.0e0,
    c = 0.77,
    d = 0.00004 * 1.0e-6,
)
```

and finally we can things up together in a `rheology` tuple along with other material properties:
```julia
rheology = (
    # Name              = "UpperCrust",
    SetMaterialParams(;
        Phase            = 1,
        Density          = PT_Density(; ρ0 = 2.75e3, β = β_upper_crust, T0 = 0.0, α = 3.5e-5),
        HeatCapacity     = ConstantHeatCapacity(; Cp = 7.5e2),
        Conductivity     = K_crust,
        CompositeRheology= CompositeRheology((disl_upper_crust, el_upper_crust, pl_crust)),
        Elasticity       = el_upper_crust,
        Gravity          = ConstantGravity(; g = 9.81),
    ),
    # Name              = "LowerCrust",
    SetMaterialParams(;
        Phase            = 2,
        Density          = PT_Density(; ρ0 = 3.0e3, β = β_lower_crust, T0 = 0.0, α = 3.5e-5),
        HeatCapacity     = ConstantHeatCapacity(; Cp = 7.5e2),
        Conductivity     = K_crust,
        CompositeRheology= CompositeRheology((disl_lower_crust, el_lower_crust, pl_crust)),
        Elasticity       = el_lower_crust,
    ),
    # Name              = "LithosphericMantle",
    SetMaterialParams(;
        Phase            = 3,
        Density          = PT_Density(; ρ0 = 3.3e3, β = β_lithospheric_mantle, T0 = 0.0, α = 3.0e-5),
        HeatCapacity     = ConstantHeatCapacity(; Cp = 1.25e3),
        Conductivity     = K_mantle,
        CompositeRheology= CompositeRheology((disl_lithospheric_mantle, diff_lithospheric_mantle, el_lithospheric_mantle, pl)),
        Elasticity       = el_lithospheric_mantle,
    ),
    # Name              = "Plume",
    SetMaterialParams(;
        Phase            = 4,
        Density          = PT_Density(; ρ0 = 3.3e3 - 50, β = β_sublithospheric_mantle, T0 = 0.0, α = 3.0e-5),
        HeatCapacity     = ConstantHeatCapacity(; Cp = 1.25e3),
        Conductivity     = K_mantle,
        CompositeRheology= CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el_sublithospheric_mantle)),
        Elasticity       = el_sublithospheric_mantle,
    ),
)
```
