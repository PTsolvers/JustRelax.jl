
function init_rheologies(;incompressible = true, linear=true)

    # Parameters

    n_reg           = 1.0e17
    cohesion        = 30.0e6
    friction        = 30.0
    friction_seed   = 5.0
    E0              = 10e9

    ## Strain softening law

    soft_C  = NonLinearSoftening(; ξ₀=ustrip(cohesion), Δ=ustrip(cohesion) / 9999)       # nonlinear softening law
    soft_ϕ   = NonLinearSoftening(; ξ₀=ustrip(friction), Δ=ustrip(friction) / 9999)

    pl      = DruckerPrager_regularised(; C = cohesion, softening_C = soft_C, softening_ϕ=soft_ϕ, ϕ=friction, η_vp=n_reg, Ψ=0.0) # regularized plasticity
    pl_seed = DruckerPrager_regularised(; C = 1e6, softening_C = soft_C, ϕ=friction_seed, η_vp=n_reg, Ψ=0.0, ) # regularized plasticity

    if incompressible == true
        el  = SetConstantElasticity(; E=E0, ν=0.5)
        β   = inv(get_Kb(el))
    else
        el  = SetConstantElasticity(; E=E0, ν=0.25)
        β   = inv(get_Kb(el))
    end

    if linear == true
        creep_crust = LinearViscous(;η=1e23)
        creep_mantle = LinearViscous(;η =1e19)
        creep_air = LinearViscous(;η = 1e18)
    else
        creep_crust = GeoParams.Dislocation.SetDislocationCreep(GeoParams.Dislocation.wet_quartzite_Ueda_2008)
        creep_mantle = DislocationCreep(A=6.52e-16, n=3.5, E=530e3, V=18e-6,  r=0.0, R=8.3145, Apparatus = Invariant)
        creep_air = LinearViscous(;η = 1e18)
    end

    # Define rheolgy struct
    rheology = (
        # Name              = "Crust",
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ=2.7e3),
            HeatCapacity      = ConstantHeatCapacity(),
            Conductivity      = ConstantConductivity(),
            CompositeRheology = CompositeRheology((creep_crust,el,pl)),
            Elasticity = el,
            Plasticity = pl,
            RadioactiveHeat   = ConstantRadioactiveHeat(),
            ShearHeat         = ConstantShearheating(1.0NoUnits),
            Gravity           = ConstantGravity(; g=9.81),
        ),  
        # Name              = "Mantle",
        SetMaterialParams(;
            Phase             = 2,
            Density           = ConstantDensity(; ρ=3.3e3),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1e3),
            Conductivity      = ConstantConductivity(; k=2.1),
            CompositeRheology = CompositeRheology((creep_mantle,el,pl)),
            Elasticity = el,
            Plasticity = pl,
            RadioactiveHeat   = ConstantRadioactiveHeat(1.3),
            ShearHeat         = ConstantShearheating(1.0NoUnits),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        
        # Name              = "Weak Inclusion",
        SetMaterialParams(;
            Phase             = 3,
            Density           = ConstantDensity(; ρ=2.7e3),
            HeatCapacity      = ConstantHeatCapacity(),
            Conductivity      = ConstantConductivity(),
            CompositeRheology = CompositeRheology((creep_crust,el,pl_seed)),
            Elasticity        = el,
            Plasticity        = pl_seed,
            RadioactiveHeat   = ConstantRadioactiveHeat(),
            ShearHeat         = ConstantShearheating(1.0NoUnits),
            Gravity           = ConstantGravity(; g=9.81),
        ),

        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase             = 4,
            Density           = ConstantDensity(; ρ=1e0),
            HeatCapacity      = ConstantHeatCapacity(; Cp=7.5e2),
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            Conductivity      = ConstantConductivity(; k=0.5),
            CompositeRheology = CompositeRheology((creep_air,el)),
        ),
    )
end

function init_phases!(phases, phase_grid, particles, xvi)
    ni = size(phases)
    @parallel (@idx ni) _init_phases!(phases, phase_grid, particles.coords, particles.index, xvi)
end

@parallel_indices (I...) function _init_phases!(phases, phase_grid, pcoords::NTuple{N, T}, index, xvi) where {N,T}

    ni = size(phases)

    for ip in cellaxes(phases)
        # quick escape
        @index(index[ip, I...]) == 0 && continue

        pᵢ = ntuple(Val(N)) do i
            @index pcoords[i][ip, I...]
        end

        d = Inf # distance to the nearest particle
        particle_phase = -1
        for offi in 0:1, offj in 0:1
            ii = I[1] + offi
            jj = I[2] + offj

            !(ii ≤ ni[1]) && continue
            !(jj ≤ ni[2]) && continue

            xvᵢ = (
                xvi[1][ii],
                xvi[2][jj],
            )
            d_ijk = √(sum((pᵢ[i] - xvᵢ[i])^2 for i in 1:N))
            if d_ijk < d
                d = d_ijk
                particle_phase = phase_grid[ii, jj]
            end
        end
        @index phases[ip, I...] = Float64(particle_phase)
    end

    return nothing
end
