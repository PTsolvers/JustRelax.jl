# from "Fingerprinting secondary mantle plumes", Cloetingh et al. 2022


function init_rheology(CharDim; is_compressible = false)
    # plasticity setup
    do_DP   = true          # do_DP=false: Von Mises, do_DP=true: Drucker-Prager (friction angle)
    η_reg   = 1.0e16Pa*s    # regularisation "viscosity" for Drucker-Prager
    Coh     = 5.0MPa       # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ       = 5.0 * do_DP  # friction angle
    G0      = 6.0e11Pa      # elastic shear modulus
    G_magma = 6.0e11Pa      # elastic shear modulus perturbation

    # soft_C = LinearSoftening((ustrip(Coh)/2, ustrip(Coh)), (0e0, 1e-1)) # softening law
    soft_C  = NonLinearSoftening(; ξ₀=ustrip(Coh), Δ=ustrip(Coh) / 2)   # softening law
    pl      = DruckerPrager_regularised(; C=Coh, ϕ=ϕ, η_vp=η_reg, Ψ=0.0, softening_C = soft_C)        # plasticity
    if is_compressible == true
        el       = SetConstantElasticity(; G=G0, ν=0.25)           # elastic spring
        el_magma = SetConstantElasticity(; G=G_magma, ν=0.25)# elastic spring
        β_rock   = 6.0e-11
        β_magma  = 6.0e-11
    else
        el       = SetConstantElasticity(; G=G0, ν=0.5)            # elastic spring
        el_magma = SetConstantElasticity(; G=G_magma, ν=0.5) # elastic spring
        β_rock   = inv(get_Kb(el))
        β_magma  = inv(get_Kb(el_magma))
    end
    creep_rock  = LinearViscous(; η=1e24 * Pa * s)
    creep_magma = LinearViscous(; η=1e18 * Pa * s)
    creep_air   = LinearViscous(; η=1e21 * Pa * s)
    g           = 9.81m/s^2

    rheology = (
        #Name="UpperCrust"
        SetMaterialParams(;
            Phase             = 1,
            Density           = PT_Density(; ρ0=2650kg / m^3, α=3e-5 / K, T0=20.0C, β=β_rock / Pa),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1050J / kg / K),
            Conductivity      = ConstantConductivity(; k=3.0Watt / K / m),
            LatentHeat        = ConstantLatentHeat(; Q_L=350e3J / kg),
            ShearHeat         = ConstantShearheating(0.0NoUnits),
            CompositeRheology = CompositeRheology((creep_rock, el, pl)),
            # Melting           = MeltingParam_Caricchi(),
            Gravity           = ConstantGravity(; g=g),
            Elasticity        = el,
            CharDim           = CharDim,
        ),
        #Name="LowerCrust"
        SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0=2650kg / m^3, α=3e-5 / K, T0=20.0C, β=β_rock / Pa),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1050J / kg / K),
            Conductivity      = ConstantConductivity(; k=3.0Watt / K / m),
            LatentHeat        = ConstantLatentHeat(; Q_L=350e3J / kg),
            ShearHeat         = ConstantShearheating(0.0NoUnits),
            CompositeRheology = CompositeRheology((creep_rock, el, pl)),
            # Melting           = MeltingParam_Caricchi(),
            Gravity           = ConstantGravity(; g=g),
            Elasticity        = el,
            CharDim           = CharDim,
        ),
        #Name="Magma"
        SetMaterialParams(;
            Phase             = 3,
            Density           = PT_Density(; ρ0=2450kg / m^3, T0=20.0C, β=β_magma / Pa),
            # Density           = PT_Density(; ρ0=2650kg / m^3, α=3e-5 / K, T0=20.0C, β=β_rock / Pa),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1050J / kg / K),
            Conductivity      = ConstantConductivity(; k=1.5Watt / K / m),
            # LatentHeat        = ConstantLatentHeat(; Q_L=350e3J / kg),
            ShearHeat         = ConstantShearheating(0.0NoUnits),
            CompositeRheology = CompositeRheology((creep_magma, el_magma)),
            # Melting           = MeltingParam_Caricchi(),
            Gravity           = ConstantGravity(; g=g),
            Elasticity        = el_magma,
            CharDim           = CharDim,
        ),
        #Name="Sticky Air"
        SetMaterialParams(;
            Phase             = 4,
            Density           = ConstantDensity(ρ = 100kg/m^3,),
            # Density           = ConstantDensity(ρ=1kg/m^3,),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1000J / kg / K),
            Conductivity      = ConstantConductivity(; k=1e-1Watt / K / m),
            LatentHeat        = ConstantLatentHeat(; Q_L=0.0J / kg),
            ShearHeat         = ConstantShearheating(0.0NoUnits),
            CompositeRheology = CompositeRheology((creep_air,)),
            Gravity           = ConstantGravity(; g=g),
            CharDim           = CharDim,
     ),
    )

end

function init_phases!(phases, phase_grid, particles, xvi)
    ni = size(phases)
    @parallel (@idx ni) _init_phases!(phases, phase_grid, particles.coords, particles.index, xvi)
end

@parallel_indices (I...) function _init_phases!(phases, phase_grid, pcoords::NTuple{N, T}, index, xvi) where {N,T}

    ni = size(phases)

    for ip in JustRelax.cellaxes(phases)
        # quick escape
        @cell(index[ip, I...]) == 0 && continue

        pᵢ = ntuple(Val(N)) do i 
            @cell pcoords[i][ip, I...]
        end

        d = Inf # distance to the nearest particle
        particle_phase = -1
        for offi in 0:1, offj in 0:1, offk in 0:1
            ii, jj, kk = I[1] + offi, I[2] + offj, I[3] + offk

            !(ii ≤ ni[1]) && continue
            !(jj ≤ ni[2]) && continue
            !(kk ≤ ni[3]) && continue

            xvᵢ = (
                xvi[1][ii], 
                xvi[2][jj], 
                xvi[3][kk], 
            )
            d_ijk = √(sum((pᵢ[i] - xvᵢ[i])^2 for i in 1:N))
            if d_ijk < d
                d = d_ijk
                particle_phase = phase_grid[ii, jj, kk]
            end
        end
        JustRelax.@cell phases[ip, I...] = Float64(particle_phase)
    end

    return nothing
end