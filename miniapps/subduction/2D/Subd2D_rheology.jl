using GeoParams.Dislocation
using GeoParams.Diffusion

function init_rheology_nonNewtonian()
    #dislocation laws
    disl_wet_olivine  = SetDislocationCreep(Dislocation.wet_olivine1_Hirth_2003)
    # diffusion laws
    diff_wet_olivine  = SetDiffusionCreep(Diffusion.wet_olivine_Hirth_2003)

    lithosphere_rheology = CompositeRheology( (disl_wet_olivine, diff_wet_olivine))
    init_rheologies(lithosphere_rheology)
end

function init_rheology_nonNewtonian_plastic()
    #dislocation laws
    disl_wet_olivine  = SetDislocationCreep(Dislocation.wet_olivine1_Hirth_2003)
    # diffusion laws
    diff_wet_olivine  = SetDiffusionCreep(Diffusion.wet_olivine_Hirth_2003)
    # plasticity
    ϕ_wet_olivine   = asind(0.1)
    C_wet_olivine   = 1e6 
    η_reg           = 1e19

    lithosphere_rheology = CompositeRheology( 
                (
                    disl_wet_olivine,
                    diff_wet_olivine,
                    DruckerPrager_regularised(; C = C_wet_olivine, ϕ = ϕ_wet_olivine, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
                ) 
            )
    init_rheologies(lithosphere_rheology)
end

function init_rheology_linear()
    lithosphere_rheology = CompositeRheology( (LinearViscous(; η=1e20),))
    init_rheologies(lithosphere_rheology)
end

function init_rheologies(lithosphere_rheology)
    # common physical properties
    α     = 2.4e-5 # 1 / K
    Cp    = 750  # J / kg K

    # Define rheolgy struct
    rheology = (
        # Name = "Asthenoshpere",
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ=3.2e3),
            HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
            Conductivity      = ConstantConductivity(; k  = 2.5),
            CompositeRheology = CompositeRheology( (LinearViscous(; η=1e20),)),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "Oceanic lithosphere",
        SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0=3.2e3, α = α, β = 0e0, T0 = 273+1474),
            HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
            Conductivity      = ConstantConductivity(; k  = 2.5),
            CompositeRheology = lithosphere_rheology,
        ),
        # Name              = "oceanic crust",
        SetMaterialParams(;
            Phase             = 3,
            Density           = ConstantDensity(; ρ=3.2e3),
            HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
            Conductivity      = ConstantConductivity(; k  = 2.5),
            CompositeRheology = CompositeRheology( (LinearViscous(; η=1e20),)),
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase             = 4,
            Density           = ConstantDensity(; ρ=100), # water density
            HeatCapacity      = ConstantHeatCapacity(; Cp=3e3),
            Conductivity      = ConstantConductivity(; k=1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e19),)),
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
        @cell phases[ip, I...] = Float64(particle_phase)
    end

    return nothing
end