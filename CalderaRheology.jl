## Rheology setup via rheology function

function init_rheology(CharDim; is_compressible=false, linear=true)

    ## plasticity setup
    do_DP   = true               # do_DP=false: Von Mises, do_DP=true: Drucker-Prager (friction angle)
    η_reg   = 1.0e16Pas           # regularisation "viscosity" for Drucker-Prager
    Coh     = 10.0MPa              # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ_fric  = 30.0 * do_DP         # friction angle
    G0      = 25e9Pa        # elastic shear modulus
    G_magma = 10e9Pa        # elastic shear modulus perturbation

    ## Strain softening law
    # soft_C = LinearSoftening((ustrip(Coh)/2, ustrip(Coh)), (0e0, 1e-1))   # linear softening law
    soft_C  = NonLinearSoftening(; ξ₀=ustrip(Coh), Δ=ustrip(Coh) / 9999)       # nonlinear softening law

    pl      = DruckerPrager_regularised(; C=Coh, ϕ=ϕ_fric, η_vp=η_reg, Ψ=0.0, softening_C=soft_C)
    
    if is_compressible == true
        el       = SetConstantElasticity(; G=G0, ν=0.25)                    # elasticity of lithosphere
        el_magma = SetConstantElasticity(; G=G_magma, ν=0.25)               # elasticity of magma
        # el_air   = SetConstantElasticity(; ν=0.25, Kb=0.101MPa)             # elasticity of air
        el_air   = SetConstantElasticity(; ν=0.25, G=G_magma)             # elasticity of air
        β_rock    = inv(get_Kb(el))
        β_magma = inv(get_Kb(el_magma))
        Kb = get_Kb(el)
    else
        el       = SetConstantElasticity(; G=G0, ν=0.5)                     # elasticity of lithosphere
        el_magma = SetConstantElasticity(; G=G_magma, ν=0.5)                # elasticity of magma
        el_air   = SetConstantElasticity(; ν=0.5, Kb=0.101MPa)              # elasticity of air
        β_rock = inv(get_Kb(el))
        β_magma = inv(get_Kb(el_magma))
        Kb = get_Kb(el)
    end

    ## Viscosity setup
    if linear == true
        creep_rock  = LinearViscous(; η=1e23 * Pa * s)                         # viscosity of lithosphere
        creep_magma = LinearViscous(; η=1e16 * Pa * s)                         # viscosity of magma
        creep_air   = LinearViscous(; η=1e16 * Pa * s)                         # viscosity of air
        g           = 9.81m / s^2
    else # nonlinear
        # creep_rock  = SetDislocationCreep(Dislocation.mafic_granulite_Wilks_1990) # viscosity of lithosphere
        creep_rock  = SetDislocationCreep(Dislocation.wet_quartzite_Ueda_2008) # viscosity of lithosphere
        creep_magma = LinearViscous(; η=1e16 * Pa * s)                         # viscosity of magma
        creep_air   = LinearViscous(; η=1e16 * Pa * s)                         # viscosity of air
        g           = 9.81m / s^2
        # linear_viscosity_rhy      = ViscosityPartialMelt_Costa_etal_2009(η=LinearMeltViscosity(A = -8.1590, B = 2.4050e+04K, T0 = -430.9606K,η0=1e1Pa*s))
        # linear_viscosity_bas      = ViscosityPartialMelt_Costa_etal_2009(η=LinearMeltViscosity(A = -9.6012, B = 1.3374e+04K, T0 = 307.8043K, η0=1e1Pa*s))
    end

    ## Rheology setup
    # Set material parameters
    return rheology = (
        # #Name="UpperCrust"
        SetMaterialParams(;
            Phase               = 1,
            Density             = PT_Density(ρ0=2700kg/m^3, β=β_rock/Pa),
            # Density           = MeltDependent_Density(ρsolid=PT_Density(ρ0=2700kg/m^3, β=β_rock/Pa),ρmelt=PT_Density(ρ0=2300kg / m^3, β=β_rock/Pa)),
            # HeatCapacity        = Latent_HeatCapacity(Cp=ConstantHeatCapacity(), Q_L=350e3J/kg),
            HeatCapacity        = ConstantHeatCapacity(Cp=1050J/kg/K),
            Conductivity        = ConstantConductivity(k=3.0Watt/K/m),
            LatentHeat          = ConstantLatentHeat(Q_L=350e3J/kg),
            ShearHeat           = ConstantShearheating(1.0NoUnits),
            CompositeRheology   = CompositeRheology((creep_rock, el, pl, )),
            # Melting             = MeltingParam_Smooth3rdOrder(a=3043.0,b=-10552.0,c=12204.9,d=-4709.0),
            Melting             = MeltingParam_Caricchi(),
            Elasticity          = el,
            CharDim             = CharDim,
            ),

        #Name="Magma"
        SetMaterialParams(;
            Phase               = 2,
            Density             = PT_Density(ρ0=2900kg/m^3, β=β_magma/Pa),
            # Density           = MeltDependent_Density(ρsolid=PT_Density(ρ0=2900kg/m^3, β=β_rock/Pa),ρmelt=PT_Density(ρ0=2800kg / m^3, β=β_rock/Pa)),
            # HeatCapacity        = Latent_HeatCapacity(Cp=ConstantHeatCapacity(), Q_L=350e3J/kg),
            HeatCapacity        = ConstantHeatCapacity(Cp=1050J/kg/K),
            Conductivity        = ConstantConductivity(k=1.5Watt/K/m),
            LatentHeat          = ConstantLatentHeat(Q_L=350e3J/kg),
            ShearHeat           = ConstantShearheating(0.0NoUnits),
            CompositeRheology   = CompositeRheology((creep_magma, el_magma)),
            # Melting             = MeltingParam_Smooth3rdOrder(),
            Melting             = MeltingParam_Caricchi(),
            Elasticity          = el_magma,
            CharDim             = CharDim,
            ),

        #Name="Thermal Anomaly"
        SetMaterialParams(;
            Phase               = 3,
            Density             = PT_Density(ρ0=2900kg/m^3, β=β_magma/Pa),
            # Density           = MeltDependent_Density(ρsolid=PT_Density(ρ0=2900kg/m^3, β=β_rock/Pa),ρmelt=PT_Density(ρ0=2800kg / m^3, β=β_rock/Pa)),
            # HeatCapacity        = Latent_HeatCapacity(Cp=ConstantHeatCapacity(), Q_L=350e3J/kg),
            HeatCapacity        = ConstantHeatCapacity(Cp=1050J/kg/K),
            Conductivity        = ConstantConductivity(k=1.5Watt/K/m),
            LatentHeat          = ConstantLatentHeat(Q_L=350e3J/kg),
            ShearHeat           = ConstantShearheating(0.0NoUnits),
            CompositeRheology   = CompositeRheology((creep_magma, el_magma)),
            # Melting             = MeltingParam_Smooth3rdOrder(),
            Melting             = MeltingParam_Caricchi(),
            Elasticity          = el_magma,
            CharDim             = CharDim,
            ),

        #Name="Sticky Air"
        SetMaterialParams(;
            Phase               = 4,
            Density             = ConstantDensity(ρ=10kg/m^3,),
            HeatCapacity        = ConstantHeatCapacity(Cp=1000J/kg/K),
            Conductivity        = ConstantConductivity(k=15Watt/K/m),
            LatentHeat          = ConstantLatentHeat(Q_L=0.0J/kg),
            ShearHeat           = ConstantShearheating(0.0NoUnits),
            CompositeRheology   = CompositeRheology((creep_air,el_air)),
            Elasticity          = el_air,
            # CompositeRheology   = CompositeRheology((creep_air, el)),
            # Elasticity          = el,
            # CompositeRheology   = CompositeRheology((creep_air, )),
            CharDim             = CharDim
            ),
        )
end

function init_phases2D!(phases, phase_grid, particles, xvi)
    ni = size(phases)
    @parallel (@idx ni) _init_phases2D!(
        phases, phase_grid, particles.coords, particles.index, xvi
    )
end

# @parallel_indices (I...) function _init_phases2D!(phases, phase_grid, pcoords::NTuple{N, T}, index, xvi) where {N,T}

#     ni = size(phases)

#     for ip in JustRelax.cellaxes(phases)
#         # quick escape
#         @cell(index[ip, I...]) == 0 && continue

#         pᵢ = ntuple(Val(N)) do i
#             @cell pcoords[i][ip, I...]
#         end

#         d = Inf # distance to the nearest particle
#         particle_phase = -1
#         for offi in 0:1, offj in 0:1
#             ii = I[1] + offi
#             jj = I[2] + offj

#             !(ii ≤ ni[1]) && continue
#             !(jj ≤ ni[2]) && continue

#             xvᵢ = (
#                 xvi[1][ii],
#                 xvi[2][jj],
#             )
#             if phase_grid[ii, jj] == 1.0
#                 particle_phase = 1.0
#             elseif phase_grid[ii, jj] == 2.0
#                 particle_phase = 2.0
#             elseif phase_grid[ii, jj] == 3.0
#                 particle_phase = 3.0
#             elseif phase_grid[ii, jj] == 4.0
#                 particle_phase = 4.0
#             end
#             # if pᵢ[end] > 0.0 && phase_grid[ii, jj] > 1.0
#             #     particle_phase = 4.0
#             # end
#         end
#         @cell phases[ip, I...] = Float64(particle_phase)
#     end

#     return nothing
# end

@parallel_indices (I...) function _init_phases2D!(
    phases, phase_grid, pcoords::NTuple{N,T}, index, xvi
) where {N,T}
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

            xvᵢ = (xvi[1][ii], xvi[2][jj])
            d_ijk = √(sum((pᵢ[i] - xvᵢ[i])^2 for i in 1:N))
            if d_ijk < d
                d = d_ijk
                particle_phase = phase_grid[ii, jj]
            end
            # if pᵢ[end] > 0.0 && phase_grid[ii, jj] > 1.0
            #     particle_phase = 4.0
            # end
        end
        @cell phases[ip, I...] = Float64(particle_phase)
    end

    return nothing
end

function init_phases3D!(phases, phase_grid, particles, xvi)
    ni = size(phases)
    @parallel (@idx ni) _init_phases3D!(
        phases, phase_grid, particles.coords, particles.index, xvi
    )
end

@parallel_indices (I...) function _init_phases3D!(
    phases, phase_grid, pcoords::NTuple{N,T}, index, xvi
) where {N,T}
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

            xvᵢ = (xvi[1][ii], xvi[2][jj], xvi[3][kk])
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
