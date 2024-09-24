# from "Fingerprinting secondary mantle plumes", Cloetingh et al. 2022

function init_rheologies(; is_plastic = true)
    # Dislocation and Diffusion creep
    # disl_upper_crust            = DislocationCreep(A=5.07e-18, n=2.3, E=154e3, V=6e-6,  r=0.0, R=8.3145)
    linear_viscosity_rhy            = LinearMeltViscosity(A = -8.1590, B = 2.4050e+04K, T0 = -430.9606K,η0=1e3Pa*s)
    # linear_viscosity_rhy            = LinearViscous(; η=1e13Pa*s)
    linear_viscosity_bas            = LinearMeltViscosity(A = -9.6012, B = 1.3374e+04K, T0 = 307.8043K, η0=1e3Pa*s)
    # linear_viscosity_bas            = LinearViscous(; η=1e9Pa*s)
    el_rhy              = SetConstantElasticity(; G=25e9, ν=0.5)
    el_bas              = SetConstantElasticity(; G=25e9, ν=0.5)
    β_rhy               = inv(get_Kb(el_rhy))
    β_bas               = inv(get_Kb(el_bas))

    # Physical properties using GeoParams ----------------
    η_reg     = 1e2
    cohesion  = 3e6
    friction  = 30.0
    pl_crust  = if is_plastic
        DruckerPrager_regularised(; C = cohesion, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    else
        DruckerPrager_regularised(; C = Inf, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    end
    friction  = asind(0.3)
    pl        = if is_plastic
        DruckerPrager_regularised(; C = cohesion, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    else
        DruckerPrager_regularised(; C = Inf, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    end
    pl_wz     = if is_plastic
        DruckerPrager_regularised(; C = 2e6, ϕ=2.0, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    else
        DruckerPrager_regularised(; C = Inf, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    end

    # crust
    #K_crust = TP_Conductivity(;
    #    a = 0.64,
    #    b = 807e0,
    #    c = 0.77,
    #    d = 0.00004*1e-6,
    #)
    K_crust  = ConstantConductivity()
    K_mantle = ConstantConductivity()
    #K_mantle = TP_Conductivity(;
    #    a = 0.73,
    #    b = 1293e0,
    #    c = 0.77,
    #    d = 0.00004*1e-6,
    #)
#

    # Define rheolgy struct
    rheology = (
        # Name              = "Rhyolite",
        SetMaterialParams(;
            Phase             = 1,
            Density           = MeltDependent_Density(ρmelt=T_Density(ρ0=2300kg / m^3), ρsolid=ConstantDensity(ρ=2700kg / m^3)),
            # Density           = PT_Density(; ρ0=2300, β=β_rhy, T0=0.0, α = 3.5e-5),
            # HeatCapacity      = ConstantHeatCapacity(),
            HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity(), Q_L=400e3J/kg),
            Conductivity      = K_crust,
            #CompositeRheology = CompositeRheology((linear_viscosity_rhy, el_rhy)),
            CompositeRheology = CompositeRheology((linear_viscosity_rhy,)),

            #Elasticity        = el_rhy,
            # Melting           = MeltingParam_Caricchi(),
            Melting           = MeltingParam_Smooth3rdOrder(a=3043.0,b=-10552.0,c=12204.9,d=-4709.0),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "Basaltic_Sill",
        SetMaterialParams(;
            Phase             = 2,
            Density           = MeltDependent_Density(ρmelt=T_Density(ρ0=2800kg / m^3)),
            # Density           = PT_Density(; ρ0=2800, β=β_bas, T0=0.0, α = 3.5e-5),
            # HeatCapacity      = ConstantHeatCapacity(),
            HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity(), Q_L=400e3J/kg),
            Conductivity      = K_crust,
            CompositeRheology = CompositeRheology((linear_viscosity_bas,)),
            # Melting           = MeltingParam_Caricchi(),
            Melting           = MeltingParam_Smooth3rdOrder(),

            #CompositeRheology = CompositeRheology((linear_viscosity_bas, el_bas)),
            #Elasticity        = el_bas,
        ),

    )
end


# function init_phases!(phases, particles)
#     ni = size(phases)

#     @parallel_indices (i, j) function init_phases!(phases, px, py, index)
#         @inbounds for ip in JustPIC._2D.cellaxes(phases)
#             # quick escape
#             @index(index[ip, i, j]) == 0 && continue

#             x = @index px[ip, i, j]
#             depth = -(@index py[ip, i, j])
#             @index phases[ip, i, j] = 1.0

#             if 0.1e3 < depth ≤ 0.2e3
#                 @index phases[ip, i, j] = 2.0

#             end
#         end
#         return nothing
#     end

#     @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index)
# end
function init_phases2D!(phases, phase_grid, particles, xvi)
    ni = size(phases)
    @parallel (@idx ni) _init_phases2D!(
        phases, phase_grid, particles.coords, particles.index, xvi
    )
end

@parallel_indices (I...) function _init_phases2D!(
    phases, phase_grid, pcoords::NTuple{N,T}, index, xvi
) where {N,T}
    ni = size(phases)

    for ip in JustRelax.cellaxes(phases)
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
        @index phases[ip, I...] = Float64(particle_phase)
    end

    return nothing
end
