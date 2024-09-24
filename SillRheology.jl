# from "Fingerprinting secondary mantle plumes", Cloetingh et al. 2022

function init_rheologies(CharDim)
    linear_viscosity_rhy            = LinearMeltViscosity(A = -8.1590, B = 2.4050e+04K, T0 = -430.9606K)#,η0=1e0Pa*s)
    linear_viscosity_bas            = LinearMeltViscosity(A = -9.6012, B = 1.3374e+04K, T0 = 307.8043K)#, η0=1e3Pa*s)

    # Define rheolgy struct
    rheology = (
        # Name              = "Rhyolite",
        SetMaterialParams(;
            Phase             = 1,
            Density           = MeltDependent_Density(ρsolid=ConstantDensity(ρ=2700kg / m^3),ρmelt=T_Density(ρ0=2300kg / m^3)),
            HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity(), Q_L=400e3J/kg),
            Conductivity      = ConstantConductivity(),
            CompositeRheology = CompositeRheology((linear_viscosity_rhy,)),
            Melting           = MeltingParam_Smooth3rdOrder(a=3043.0,b=-10552.0,c=12204.9,d=-4709.0),
            CharDim           = CharDim,
        ),
        # Name              = "Basaltic_Sill",
        SetMaterialParams(;
            Phase             = 2,
            Density           = MeltDependent_Density(ρsolid=ConstantDensity(ρ=3000kg/m^3),ρmelt=T_Density(ρ0=2800kg / m^3)),
            HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity(), Q_L=400e3J/kg),
            Conductivity      = ConstantConductivity(),
            CompositeRheology = CompositeRheology((linear_viscosity_bas,)),
            Melting           = MeltingParam_Smooth3rdOrder(),
            CharDim           = CharDim,
        ),
        # # Name              = "Basaltic_Sill_thermal_Anomaly",
        # SetMaterialParams(;
        #     Phase             = 3,
        #     Density           = MeltDependent_Density(ρsolid=ConstantDensity(ρ=2900kg/m^3),ρmelt=T_Density(ρ0=2800kg / m^3)),
        #     HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity(), Q_L=400e3J/kg),
        #     Conductivity      = ConstantConductivity(),
        #     CompositeRheology = CompositeRheology((linear_viscosity_bas,)),
        #     Melting           = MeltingParam_Smooth3rdOrder(),
        #     CharDim           = CharDim,
        # ),
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
function init_phases!(phases, phase_grid, particles, xvi)
    ni = size(phases)
    @parallel (@idx ni) _init_phases!(
        phases, phase_grid, particles.coords, particles.index, xvi
    )
end

@parallel_indices (I...) function _init_phases!(
    phases, phase_grid, pcoords::NTuple{N,T}, index, xvi
) where {N,T}
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
