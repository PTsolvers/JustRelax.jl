# from "Fingerprinting secondary mantle plumes", Cloetingh et al. 2022

function init_rheologies()

    # Dislocation and Diffusion creep
    background_viscosity = LinearViscous(; η = 1e21)
    magma_viscosity      = LinearViscous(; η = 1e16)
    air_viscosity        = LinearViscous(; η = 1e19)
    
    # Elasticity
    background_elasticity = SetConstantElasticity(; G=25e9, ν=0.5)
    background_β          = inv(get_Kb(background_elasticity))

    # Physical properties using GeoParams ----------------
    η_reg     = 1e16
    cohesion  = 3e6
    friction  = asind(0.2)
    background_pl = DruckerPrager_regularised(; C = cohesion, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity

    # Define rheolgy struct
    rheology = (
        # Name              = "UpperCrust",
        SetMaterialParams(;
            Phase             = 1,
            Density           = PT_Density(; ρ0=2.7e3, β=background_β, T0=273e0, α = 3.5e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=1050e0),
            Conductivity      = ConstantConductivity(k = 3.0),
            CompositeRheology = CompositeRheology((background_viscosity, background_elasticity, background_pl)),
            Elasticity        = background_elasticity,
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "LowerCrust",
        SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0=2.6e3, β=0e0, T0=273e0, α = 3.5e-5),
            Conductivity      = ConstantConductivity(k = 1.5),
            HeatCapacity      = ConstantHeatCapacity(; cp=1050e0),
            CompositeRheology = CompositeRheology((magma_viscosity, )),
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase             = 3,
            Density           = ConstantDensity(; ρ=1e3),
            HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
            Conductivity      = ConstantConductivity(; k=15.0),
            CompositeRheology = CompositeRheology((air_viscosity,)),
        ),
    )
    return rheology
end

# @inline inellipsoid(x::T, y::T, z::T, x0::T, y0::T, z0::T, a::T, b::T, c::T) where T = ((x - x0) / a)^2 + ((y - y0) / b)^2 + ((z - z0) / c)^2 ≤ one(T)

@inline inellipsoid(x::T, y::T, z::T, x0::T, y0::T, z0::T, a::T, b::T, c::T) where T = inellipsoid((x, y, z), (x0, y0, z0), (a, b, c))
@inline inellipsoid(x::NTuple{N, T}, x0::NTuple{N, T}, a::NTuple{N, T}) where {N, T} = sum(x -> ((x[1] - x[2]) / x[3])^2 , zip(x, x0, a)) ≤ one(T)

function init_phases!(phases, particles, Lx, Ly, d; a=15e3, b=5e3, c=5e3)
    ni = size(phases)

    @parallel_indices (I...) function init_phases!(phases, px, py, pz, index, Lx, Ly, a, b, c)
        
        @inbounds for ip in JustRelax.cellaxes(phases)
            # quick abortion
            @cell(index[ip, I...]) == false && continue

            x = @cell px[ip, I...]
            y = @cell py[ip, I...]
            depth = -(@cell pz[ip, I...])

            # volcanic chamber
            if inellipsoid(x, y, depth, Lx * 0.5, Ly * 0.5, abs(d), a, b, c)
                @cell phases[ip, I...] = 2.0
            else
                @cell phases[ip, I...] = 1.0
            end
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, Lx, Ly, a, b, c)
end

# # Initial thermal profile
# @parallel_indices (I...) function init_T!(T, xv, yv, zv, dTdZ, lx, ly, d, a, b, c)
    
#     x, y, depth = xv[I[1]], yv[I[2]], abs(zv[I[3]])

#     Tᵢ = if inellipsoid(x, y, depth, lx * 0.5, ly * 0.5, abs(d) + 2e3, 2e3, 2e3, 2e3)
#         900e0
#         # 1_050e0

#     elseif inellipsoid(x, y, depth, lx * 0.5, ly * 0.5, abs(d), a, b, c)
#         900e0
#     else 
#         min(depth * dTdZ * 1e-3, 1400e0)
#     end

#     T[I...] = Tᵢ + 273e0

#     return nothing
# end


# Initial thermal profile
@parallel_indices (I...) function init_T!(T, xv, yv, zv, dTdZ, lx, ly, d, a, b, c)
    
    x, y, depth = xv[I[1]], yv[I[2]], abs(zv[I[3]])

    T[I...] = min(depth * dTdZ * 1e-3, 1400e0) + 273e0

    if inellipsoid(x, y, depth, lx * 0.5, ly * 0.5, abs(d) + 2e3, 1.5e3, 1.5e3, 1.5e3)
        T[I...] *= 1.3

    elseif inellipsoid(x, y, depth, lx * 0.5, ly * 0.5, abs(d), a, b, c)
        T[I...] *= 1.2 
    end

    return nothing
end
