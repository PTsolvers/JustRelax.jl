# Model definition shared by Plume3D.jl and Plume3D_MPI.jl.
# Rheology after Cloetingh et al. (2022), "Fingerprinting secondary mantle plumes".
#
# Include this after `@init_parallel_stencil` and `using JustPIC._3D`: the kernels below
# need both.

function init_rheologies()

    # Dislocation and diffusion creep
    disl_upper_crust = DislocationCreep(A = 5.07e-18, n = 2.3, E = 154.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)
    disl_lower_crust = DislocationCreep(A = 2.08e-23, n = 3.2, E = 238.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)
    disl_lithospheric_mantle = DislocationCreep(A = 2.51e-17, n = 3.5, E = 530.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)
    disl_sublithospheric_mantle = DislocationCreep(A = 2.51e-17, n = 3.5, E = 530.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)
    diff_lithospheric_mantle = DislocationCreep(A = 2.51e-17, n = 1.0, E = 530.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)
    diff_sublithospheric_mantle = DislocationCreep(A = 2.51e-17, n = 1.0, E = 530.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)

    # Elasticity
    el_upper_crust = SetConstantElasticity(; G = 25.0e9, ν = 0.5)
    el_lower_crust = SetConstantElasticity(; G = 25.0e9, ν = 0.5)
    el_lithospheric_mantle = SetConstantElasticity(; G = 67.0e9, ν = 0.5)
    el_sublithospheric_mantle = SetConstantElasticity(; G = 67.0e9, ν = 0.5)
    β_upper_crust = inv(get_Kb(el_upper_crust))
    β_lower_crust = inv(get_Kb(el_lower_crust))
    β_lithospheric_mantle = inv(get_Kb(el_lithospheric_mantle))
    β_sublithospheric_mantle = inv(get_Kb(el_sublithospheric_mantle))

    # Regularised Drucker-Prager plasticity
    η_reg = 1.0e16
    cohesion = 3.0e6
    pl_crust = DruckerPrager_regularised(; C = cohesion, ϕ = asind(0.2), η_vp = η_reg, Ψ = 0.0)
    pl = DruckerPrager_regularised(; C = cohesion, ϕ = asind(0.3), η_vp = η_reg, Ψ = 0.0)

    # Pressure- and temperature-dependent conductivities
    K_crust = TP_Conductivity(; a = 0.64, b = 807.0e0, c = 0.77, d = 0.00004 * 1.0e-6)
    K_mantle = TP_Conductivity(; a = 0.73, b = 1293.0e0, c = 0.77, d = 0.00004 * 1.0e-6)

    return (
        # Name              = "UpperCrust",
        SetMaterialParams(;
            Phase = 1,
            Density = PT_Density(; ρ0 = 2.75e3, β = β_upper_crust, T0 = 0.0, α = 3.5e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = 7.5e2),
            Conductivity = K_crust,
            CompositeRheology = CompositeRheology((disl_upper_crust, el_upper_crust, pl_crust)),
            Elasticity = el_upper_crust,
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name              = "LowerCrust",
        SetMaterialParams(;
            Phase = 2,
            Density = PT_Density(; ρ0 = 3.0e3, β = β_lower_crust, T0 = 0.0, α = 3.5e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = 7.5e2),
            Conductivity = K_crust,
            CompositeRheology = CompositeRheology((disl_lower_crust, el_lower_crust, pl_crust)),
            Elasticity = el_lower_crust,
        ),
        # Name              = "LithosphericMantle",
        SetMaterialParams(;
            Phase = 3,
            Density = PT_Density(; ρ0 = 3.3e3, β = β_lithospheric_mantle, T0 = 0.0, α = 3.0e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1.25e3),
            Conductivity = K_mantle,
            CompositeRheology = CompositeRheology((disl_lithospheric_mantle, diff_lithospheric_mantle, el_lithospheric_mantle, pl)),
            Elasticity = el_lithospheric_mantle,
        ),
        # Name              = "SubLithosphericMantle",
        SetMaterialParams(;
            Phase = 4,
            Density = PT_Density(; ρ0 = 3.3e3, β = β_sublithospheric_mantle, T0 = 0.0, α = 3.0e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1.25e3),
            Conductivity = K_mantle,
            CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el_sublithospheric_mantle)),
            Elasticity = el_sublithospheric_mantle,
        ),
        # Name              = "Plume", 50 kg/m³ lighter than the sublithospheric mantle
        SetMaterialParams(;
            Phase = 5,
            Density = PT_Density(; ρ0 = 3.3e3 - 50, β = β_sublithospheric_mantle, T0 = 0.0, α = 3.0e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1.25e3),
            Conductivity = K_mantle,
            CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el_sublithospheric_mantle)),
            Elasticity = el_sublithospheric_mantle,
        ),
    )
end

# Layered lithosphere with a cubic plume of half-width `r` centered at depth `d`.
# The particle coordinates are global, so under MPI every rank builds the part of the
# model that falls inside its own subdomain without any extra bookkeeping.
function init_phases!(phases, particles, Lx, Ly; d = 650.0e3, r = 50.0e3)
    ni = size(phases)

    @parallel_indices (I...) function _init_phases!(phases, px, py, pz, index, r, d, Lx, Ly)
        for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, I...]) == 0 && continue

            x = @index px[ip, I...]
            y = @index py[ip, I...]
            depth = -(@index pz[ip, I...])

            if 0.0e0 ≤ depth ≤ 21.0e3
                @index phases[ip, I...] = 1.0

            elseif 35.0e3 ≥ depth > 21.0e3
                @index phases[ip, I...] = 2.0

            elseif 90.0e3 ≥ depth > 35.0e3
                @index phases[ip, I...] = 3.0

            elseif depth > 90.0e3
                @index phases[ip, I...] = 4.0

            end

            # plume - rectangular
            if ((x - Lx * 0.5)^2 ≤ r^2) && ((y - Ly * 0.5)^2 ≤ r^2) && ((depth - d)^2 ≤ r^2)
                @index phases[ip, I...] = 5.0
            end
        end
        return nothing
    end

    return @parallel (@idx ni) _init_phases!(phases, particles.coords..., particles.index, r, d, Lx, Ly)
end

# Piecewise-linear geotherm
@parallel_indices (I...) function init_T!(T, z)
    depth = -z[I[3]]

    if depth < 0.0e0
        T[I...] = 273.0

    elseif 0.0e0 ≤ depth < 35.0e3
        dTdZ = (923 - 273) / 35.0e3
        offset = 273.0e0
        T[I...] = depth * dTdZ + offset

    elseif 110.0e3 > depth ≥ 35.0e3
        dTdZ = (1492 - 923) / 75.0e3
        offset = 923.0e0
        T[I...] = (depth - 35.0e3) * dTdZ + offset

    elseif depth ≥ 110.0e3
        dTdZ = (1837 - 1492) / 590.0e3
        offset = 1492.0e0
        T[I...] = (depth - 110.0e3) * dTdZ + offset

    end

    return nothing
end

# Thermal rectangular perturbation, co-located with the plume
function rectangular_perturbation!(T, xc, yc, zc, r, xvi)

    @parallel_indices (i, j, k) function _rectangular_perturbation!(T, xc, yc, zc, r, x, y, z)
        if (abs(x[i] - xc) ≤ r) && (abs(y[j] - yc) ≤ r) && (abs(z[k] - zc) ≤ r)
            depth = abs(z[k])
            dTdZ = (2047 - 2017) / 50.0e3
            offset = 2017
            T[i, j, k] = (depth - 585.0e3) * dTdZ + offset
        end
        return nothing
    end

    return @parallel _rectangular_perturbation!(T, xc, yc, zc, r, xvi...)
end
