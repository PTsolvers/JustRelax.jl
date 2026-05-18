using GeoParams
using GLMakie
using SpecialFunctions: erf

const SECONDS_PER_MYR = 1.0e6 * 365.25 * 24 * 60 * 60

const TABLE_S2 = Dict(
    :lubricating_layer => (
        rho_kg_m3 = 3300.0,
        disl = (E_J_mol = 154.0e3, V_m3_mol = 8.0e-6, A = 5.0e-6, n = 2.3),
        diff = nothing,
        C_Pa = 0.1e6,
        phi_deg = 0.1,
    ),
    :oceanic_crust => (
        rho_kg_m3 = 3000.0,
        disl = (E_J_mol = 154.0e3, V_m3_mol = 8.0e-6, A = 5.0e-18, n = 2.3),
        diff = nothing,
        C_Pa = 30.0e6,
        phi_deg = 30.0,
    ),
    :lithospheric_mantle => (
        rho_kg_m3 = 3300.0,
        disl = (E_J_mol = 530.0e3, V_m3_mol = 13.0e-6, A = 1.1e-17, n = 3.5),
        diff = (E_J_mol = 375.0e3, V_m3_mol = 4.0e-6, A = 2.2e-10),
        C_Pa = 30.0e6,
        phi_deg = 30.0,
    ),
    :upper_mantle => (
        rho_kg_m3 = 3300.0,
        disl = (E_J_mol = 530.0e3, V_m3_mol = 13.0e-6, A = 1.1e-17, n = 3.5),
        diff = (E_J_mol = 375.0e3, V_m3_mol = 4.0e-6, A = 2.2e-10),
        C_Pa = 30.0e6,
        phi_deg = 30.0,
    ),
    :lower_mantle => (
        rho_kg_m3 = 3300.0,
        disl = nothing,
        diff = (E_J_mol = 300.0e3, V_m3_mol = 2.0e-6, A = 2.0e-13),
        C_Pa = 30.0e6,
        phi_deg = 30.0,
    ),
)

function ocean_80myr_temperature(depth_m)
    surface_K = 273.0
    mantle_K = 273.0 + 1350.0
    cmb_K = 2873.0
    kappa = 1.0e-6
    age_s = 80.0 * SECONDS_PER_MYR

    if depth_m <= 2.64e6
        return surface_K + (mantle_K - surface_K) * erf(depth_m / (2.0 * sqrt(kappa * age_s)))
    else
        return mantle_K + (cmb_K - mantle_K) * (depth_m - 2.64e6) / 250.0e3
    end
end

function ocean_80myr_material(depth_m)
    if depth_m <= 15.0e3
        return :lubricating_layer
    elseif depth_m <= 22.0e3
        return :oceanic_crust
    elseif depth_m <= 120.0e3
        return :lithospheric_mantle
    elseif depth_m <= 660.0e3
        return :upper_mantle
    else
        return :lower_mantle
    end
end

function lithostatic_pressure(depth_m; dz_m = 250.0)
    g = 9.81
    pressure = 0.0
    for z in 0.0:dz_m:(depth_m - dz_m)
        material = TABLE_S2[ocean_80myr_material(z + 0.5 * dz_m)]
        pressure += material.rho_kg_m3 * g * dz_m
    end
    return pressure
end

function geoparams_dislocation_creep(creep)
    return DislocationCreep(;
        n = creep.n * NoUnits,
        A = creep.A * Pa^(-creep.n) / s,
        E = creep.E_J_mol * J / mol,
        V = creep.V_m3_mol * m^3 / mol,
        Apparatus = Invariant,
    )
end

function geoparams_diffusion_creep(creep; prefactor_scale = 1.0)
    return DiffusionCreep(;
        n = 1NoUnits,
        p = 0NoUnits,
        d = 1.0m,
        A = creep.A * prefactor_scale * Pa^-1 / s,
        E = creep.E_J_mol * J / mol,
        V = creep.V_m3_mol * m^3 / mol,
        Apparatus = Invariant,
    )
end

function geoparams_viscosity(creep_law, strain_rate, pressure_Pa, temperature_K)
    tau_II = compute_τII(
        creep_law,
        strain_rate / s,
        (; P = pressure_Pa * Pa, T = temperature_K * K),
    )
    return ustrip(Pa, tau_II) / (2.0 * strain_rate)
end

function creep_pressure(material_name, pressure_Pa; upper_mantle_pressure_scale = 0.75)
    lithosphere_base_pressure = lithostatic_pressure(160.0e3)
    if material_name in (:lithospheric_mantle, :upper_mantle)
        return upper_mantle_pressure_scale * max(pressure_Pa - lithosphere_base_pressure, 0.0)
    else
        return pressure_Pa
    end
end

function effective_viscosity(
        material_name,
        strain_rate,
        pressure_Pa,
        temperature_K;
        lower_mantle_prefactor_scale = 1.0,
        upper_mantle_pressure_scale = 0.75,
    )
    material = TABLE_S2[material_name]
    pressure_creep = creep_pressure(
        material_name,
        pressure_Pa;
        upper_mantle_pressure_scale,
    )

    eta_disl = isnothing(material.disl) ? Inf :
        geoparams_viscosity(
            geoparams_dislocation_creep(material.disl),
            strain_rate,
            pressure_creep,
            temperature_K,
        )
    diff_scale = isnothing(material.disl) ? lower_mantle_prefactor_scale : 1.0
    eta_diff = isnothing(material.diff) ? Inf :
        geoparams_viscosity(
            geoparams_diffusion_creep(material.diff; prefactor_scale = diff_scale),
            strain_rate,
            pressure_creep,
            temperature_K,
        )

    eta_comp = isinf(eta_disl) ? eta_diff :
        isinf(eta_diff) ? eta_disl :
        1.0 / (1.0 / eta_disl + 1.0 / eta_diff)
    yield_stress = material.C_Pa + pressure_Pa * sind(material.phi_deg)
    eta_plastic = yield_stress / (2.0 * strain_rate)

    return clamp(min(eta_comp, eta_plastic), 1.0e19, 1.0e23)
end

function model_viscosity_profile(;
        n = 5000,
        strain_rate = 1.0e-15,
        lower_mantle_prefactor_scale = 1.0,
    )
    depth = collect(range(0.0, 2.89e6; length = n))
    viscosity = similar(depth)

    for i in eachindex(depth)
        material_name = ocean_80myr_material(depth[i])
        pressure = lithostatic_pressure(depth[i])
        temperature = ocean_80myr_temperature(depth[i])
        viscosity[i] = effective_viscosity(
            material_name,
            strain_rate,
            pressure,
            temperature;
            lower_mantle_prefactor_scale,
        )

        if material_name == :lithospheric_mantle && depth[i] <= 100.0e3
            viscosity[i] = 1.0e23
        end
    end

    return log10.(viscosity), depth
end

function fig_s1b_geoparams_profiles()
    model1 = model_viscosity_profile(; lower_mantle_prefactor_scale = 0.1)
    model2 = model_viscosity_profile(; lower_mantle_prefactor_scale = 40.0)
    model3 = model_viscosity_profile(; lower_mantle_prefactor_scale = 0.05)
    return model1, model2, model3
end

function plot_fig_s1b_geoparams()
    model1, model2, model3 = fig_s1b_geoparams_profiles()

    fig = Figure(size = (560, 720))
    ax = Axis(
        fig[1, 1],
        xlabel = "Viscosity",
        ylabel = "Depth (m)",
        yreversed = true,
        xticks = 19:23,
    )

    lines!(ax, model1[1], model1[2]; color = :red, linewidth = 2, label = "Model1 Ocean 80 Myr")
    lines!(ax, model2[1], model2[2]; color = :blue, linewidth = 2, linestyle = :dashdot, label = "Model2 Ocean 80 Myr")
    lines!(ax, model3[1], model3[2]; color = :limegreen, linewidth = 3, linestyle = :dot, label = "Model3 Ocean 80 Myr")
    xlims!(ax, 18.5, 23.5)
    ylims!(ax, 2.89e6, 0)
    axislegend(ax; position = :lb, framevisible = true)

    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    fig = plot_fig_s1b_geoparams()
    display(fig)
end
