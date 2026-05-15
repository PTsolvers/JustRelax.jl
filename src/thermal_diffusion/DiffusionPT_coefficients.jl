function PTThermalCoeffs(
        ::Type{CPUBackend}, K, ρCp, dt, di::NTuple, li::NTuple; ϵ = 1.0e-8, CFL = 0.9 / √3
    )
    return PTThermalCoeffs(K, ρCp, dt, di, li; ϵ = ϵ, CFL = CFL)
end

function PTThermalCoeffs(K, ρCp, dt, di, li::NTuple; ϵ = 1.0e-8, CFL = 0.9 / √3)
    Vpdτ = min(di...) * CFL
    max_lxyz = max(li...)
    max_lxyz2 = max_lxyz^2
    Re = @. π + √(π * π + ρCp * max_lxyz2 / K / dt) # Numerical Reynolds number
    θr_dτ = @. max_lxyz / Vpdτ / Re
    dτ_ρ = @. Vpdτ * max_lxyz / K / Re

    return JustRelax.PTThermalCoeffs(CFL, ϵ, max_lxyz, max_lxyz2, Vpdτ, θr_dτ, dτ_ρ)
end

# with phase ratios
function PTThermalCoeffs(
        ::Type{CPUBackend},
        rheology,
        phase_ratios,
        args,
        dt,
        ni,
        di::NTuple,
        li::NTuple;
        ϵ = 1.0e-8,
        CFL = 0.9 / √3,
    )
    return PTThermalCoeffs(rheology, phase_ratios, args, dt, ni, di, li; ϵ = ϵ, CFL = CFL)
end

function PTThermalCoeffs(
        rheology, phase_ratios, args, dt, ni, di::NTuple, li::NTuple; ϵ = 1.0e-8, CFL = 0.9 / √3
    )
    Vpdτ = min(di...) * CFL
    max_lxyz = max(li...)
    θr_dτ, dτ_ρ = @zeros(ni...), @zeros(ni...)

    @parallel (@idx ni) compute_pt_thermal_arrays!(
        θr_dτ, dτ_ρ, rheology, phase_ratios.center, args, max_lxyz, Vpdτ, inv(dt)
    )

    return JustRelax.PTThermalCoeffs(CFL, ϵ, max_lxyz, max_lxyz^2, Vpdτ, θr_dτ, dτ_ρ)
end

# without phase ratios
function PTThermalCoeffs(
        ::Type{CPUBackend},
        rheology::MaterialParams,
        args,
        dt,
        ni,
        di::NTuple,
        li::NTuple;
        ϵ = 1.0e-8,
        CFL = 0.9 / √3,
    )
    return PTThermalCoeffs(rheology, args, dt, ni, di, li; ϵ = ϵ, CFL = CFL)
end

function PTThermalCoeffs(
        rheology::MaterialParams, args, dt, ni, di::NTuple, li::NTuple; ϵ = 1.0e-8, CFL = 0.9 / √3
    )
    Vpdτ = min(di...) * CFL
    max_lxyz = max(li...)
    θr_dτ, dτ_ρ = @zeros(ni...), @zeros(ni...)

    @parallel (@idx ni) compute_pt_thermal_arrays!(
        θr_dτ, dτ_ρ, rheology, args, max_lxyz, Vpdτ, inv(dt)
    )

    return JustRelax.PTThermalCoeffs(CFL, ϵ, max_lxyz, max_lxyz^2, Vpdτ, θr_dτ, dτ_ρ)
end

@parallel_indices (I...) function compute_pt_thermal_arrays!(
        θr_dτ::AbstractArray, dτ_ρ, rheology, phase, args, max_lxyz, Vpdτ, _dt
    )
    _compute_pt_thermal_arrays!(
        θr_dτ, dτ_ρ, rheology, phase, args, max_lxyz, Vpdτ, _dt, I...
    )

    return nothing
end

@parallel_indices (I...) function compute_pt_thermal_arrays!(
        θr_dτ::AbstractArray, dτ_ρ, rheology, args, max_lxyz, Vpdτ, _dt
    )
    _compute_pt_thermal_arrays!(θr_dτ, dτ_ρ, rheology, args, max_lxyz, Vpdτ, _dt, I...)

    return nothing
end

function _compute_pt_thermal_arrays!(
        θr_dτ, dτ_ρ, rheology, phase, args, max_lxyz, Vpdτ, _dt, Idx::Vararg{Int, N}
    ) where {N}
    args_ij = (; T = args.T[Idx.+1...], P = args.P[Idx...])
    phase_ij = phase[Idx...]
    ρCp = compute_ρCp(rheology, phase_ij, args_ij)
    _K = inv(fn_ratio(compute_conductivity, rheology, phase_ij, args_ij))

    _Re = inv(π + √(π * π + ρCp * max_lxyz^2 * _K * _dt)) # Numerical Reynolds number
    θr_dτ[Idx...] = max_lxyz / Vpdτ * _Re
    dτ_ρ[Idx...] = Vpdτ * max_lxyz * _K * _Re

    return nothing
end

function _compute_pt_thermal_arrays!(
        θr_dτ, dτ_ρ, rheology, args, max_lxyz, Vpdτ, _dt, Idx::Vararg{Int, N}
    ) where {N}
    args_ij = (; T = args.T[Idx.+1...], P = args.P[Idx...])

    ρCp = compute_ρCp(rheology, args_ij)
    _K = inv(compute_conductivity(rheology, args_ij))

    _Re = inv(π + √(π * π + ρCp * max_lxyz^2 * _K * _dt)) # Numerical Reynolds number
    θr_dτ[Idx...] = max_lxyz / Vpdτ * _Re
    dτ_ρ[Idx...] = Vpdτ * max_lxyz * _K * _Re

    return nothing
end

function update_thermal_coeffs!(
        pt_thermal::JustRelax.PTThermalCoeffs, rheology, phase_ratios, args, dt
    )
    ni = size(pt_thermal.dτ_ρ)
    @parallel (@idx ni) compute_pt_thermal_arrays!(
        pt_thermal.θr_dτ,
        pt_thermal.dτ_ρ,
        rheology,
        phase_ratios.center,
        args,
        pt_thermal.max_lxyz,
        pt_thermal.Vpdτ,
        inv(dt),
    )
    return nothing
end

function update_thermal_coeffs!(pt_thermal::JustRelax.PTThermalCoeffs, rheology, args, dt)
    ni = size(pt_thermal.dτ_ρ)
    @parallel (@idx ni) compute_pt_thermal_arrays!(
        pt_thermal.θr_dτ,
        pt_thermal.dτ_ρ,
        rheology,
        args,
        pt_thermal.max_lxyz,
        pt_thermal.Vpdτ,
        inv(dt),
    )
    return nothing
end

function update_thermal_coeffs!(
        pt_thermal::JustRelax.PTThermalCoeffs, rheology, ::Nothing, args, dt
    )
    ni = size(pt_thermal.dτ_ρ)
    @parallel (@idx ni) compute_pt_thermal_arrays!(
        pt_thermal.θr_dτ,
        pt_thermal.dτ_ρ,
        rheology,
        args,
        pt_thermal.max_lxyz,
        pt_thermal.Vpdτ,
        inv(dt),
    )
    return nothing
end
