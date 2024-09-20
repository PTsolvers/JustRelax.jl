# import JustRelax.compute_ρCp

function subgrid_characteristic_time!(
    subgrid_arrays,
    particles,
    dt₀,
    phases::PhaseRatios,
    rheology,
    thermal::JustRelax.ThermalArrays,
    stokes::JustRelax.StokesArrays,
    xci,
    di,
)
    ni = size(stokes.P)
    @parallel (@idx ni) subgrid_characteristic_time!(
        dt₀, phases.center, rheology, thermal.Tc, stokes.P, di
    )
    return nothing
end

function subgrid_characteristic_time!(
    subgrid_arrays,
    particles,
    dt₀,
    phases::AbstractArray{Int,N},
    rheology,
    thermal::JustRelax.ThermalArrays,
    stokes::JustRelax.StokesArrays,
    xci,
    di,
) where {N}
    ni = size(stokes.P)
    @parallel (@idx ni) subgrid_characteristic_time!(
        dt₀, phases, rheology, thermal.Tc, stokes.P, di
    )
    return nothing
end

@parallel_indices (I...) function subgrid_characteristic_time!(
    dt₀, phase_ratios, rheology, T, P, di
)
    Pᵢ, Tᵢ = P[I...], T[I...]
    argsᵢ = (; P=Pᵢ, T=Tᵢ)
    phaseᵢ = phase_ratios[I...]

    # Compute the characteristic timescale `dt₀` of the local cell
    ρCp = compute_ρCp(rheology, phaseᵢ, argsᵢ)
    K = compute_conductivity(rheology, phaseᵢ, argsᵢ)
    sum_dxi = mapreduce(x -> inv(x)^2, +, di)
    dt₀[I...] = ρCp / (2 * K * sum_dxi)

    return nothing
end
