# import JustRelax.compute_ρCp

"""
    subgrid_characteristic_time!(subgrid_arrays, particles, dt₀, phases, rheology, thermal::ThermalArrays, stokes::StokesArrays[, di])

Compute, per cell, the characteristic thermal diffusion timescale `dt₀ = ρCp / (2 K Σ dxi⁻²)`
used for JustPIC's subgrid-diffusion correction of particle temperature, evaluating
`rheology`'s density/heat-capacity/conductivity at the local phase (from `phases`, either a
`JustPIC.PhaseRatios` or an integer phase-id array) and temperature/pressure.
"""
function subgrid_characteristic_time!(
        subgrid_arrays,
        particles,
        dt₀,
        phases::JustPIC.PhaseRatios,
        rheology,
        thermal::JustRelax.ThermalArrays,
        stokes::JustRelax.StokesArrays,
    )
    ni = size(stokes.P)
    @parallel (@idx ni) subgrid_characteristic_time!(
        dt₀, phases.center, rheology, thermal.T, stokes.P, particles.di.vertex
    )
    return nothing
end

function subgrid_characteristic_time!(
        subgrid_arrays,
        particles,
        dt₀,
        phases::AbstractArray{Int, N},
        rheology,
        thermal::JustRelax.ThermalArrays,
        stokes::JustRelax.StokesArrays,
        di,
    ) where {N}
    ni = size(stokes.P)
    @parallel (@idx ni) subgrid_characteristic_time!(
        dt₀, phases, rheology, thermal.T, stokes.P, di
    )
    return nothing
end

@parallel_indices (I...) function subgrid_characteristic_time!(
        dt₀, phase_ratios, rheology, T, P, di
    )
    argsᵢ = getindex_NamedTuple((; T, P), I...)
    phaseᵢ = @cell phase_ratios[I...]

    # Compute the characteristic timescale `dt₀` of the local cell
    ρCp = compute_ρCp(rheology, phaseᵢ, argsᵢ)
    K = compute_conductivity(rheology, phaseᵢ, argsᵢ)
    sum_dxi = mapreduce(x -> inv(x)^2, +, @dxi(di, I...))
    dt₀[I...] = ρCp / (2 * K * sum_dxi)

    return nothing
end
