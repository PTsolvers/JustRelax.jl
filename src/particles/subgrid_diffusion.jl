# import JustRelax.compute_ρCp

"""
    subgrid_characteristic_time!(subgrid_arrays, particles, dt₀, phases, rheology, thermal, stokes)

Compute, for every cell, the characteristic thermal-diffusion time and store it in `dt₀`.
The time is derived from the local thermal diffusivity of `rheology` evaluated at the current
temperature `thermal.T` and pressure `stokes.P`, and the grid spacing. It sets the relaxation
scale for subgrid temperature diffusion of the particles. `phases` is either a
`JustPIC.PhaseRatios` (phase-averaged) or an integer phase array (with the grid spacing `di`
passed as a trailing argument).
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
    Pᵢ, Tᵢ = P[I...], T[I .+ 1...]
    argsᵢ = (; P = Pᵢ, T = Tᵢ)
    phaseᵢ = @cell phase_ratios[I...]

    # Compute the characteristic timescale `dt₀` of the local cell
    ρCp = compute_ρCp(rheology, phaseᵢ, argsᵢ)
    K = compute_conductivity(rheology, phaseᵢ, argsᵢ)
    sum_dxi = mapreduce(x -> inv(x)^2, +, @dxi(di, I...))
    dt₀[I...] = ρCp / (2 * K * sum_dxi)

    return nothing
end
