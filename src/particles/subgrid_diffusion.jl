# import JustRelax.compute_ρCp

# `particles.di.vertex` also holds the ghost cell on each side of the grid; grid cell `I` is
# entry `I + 1`.
particle_cell_spacing(particles) = map(_inner_cells, particles.di.vertex)
_inner_cells(di::Number) = di
_inner_cells(di::AbstractVector) = di[2:(end - 1)]

# `args` of the subgrid kernel are indexed at the cell `I`, so they must be numbers or cell arrays
function check_subgrid_args(args, ni)
    all(a -> a isa Number || size(a) == ni, values(args)) ||
        throw(DimensionMismatch("subgrid_characteristic_time!: args must be numbers or cell arrays of size $ni, got sizes $(map(a -> a isa Number ? () : size(a), args))"))
    return nothing
end

"""
    subgrid_characteristic_time!(subgrid_arrays, particles, dt₀, phases, rheology, thermal::ThermalArrays, stokes::StokesArrays[, di]; args = (;))

Compute, per cell, the characteristic thermal diffusion timescale `dt₀ = ρCp / (2 K Σ dxi⁻²)`
used for JustPIC's subgrid-diffusion correction of particle temperature, evaluating
`rheology`'s density/heat-capacity/conductivity at the local phase (from `phases`, either a
`JustPIC.PhaseRatios` or an integer phase-id array). The material laws receive the temperature
`thermal.T`, the pressure `stokes.P` and the fields of `args`, numbers or cell arrays of size
`size(stokes.P)` that add arguments (e.g. phase fractions) or replace `T` and `P`. `dt₀` must be
ghosted, with size `size(stokes.P) .+ 2`; values are written to its interior region.
"""
function subgrid_characteristic_time!(
        subgrid_arrays,
        particles,
        dt₀,
        phases::JustPIC.PhaseRatios,
        rheology,
        thermal::JustRelax.ThermalArrays,
        stokes::JustRelax.StokesArrays;
        args = (;),
    )
    ni = size(stokes.P)
    size(dt₀) == ni .+ 2 || throw(DimensionMismatch("dt₀ must have size $(ni .+ 2), got $(size(dt₀))"))
    check_subgrid_args(args, ni)
    @parallel (@idx ni) subgrid_characteristic_time!(
        dt₀, phases.center, rheology, thermal.T, stokes.P, args, particle_cell_spacing(particles)
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
        di;
        args = (;),
    ) where {N}
    ni = size(stokes.P)
    size(dt₀) == ni .+ 2 || throw(DimensionMismatch("dt₀ must have size $(ni .+ 2), got $(size(dt₀))"))
    check_subgrid_args(args, ni)
    @parallel (@idx ni) subgrid_characteristic_time!(
        dt₀, phases, rheology, thermal.T, stokes.P, args, di
    )
    return nothing
end

@parallel_indices (I...) function subgrid_characteristic_time!(
        dt₀, phase_ratios, rheology, T, P, args, di
    )
    # Temperature carries ghost cells, whereas pressure and `args` are physical-sized.
    # Index them explicitly so this remains device-compatible for GPU arrays; a field of `args`
    # replaces `T` or `P` of the same name.
    argsᵢ = (; T = T[(I .+ 1)...], P = P[I...], map(a -> getindex_or_scalar(a, I...), args)...)
    phaseᵢ = getindex_phase(phase_ratios, I...)

    # Compute the characteristic timescale `dt₀` of the local cell
    ρCp = compute_ρCp(rheology, phaseᵢ, argsᵢ)
    K = compute_conductivity(rheology, phaseᵢ, argsᵢ)
    sum_dxi = mapreduce(x -> inv(x)^2, +, @dxi(di, I...))
    dt₀[(I .+ 1)...] = ρCp / (2 * K * sum_dxi)

    return nothing
end
