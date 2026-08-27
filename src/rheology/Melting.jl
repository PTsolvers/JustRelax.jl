"""
    compute_melt_fraction!(ϕ, rheology, args)
    compute_melt_fraction!(ϕ, phase_ratios::JustPIC.PhaseRatios, rheology, args)
    compute_melt_fraction!(ϕ, dϕdT, phase_ratios::JustPIC.PhaseRatios, rheology, args)

Fill the melt-fraction array `ϕ` from the GeoParams melting parameterisation of `rheology`,
with `args` supplying the state variables it needs (typically `P` and `T`, as scalars or
index-matched arrays). Given `phase_ratios`, the melt fraction is averaged over the phases
present in each cell.

The five-argument form also fills `dϕdT`, the temperature derivative of the melt fraction,
in the same pass. `dϕdT` is what activates `GeoParams.Latent_HeatCapacity`: it contributes
`Q_L * dϕdT` to `Cp`, and defaults to zero when absent from the `args` handed to the
thermal kernels. `ϕ` is clamped to `[0, 1]` and `dϕdT` is not, so a melting law with
`apply_bounds = false` can return a flat `ϕ` alongside a nonzero `dϕdT`.

There is no fused single-phase form: pair the three-argument method with
[`compute_melt_fraction_derivative!`](@ref), at the cost of a second pass over the grid.
"""
function compute_melt_fraction!(ϕ, rheology, args)
    ni = size(ϕ)
    @parallel (@idx ni) compute_melt_fraction_kernel!(ϕ, rheology, args)
    return nothing
end

@parallel_indices (I...) function compute_melt_fraction_kernel!(ϕ, rheology, args)
    args_ijk = getindex_NamedTuple(args, size(ϕ), I...)
    @inbounds ϕ[I...] = compute_meltfraction(rheology, args_ijk)
    return nothing
end

function compute_melt_fraction!(ϕ, phase_ratios::JustPIC.PhaseRatios, rheology, args)
    ni = size(ϕ)
    @parallel (@idx ni) compute_melt_fraction_kernel!(
        ϕ, phase_ratios.center, rheology, args
    )
    return nothing
end

@parallel_indices (I...) function compute_melt_fraction_kernel!(
        ϕ, phase_ratios, rheology, args
    )
    args_ijk = getindex_NamedTuple(args, size(ϕ), I...)
    @inbounds ϕ[I...] = fn_ratio(compute_meltfraction, rheology, @cell(phase_ratios[I...]), args_ijk)
    return nothing
end

function compute_melt_fraction!(ϕ, dϕdT, phase_ratios::JustPIC.PhaseRatios, rheology, args)
    ni = size(ϕ)
    size(dϕdT) == ni || throw(DimensionMismatch("size(dϕdT) = $(size(dϕdT)) must match size(ϕ) = $ni"))
    @parallel (@idx ni) compute_melt_fraction_kernel!(
        ϕ, dϕdT, phase_ratios.center, rheology, args
    )
    return nothing
end

@parallel_indices (I...) function compute_melt_fraction_kernel!(
        ϕ, dϕdT, phase_ratios, rheology, args
    )
    args_ijk = getindex_NamedTuple(args, size(ϕ), I...)
    ratio = @cell phase_ratios[I...]
    @inbounds ϕ[I...] = fn_ratio(compute_meltfraction, rheology, ratio, args_ijk)
    @inbounds dϕdT[I...] = fn_ratio(compute_dϕdT, rheology, ratio, args_ijk)
    return nothing
end

"""
    compute_melt_fraction_derivative!(dϕdT, rheology, args)
    compute_melt_fraction_derivative!(dϕdT, phase_ratios::JustPIC.PhaseRatios, rheology, args)

Fill `dϕdT`, the temperature derivative of the melt fraction, without touching
`ϕ`. Use this when `ϕ` is advected on particles or otherwise not recomputed on
the grid; when it is, the fused
[`compute_melt_fraction!`](@ref)`(ϕ, dϕdT, …)` does both in one pass.

In-place GeoParams equivalent: `compute_dϕdT!`.
"""
function compute_melt_fraction_derivative!(dϕdT, rheology, args)
    ni = size(dϕdT)
    @parallel (@idx ni) compute_melt_fraction_derivative_kernel!(dϕdT, rheology, args)
    return nothing
end

@parallel_indices (I...) function compute_melt_fraction_derivative_kernel!(
        dϕdT, rheology, args
    )
    args_ijk = getindex_NamedTuple(args, size(dϕdT), I...)
    @inbounds dϕdT[I...] = compute_dϕdT(rheology, args_ijk)
    return nothing
end

function compute_melt_fraction_derivative!(
        dϕdT, phase_ratios::JustPIC.PhaseRatios, rheology, args
    )
    ni = size(dϕdT)
    @parallel (@idx ni) compute_melt_fraction_derivative_kernel!(
        dϕdT, phase_ratios.center, rheology, args
    )
    return nothing
end

@parallel_indices (I...) function compute_melt_fraction_derivative_kernel!(
        dϕdT, phase_ratios, rheology, args
    )
    args_ijk = getindex_NamedTuple(args, size(dϕdT), I...)
    @inbounds dϕdT[I...] = fn_ratio(compute_dϕdT, rheology, @cell(phase_ratios[I...]), args_ijk)
    return nothing
end
