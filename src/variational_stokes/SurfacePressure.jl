# A momentum-only view of pressure. Constitutive and continuity kernels continue
# to receive the physical pressure array, including its absolute pressure level.
struct SurfacePressure{P, L}
    pressure::P
    load::L
end

Adapt.@adapt_structure SurfacePressure

function prepare_surface_pressure(P, surface_pressure, ϕ, flow_bcs)
    isnothing(surface_pressure) && return P
    flow_bcs.free_surface && throw(
        ArgumentError(
            "`surface_pressure` applies to the embedded RockRatio interface; use `flow_bcs.free_surface = false` to avoid the zero-traction box-top condition",
        )
    )
    load = if surface_pressure isa Real
        isfinite(surface_pressure) || throw(ArgumentError("`surface_pressure` must be finite"))
        convert(eltype(P), surface_pressure)
    elseif surface_pressure isa AbstractArray{<:Real}
        axes(surface_pressure) == axes(P) || throw(
            ArgumentError(
                "`surface_pressure` must be a cell-centered field with the same axes as `stokes.P`",
            )
        )
        backend(parent(surface_pressure)) == backend(P) || throw(
            ArgumentError(
                "`surface_pressure` and `stokes.P` must use the same backend",
            )
        )
        # Snapshot the prescribed field for this solve. Halo exchange must not
        # overwrite the caller's field, which may also be used by another model.
        snapshot = similar(P)
        copyto!(snapshot, surface_pressure)
        update_halo!(snapshot)
        snapshot
    else
        throw(ArgumentError("`surface_pressure` must be nothing, a real scalar, or a cell-centered real array"))
    end
    # The reduced pressure weight depends on the surrounding face masks. Exchange
    # those masks too, before evaluating a load stencil across a rank boundary.
    faces = ndims(P) == 2 ? (ϕ.Vx, ϕ.Vy) : (ϕ.Vx, ϕ.Vy, ϕ.Vz)
    update_halo!(ϕ.center, faces...)
    return SurfacePressure(P, load)
end

@inline surface_pressure_value(P::Real, I...) = P
Base.@propagate_inbounds @inline surface_pressure_value(P::AbstractArray, I...) = P[I...]

# Eliminated pressure rows have zero weight in the reduced pressure operator.
# Apply the same projection to the prescribed field: otherwise a constant load
# would lose part of its pressure offset when a sliver pressure cell is removed.
Base.@propagate_inbounds @inline function surface_pressure_weight(ϕ, I...)
    return isvalid_c(ϕ, I...) ? ϕ.center[I...] : zero(eltype(ϕ.center))
end

Base.@propagate_inbounds @inline function surface_pressure_force(Pload, ϕ, ::Val{d}, _d, I::Vararg{Integer, N}) where {d, N}
    J = ntuple(k -> k == d ? wrap_next(I[k], size(ϕ.center, k)) : I[k], Val(N))
    F = ntuple(k -> I[k] + (k == d), Val(N))
    faces = (ϕ.Vx, ϕ.Vy, ϕ.Vz)
    wf = faces[d][F...]
    wp_lo = surface_pressure_weight(ϕ, I...)
    wp_hi = surface_pressure_weight(ϕ, J...)
    p_lo = surface_pressure_value(Pload, I...)
    p_hi = surface_pressure_value(Pload, J...)
    # G(Wp * Pload) - Wv * G(Pload), Larionov et al. (2017), §5.1.
    # Factoring each endpoint makes cancellation exact in fully filled stencils.
    return ((wp_hi - wf) * p_hi - (wp_lo - wf) * p_lo) * _d
end

Base.@propagate_inbounds @inline function variational_pressure_gradient(P, ϕ, d::Val, _d, I::Vararg{Integer, N}) where {N}
    return (_masked_next_wrap(P, ϕ.center, d, I) - center(P, ϕ.center, I...)) * _d
end

Base.@propagate_inbounds @inline function variational_pressure_gradient(P::SurfacePressure, ϕ, d::Val, _d, I::Vararg{Integer, N}) where {N}
    return variational_pressure_gradient(P.pressure, ϕ, d, _d, I...) -
        surface_pressure_force(P.load, ϕ, d, _d, I...)
end
