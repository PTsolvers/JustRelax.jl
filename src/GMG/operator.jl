# Homogeneous augmented Stokes error operator on one level. With e = (v, p),
#
#   Π     = p - γ (∇·v + C p)
#   K_v e = -∇·(2η ε_dev(v)) + ∇Π
#   K_p e = ∇·v + C p
#
# which is -d(r_aug)/dx for r_aug = (F_v - ∇(γ F_p), F_p), F_p = -∇·v - C p,
# with zero forcing and histories. Coefficients are frozen per operator epoch.

"""Number of shear (edge) stress components: 1 in 2D, 3 in 3D."""
@inline gmg_npairs(::Val{N}) where {N} = N * (N - 1) ÷ 2
# Edge array index of the unordered direction pair (a, b): (1,2) → 1, (1,3) → 2, (2,3) → 3.
@inline gmg_pair_index(a, b) = a + b - 2
@inline gmg_pairs(::Val{2}) = ((1, 2),)
@inline gmg_pairs(::Val{3}) = ((1, 2), (1, 3), (2, 3))

"""
Frozen coefficients of one level: positive center viscosity `η`, shear
viscosity at the edges `η_edge` (vertices in 2D), pressure compliance
`compliance = 1/η_b` at centers (zero when incompressible), the augmentation
`γ` at centers (entering as the product `∇(γ F_p)`), and the PH pressure-step
scale `pressure_scale` at centers.
"""
struct GMGCoefficients{N, A, E}
    η::A
    η_edge::NTuple{E, A}
    compliance::A
    γ::A
    pressure_scale::A
end

"""DR/PH smoother controls; fixed during one sweep."""
struct GMGSmoother{T}
    substeps::Int
    spectral_range::T
    pressure_step::T
end

function GMGSmoother(; substeps::Integer = 5, spectral_range::Real = 8, pressure_step::Real = 1)
    substeps >= 1 || throw(ArgumentError("smoother needs at least one velocity substep"))
    spectral_range >= 1 || throw(ArgumentError("spectral_range must be at least 1"))
    pressure_step >= 0 || throw(ArgumentError("pressure_step must be nonnegative"))
    T = promote_type(typeof(float(spectral_range)), typeof(float(pressure_step)))
    return GMGSmoother{T}(Int(substeps), T(spectral_range), T(pressure_step))
end

"""
    GMGLevelOperator(level; no_slip)

Matrix-free operator workspace for one level: coefficients, reciprocal cell
widths and center distances, stress and output scratch, the velocity diagonal,
owned-retained masks, and the level halo exchange. `no_slip[d]` selects the
homogeneous tangential correction condition on the lower/upper physical
boundary normal to `d`; `false` is free slip. Normal velocity corrections on
nonperiodic physical boundaries are zero.
"""
struct GMGLevelOperator{N, A, V, E, F, X, T}
    coefficients::GMGCoefficients{N, A, E}
    inv_width::NTuple{N, V}
    inv_distance::NTuple{N, V}
    τ::NTuple{N, A}
    τ_edge::NTuple{E, A}
    Π::A
    divergence::A
    out::F
    direction::F
    inv_diagonal::NTuple{N, A}
    velocity_mask::NTuple{N, A}
    velocity_weight::NTuple{N, A}
    pressure_mask::A
    exchange::X
    no_slip::NTuple{N, NTuple{2, Bool}}
    boundary::NTuple{N, NTuple{2, Bool}}
    λ_upper::Base.RefValue{T}
end

# Center distance across each stored vertex; a nonperiodic physical wall
# mirrors the adjacent cell (ghost center at one cell width).
function gmg_center_distance(domain::GMGDomain{N, T}, partition::GMGPartition{N}, d) where {N, T}
    x, periodic, n = domain.xvi[d], partition.periodic[d], domain.ni[d]
    width(g) = periodic ? gmg_width(x, g, true) : gmg_width(x, clamp(g, 1, n), false)
    vertices = first(partition.storage[d]):(last(partition.storage[d]) + 1)
    return [(width(g - 1) + width(g)) / 2 for g in vertices]
end

function gmg_velocity_mask(template, partition::GMGPartition{N}, shape, C) where {N}
    mask = zeros(eltype(template), shape)
    owned = collect(gmg_owned_velocity(partition, Val(C)))
    if !partition.periodic[C]
        # Eliminated normal boundary faces carry a prescribed zero correction.
        first(partition.owned[C]) == 1 && (owned[C] = (first(owned[C]) + 1):last(owned[C]))
        last(partition.owned[C]) == partition.global_ni[C] && (owned[C] = first(owned[C]):(last(owned[C]) - 1))
    end
    mask[owned...] .= 1
    return copyto!(similar(template, eltype(template), shape), mask)
end

# Dual measure of owned retained component-C faces (zero elsewhere): the inner
# product in which the velocity block is symmetric. Transverse storage carries a ghost.
function gmg_velocity_weight(mask, measures::GMGMeasures{N}, ::Val{C}) where {N, C}
    face, cell = map(Array, measures.face), map(Array, measures.cell)
    host = Array(mask)
    for I in CartesianIndices(host)
        iszero(host[I]) && continue
        host[I] = prod(ntuple(d -> d == C ? face[d][I[d]] : cell[d][I[d] - 1], Val(N)))
    end
    return copyto!(similar(mask), host)
end

function GMGLevelOperator(
        level::GMGLevel;
        no_slip = ntuple(_ -> (false, false), Val(length(level.partition.global_ni)))
    )
    return gmg_level_operator(level, level.partition, no_slip)
end

function gmg_level_operator(level::GMGLevel, partition::GMGPartition{N}, no_slip::NTuple{N, NTuple{2, Bool}}) where {N}
    template = level.correction.p
    T = eltype(template)
    ni = size(template)
    device(x) = copyto!(similar(template, T, size(x)), x)
    zeros_like(shape) = fill!(similar(template, T, shape), zero(T))
    E = gmg_npairs(Val(N))
    edge_shape(a, b) = ntuple(k -> ni[k] + (k == a || k == b), Val(N))
    pairs = gmg_pairs(Val(N))
    coefficients = GMGCoefficients{N, typeof(template), E}(
        fill!(similar(template), one(T)),
        ntuple(k -> fill!(similar(template, T, edge_shape(pairs[k]...)), one(T)), Val(E)),
        zeros_like(ni), zeros_like(ni), fill!(similar(template), one(T)),
    )
    inv_width = map(c -> device(inv.(Array(c))), level.measures.cell)
    inv_distance = ntuple(d -> device(inv.(gmg_center_distance(level.domain, partition, d))), Val(N))
    vshape = map(size, level.correction.v)
    boundary = ntuple(Val(N)) do d
        physical = !partition.periodic[d]
        return (physical && first(partition.owned[d]) == 1, physical && last(partition.owned[d]) == partition.global_ni[d])
    end
    pmask = zeros(T, ni)
    pmask[partition.owned_local...] .= 1
    velocity_mask = ntuple(C -> gmg_velocity_mask(template, partition, vshape[C], C), Val(N))
    return GMGLevelOperator(
        coefficients, inv_width, inv_distance,
        ntuple(_ -> zeros_like(ni), Val(N)),
        ntuple(k -> zeros_like(edge_shape(pairs[k]...)), Val(E)),
        zeros_like(ni), zeros_like(ni),
        gmg_fields(template, ni), gmg_fields(template, ni),
        ntuple(C -> zeros_like(vshape[C]), Val(N)),
        velocity_mask, ntuple(C -> gmg_velocity_weight(velocity_mask[C], level.measures, Val(C)), Val(N)),
        device(pmask), GMGExchange(level), no_slip, boundary, Ref(zero(T)),
    )
end

"""Exchange the halos of one pressure-located (center) array."""
function exchange_level!(exchange::GMGExchange{N}, A::AbstractArray{<:Any, N}) where {N}
    for d in 1:N
        gmg_exchange_direction!(exchange.pressure[d], A, d, exchange.comm)
    end
    return nothing
end

"""
    gmg_correction_bcs!(op, fields)

Apply homogeneous correction BCs on physical boundaries: zero normal velocity,
and tangential ghosts mirrored (free slip) or negated (no slip).
"""
function gmg_correction_bcs!(op::GMGLevelOperator{N}, fields::GMGCoupledFields{N}) where {N}
    for C in 1:N
        V = fields.v[C]
        for d in 1:N, side in 1:2
            op.boundary[d][side] || continue
            n = size(V, d)
            if d == C
                gmg_slab(V, d, side == 1 ? (1:1) : (n:n)) .= 0
            else
                ghost, inner = side == 1 ? (1, 2) : (n, n - 1)
                s = op.no_slip[d][side] ? -one(eltype(V)) : one(eltype(V))
                gmg_slab(V, d, ghost:ghost) .= s .* gmg_slab(V, d, inner:inner)
            end
        end
    end
    return nothing
end

"""Apply correction BCs, then exchange halos, so `fields` can be differentiated."""
function gmg_prepare!(op::GMGLevelOperator, fields::GMGCoupledFields)
    gmg_correction_bcs!(op, fields)
    exchange_level!(op.exchange, fields)
    return nothing
end
