@inline gmg_shift(I::NTuple{N, Int}, d, s) where {N} = ntuple(k -> I[k] + ifelse(k == d, s, 0), Val(N))

# Centers: deviatoric normal stress, divergence and augmented pressure Π.
@parallel_indices (I...) function _gmg_center_stress!(τ, Π, divergence, V, p, η, compliance, γ, inv_width)
    N = Val(length(I))
    ε = ntuple(N) do d
        # Lower face of center I in V_d storage: transverse indices carry a ghost.
        J = ntuple(k -> I[k] + ifelse(k == d, 0, 1), N)
        @inbounds (V[d][gmg_shift(J, d, 1)...] - V[d][J...]) * inv_width[d][I[d]]
    end
    div = sum(ε)
    @inbounds begin
        for d in eachindex(τ)
            τ[d][I...] = 2 * η[I...] * (ε[d] - div / 3)
        end
        divergence[I...] = div
        Π[I...] = p[I...] - γ[I...] * (div + compliance[I...] * p[I...])
    end
    return nothing
end

# Edges of the (a, b) plane: τ_ab = η (∂_b V_a + ∂_a V_b). Along a and b the
# edge index is a vertex; V_a/V_b store those directions as face or ghosted
# cell, so both use the same storage index of the lower neighbor.
@parallel_indices (I...) function _gmg_edge_stress!(τe, η_edge, Va, Vb, a, b, inv_distance_a, inv_distance_b)
    N = Val(length(I))
    J = ntuple(k -> I[k] + ifelse(k == a || k == b, 0, 1), N)
    @inbounds begin
        ∂bVa = (Va[gmg_shift(J, b, 1)...] - Va[J...]) * inv_distance_b[I[b]]
        ∂aVb = (Vb[gmg_shift(J, a, 1)...] - Vb[J...]) * inv_distance_a[I[a]]
        τe[I...] = η_edge[I...] * (∂bVa + ∂aVb)
    end
    return nothing
end

# Momentum rows of component C at interior storage faces; other entries are zero.
@parallel_indices (I...) function _gmg_momentum!(out, τ, τ_edge, Π, C, inv_width, inv_distance)
    N = Val(length(I))
    n = size(Π)
    inside = ntuple(k -> k == C ? 2 <= I[k] <= n[k] : 2 <= I[k] <= n[k] + 1, N)
    @inbounds out[I...] = zero(eltype(out))
    @inbounds if all(inside)
        right = ntuple(k -> I[k] - ifelse(k == C, 0, 1), N)
        left = gmg_shift(right, C, -1)
        _dist = inv_distance[C][I[C]]
        force = (τ[C][right...] - τ[C][left...]) * _dist
        for b in 1:length(I)
            b == C && continue
            # Edge along C at the face vertex, along b at the cell's upper/lower vertex.
            upper = ntuple(k -> I[k] - ifelse(k == C || k == b, 0, 1), N)
            lower = gmg_shift(upper, b, -1)
            τe = τ_edge[JustRelax.gmg_pair_index(b, C)]
            force += (τe[upper...] - τe[lower...]) * inv_width[b][I[b] - 1]
        end
        out[I...] = (Π[right...] - Π[left...]) * _dist - force
    end
    return nothing
end

"""
    gmg_apply_operator!(op, e)

Evaluate `op.out = K e` for a prepared correction `e` (BCs applied, halos
exchanged; see `gmg_prepare!`). Rows are valid on owned retained DOFs.
"""
function gmg_apply_operator!(op::JustRelax.GMGLevelOperator{N}, e::JustRelax.GMGCoupledFields{N}) where {N}
    coefficients = op.coefficients
    ni = size(e.p)
    @parallel (@idx ni) _gmg_center_stress!(
        op.τ, op.Π, op.divergence, e.v, e.p, coefficients.η, coefficients.compliance,
        coefficients.γ, op.inv_width,
    )
    for (k, (a, b)) in enumerate(JustRelax.gmg_pairs(Val(N)))
        τe = op.τ_edge[k]
        @parallel (@idx size(τe)) _gmg_edge_stress!(
            τe, coefficients.η_edge[k], e.v[a], e.v[b], a, b, op.inv_distance[a], op.inv_distance[b],
        )
    end
    for C in 1:N
        V = op.out.v[C]
        @parallel (@idx size(V)) _gmg_momentum!(V, op.τ, op.τ_edge, op.Π, C, op.inv_width, op.inv_distance)
    end
    op.out.p .= op.divergence .+ coefficients.compliance .* e.p
    return nothing
end

# Harmonic mean of the up to four cells around an edge (clamped at storage ends).
@parallel_indices (I...) function _gmg_edge_viscosity!(η_edge, η, a, b)
    N = Val(length(I))
    n = size(η)
    acc = zero(eltype(η_edge))
    for sa in -1:0, sb in -1:0
        J = ntuple(k -> clamp(I[k] + ifelse(k == a, sa, 0) + ifelse(k == b, sb, 0), 1, n[k]), N)
        acc += inv(@inbounds η[J...])
    end
    @inbounds η_edge[I...] = 4 / acc
    return nothing
end

"""
    gmg_set_coefficients!(op, η; compliance = 0, γ = 0, pressure_scale = 1)

Set the center viscosity, compliance, augmentation and PH pressure-step scale
(each broadcast over centers; halos are exchanged here) and derive the edge
viscosities harmonically. Starts a new operator epoch: call
[`gmg_refresh_bounds!`](@ref) before smoothing.
"""
function gmg_set_coefficients!(op::JustRelax.GMGLevelOperator{N}, η; compliance = 0, γ = 0, pressure_scale = 1) where {N}
    c = op.coefficients
    c.η .= η
    c.compliance .= compliance
    c.γ .= γ
    c.pressure_scale .= pressure_scale
    foreach(A -> JustRelax.exchange_level!(op.exchange, A), (c.η, c.compliance, c.γ, c.pressure_scale))
    gmg_edge_coefficients!(op)
    return nothing
end

function gmg_edge_coefficients!(op::JustRelax.GMGLevelOperator{N}) where {N}
    for (k, (a, b)) in enumerate(JustRelax.gmg_pairs(Val(N)))
        η_edge = op.coefficients.η_edge[k]
        @parallel (@idx size(η_edge)) _gmg_edge_viscosity!(η_edge, op.coefficients.η, a, b)
    end
    return nothing
end

"""
    gmg_restrict_coefficients!(coarse, fine, transfer)

Coarse coefficients from fine ones: viscosity by volume-weighted harmonic
averaging; compliance, augmentation and pressure-step scale by volume-weighted
averaging. For compliance this is the Galerkin pressure row of injected
pressure; for `γ` it is the exact Galerkin grad-div term of a transfer that
commutes with divergence. Uses `fine.out.p` as scratch.
"""
function gmg_restrict_coefficients!(coarse::JustRelax.GMGLevelOperator{N}, fine::JustRelax.GMGLevelOperator{N}, transfer::JustRelax.GMGTransfer{N}) where {N}
    index = map(t -> t.restrict_index, transfer.pressure)
    weight = map(t -> t.restrict_weight, transfer.pressure)
    cf, cc = fine.coefficients, coarse.coefficients
    fluidity = fine.out.p
    fluidity .= inv.(cf.η)
    gmg_apply_tables!(cc.η, fluidity, 0, index, weight)
    pm = coarse.pressure_mask
    # Non-owned coarse entries are zero after restriction; keep them finite until the exchange.
    cc.η .= pm ./ (cc.η .+ (1 .- pm)) .+ (1 .- pm)
    gmg_apply_tables!(cc.compliance, cf.compliance, 0, index, weight)
    gmg_apply_tables!(cc.γ, cf.γ, 0, index, weight)
    gmg_apply_tables!(cc.pressure_scale, cf.pressure_scale, 0, index, weight)
    foreach(A -> JustRelax.exchange_level!(coarse.exchange, A), (cc.η, cc.compliance, cc.γ, cc.pressure_scale))
    gmg_edge_coefficients!(coarse)
    return nothing
end

# Global 3^N coloring of storage positions; offsets map storage to global indices.
@inline gmg_color(I, offset) = sum(ntuple(k -> mod(I[k] + offset[k], 3) * 3^(k - 1), Val(length(I))))

@parallel_indices (I...) function _gmg_probe!(V, color, offset)
    @inbounds V[I...] = ifelse(gmg_color(I, offset) == color, one(eltype(V)), zero(eltype(V)))
    return nothing
end

@parallel_indices (I...) function _gmg_probe_accumulate!(diagonal, row, out, same, color, offset)
    @inbounds begin
        value = out[I...]
        row[I...] += abs(value)
        same && gmg_color(I, offset) == color && (diagonal[I...] += value)
    end
    return nothing
end

"""
    gmg_refresh_bounds!(op)

Recompute the exact velocity-block diagonal (pressure fixed) and a Gershgorin
upper bound of `D⁻¹A` by colored probing: 3^N applications per component.
Requires a stencil reach of one storage index, which holds for this operator;
periodic directions need global cell counts divisible by 3. The bound is
reduced over owned retained rows on the level communicator.
"""
function gmg_refresh_bounds!(op::JustRelax.GMGLevelOperator{N}, partition::JustRelax.GMGPartition{N}) where {N}
    for d in 1:N
        partition.periodic[d] && JustRelax.gmg_require(
            partition.global_ni[d] % 3 == 0, "GMG probing needs periodic cell counts divisible by 3", partition.comm
        )
    end
    probe = op.direction
    T = eltype(probe.p)
    diagonal = map(similar, op.inv_diagonal)
    rows = map(similar, op.inv_diagonal)
    foreach(x -> fill!(x, zero(T)), (diagonal..., rows...))
    offset = map(first, partition.storage)
    for C in 1:N, color in 0:(3^N - 1)
        foreach(x -> fill!(x, zero(T)), (probe.p, probe.v...))
        V = probe.v[C]
        @parallel (@idx size(V)) _gmg_probe!(V, color, offset)
        JustRelax.gmg_correction_bcs!(op, probe)
        gmg_apply_operator!(op, probe)
        for R in 1:N
            out = op.out.v[R]
            @parallel (@idx size(out)) _gmg_probe_accumulate!(diagonal[R], rows[R], out, R == C, color, offset)
        end
    end
    bound = zero(T)
    for C in 1:N
        mask = op.velocity_mask[C]
        op.inv_diagonal[C] .= mask ./ (diagonal[C] .+ (1 .- mask))
        bound = max(bound, maximum(mask .* rows[C] ./ (diagonal[C] .+ (1 .- mask))))
    end
    positive = all(C -> minimum(op.velocity_mask[C] .* diagonal[C] .+ (1 .- op.velocity_mask[C])) > 0, 1:N)
    JustRelax.gmg_require(positive, "GMG velocity diagonal must be positive on owned retained rows", partition.comm)
    op.λ_upper[] = JustRelax.gmg_max(bound, partition.comm)
    return op.λ_upper[]
end

@parallel_indices (I...) function _gmg_chebyshev!(V, direction, rhs, out, inv_diagonal, c_direction, c_residual)
    @inbounds begin
        # Only owned retained rows (nonzero inverse diagonal) move; halos come from the exchange.
        if !iszero(inv_diagonal[I...])
            previous = ifelse(iszero(c_direction), zero(eltype(V)), c_direction * direction[I...])
            d = previous + c_residual * inv_diagonal[I...] * (rhs[I...] - out[I...])
            direction[I...] = d
            V[I...] += d
        end
    end
    return nothing
end

"""
    gmg_velocity_smooth!(op, e, rhs, smoother)

`substeps` Chebyshev velocity steps on the momentum rows of `K e = rhs` with the
pressure fixed, targeting `[λ/spectral_range, λ]` of `D⁻¹A`. The recurrence
starts fresh every call. On return `e` is prepared (BCs, halos).
"""
function gmg_velocity_smooth!(op::JustRelax.GMGLevelOperator{N}, e::JustRelax.GMGCoupledFields{N}, rhs::JustRelax.GMGCoupledFields{N}, smoother::JustRelax.GMGSmoother) where {N}
    T = eltype(e.p)
    b = op.λ_upper[]
    a = b / T(smoother.spectral_range)
    θ, δ = (a + b) / 2, (b - a) / 2
    σ = iszero(δ) ? zero(T) : θ / δ
    ρ = iszero(δ) ? zero(T) : inv(σ)
    for step in 1:smoother.substeps
        JustRelax.gmg_prepare!(op, e)
        gmg_apply_operator!(op, e)
        if step == 1 || iszero(δ)
            c_direction, c_residual = zero(T), inv(θ)
        else
            ρ_new = inv(2σ - ρ)
            c_direction, c_residual = ρ_new * ρ, 2ρ_new / δ
            ρ = ρ_new
        end
        for C in 1:N
            V = e.v[C]
            @parallel (@idx size(V)) _gmg_chebyshev!(
                V, op.direction.v[C], rhs.v[C], op.out.v[C], op.inv_diagonal[C], c_direction, c_residual,
            )
        end
    end
    JustRelax.gmg_prepare!(op, e)
    return nothing
end

"""
    gmg_dr_smooth!(op, e, rhs, smoother)

One DR/PH sweep on `K e = rhs`: [`gmg_velocity_smooth!`](@ref), then one
pressure step from fresh continuity,
`p += pressure_step · pressure_scale · (rhs_p - K_p e)`. On return `e` is
prepared (BCs, halos).
"""
function gmg_dr_smooth!(op::JustRelax.GMGLevelOperator{N}, e::JustRelax.GMGCoupledFields{N}, rhs::JustRelax.GMGCoupledFields{N}, smoother::JustRelax.GMGSmoother) where {N}
    T = eltype(e.p)
    gmg_velocity_smooth!(op, e, rhs, smoother)
    gmg_apply_operator!(op, e)
    β = T(smoother.pressure_step)
    e.p .+= ifelse.(iszero.(op.pressure_mask), zero(T), β .* op.coefficients.pressure_scale .* (rhs.p .- op.out.p))
    JustRelax.gmg_prepare!(op, e)
    return nothing
end
