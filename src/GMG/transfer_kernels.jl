# Tensor-product gather: out[I] = β out[I] + Σ_a Π_d w_d[I_d, a_d] input[idx_d[I_d, a_d]...].
# Padded entries carry zero weight and are skipped, so stale halo/ghost values
# in `input` never enter the sum.
@parallel_indices (I...) function _gmg_transfer!(out, input, β, index, weight)
    N = Val(length(I))
    acc = zero(eltype(out))
    for a in CartesianIndices(ntuple(d -> axes(index[d], 2), N))
        w = prod(ntuple(d -> @inbounds(weight[d][I[d], a[d]]), N))
        iszero(w) && continue
        J = ntuple(d -> Int(@inbounds(index[d][I[d], a[d]])), N)
        acc += w * @inbounds(input[J...])
    end
    # β = 0 overwrites, so NaN or stale values already in `out` cannot leak through.
    @inbounds out[I...] = iszero(β) ? acc : muladd(β, out[I...], acc)
    return nothing
end

function gmg_apply_tables!(out, input, β, index, weight)
    @parallel (@idx size(out)) _gmg_transfer!(out, input, convert(eltype(out), β), index, weight)
    return nothing
end

"""
    gmg_prolong!(fine, coarse, transfer; β = 1)

Add the prolongated coarse correction to `fine` (`fine = β fine + P coarse`) on
owned, retained DOFs. Coarse values read across a rank boundary must already
hold valid halos; fine halos and velocity ghosts are not updated.
"""
function gmg_prolong!(fine::JustRelax.GMGCoupledFields{N}, coarse::JustRelax.GMGCoupledFields{N}, transfer::JustRelax.GMGTransfer{N}; β = 1) where {N}
    gmg_apply_tables!(fine.p, coarse.p, β, map(t -> t.prolong_index, transfer.pressure), map(t -> t.prolong_weight, transfer.pressure))
    ntuple(Val(N)) do C
        tables = JustRelax.gmg_component_tables(transfer, Val(C))
        gmg_apply_tables!(fine.v[C], coarse.v[C], β, map(t -> t.prolong_index, tables), map(t -> t.prolong_weight, tables))
    end
    return nothing
end

"""
    gmg_restrict!(coarse, fine, transfer)

Overwrite `coarse` with `M_c⁻¹ Pᵀ M_f fine`, the measure-weighted transpose of
[`gmg_prolong!`](@ref), on owned retained coarse DOFs; every other coarse entry
is set to zero. Fine values read across a rank boundary must hold valid halos.
"""
function gmg_restrict!(coarse::JustRelax.GMGCoupledFields{N}, fine::JustRelax.GMGCoupledFields{N}, transfer::JustRelax.GMGTransfer{N}) where {N}
    gmg_apply_tables!(coarse.p, fine.p, 0, map(t -> t.restrict_index, transfer.pressure), map(t -> t.restrict_weight, transfer.pressure))
    ntuple(Val(N)) do C
        tables = JustRelax.gmg_component_tables(transfer, Val(C))
        gmg_apply_tables!(coarse.v[C], fine.v[C], 0, map(t -> t.restrict_index, tables), map(t -> t.restrict_weight, tables))
    end
    return nothing
end
