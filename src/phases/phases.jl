"""
    nphases(x::JustRelax.PhaseRatio)

Return the number of phases in `x::JustRelax.PhaseRatio`.
"""
@inline nphases(x::JustRelax.PhaseRatio) = nphases(x.center)

@inline function nphases(
    ::CellArray{StaticArraysCore.SArray{Tuple{N},T,N1,N},N2,N3,T_Array}
) where {N,T,N1,N2,N3,T_Array}
    return Val(N)
end

@inline function numphases(
    ::CellArray{StaticArraysCore.SArray{Tuple{N},T,N1,N},N2,N3,T_Array}
) where {N,T,N1,N2,N3,T_Array}
    return N
end

"""
    fn_ratio(fn::F, rheology::NTuple{N, AbstractMaterialParamsStruct}, ratio) where {N, F}

Average the function `fn` over the material phases in `rheology` using the phase ratios `ratio`.    
"""
@generated function fn_ratio(
    fn::F, rheology::NTuple{N,AbstractMaterialParamsStruct}, ratio
) where {N,F}
    quote
        Base.@_inline_meta
        x = 0.0
        Base.@nexprs $N i -> x += iszero(ratio[i]) ? 0.0 : fn(rheology[i]) * ratio[i]
        return x
    end
end

@generated function fn_ratio(
    fn::F, rheology::NTuple{N,AbstractMaterialParamsStruct}, ratio, args::NamedTuple
) where {N,F}
    quote
        Base.@_inline_meta
        x = 0.0
        Base.@nexprs $N i -> x += begin
            r = ratio[i]
            isone(r) && return fn(rheology[i], args) * r
            iszero(r) ? 0.0 : fn(rheology[i], args) * r
        end
        return x
    end
end

function phase_ratios_center(phase_ratios, particles, grid, phases)
    return phase_ratios_center(backend(phase_ratios), phase_ratios, particles, grid, phases)
end

function phase_ratios_center(
    ::CPUBackendTrait, phase_ratios::JustRelax.PhaseRatio, particles, grid::Geometry, phases
)
    return _phase_ratios_center(phase_ratios, particles, grid, phases)
end

function _phase_ratios_center(
    phase_ratios::JustRelax.PhaseRatio, particles, grid::Geometry, phases
)
    ni = size(phases)
    @parallel (@idx ni) phase_ratios_center_kernel(
        phase_ratios.center, particles.coords, grid.xci, grid.di, phases
    )
    return nothing
end

@parallel_indices (I...) function phase_ratios_center_kernel(
    ratio_centers, pxi::NTuple{N,T1}, xci::NTuple{N,T2}, di::NTuple{N,T3}, phases
) where {N,T1,T2,T3}

    # index corresponding to the cell center
    cell_center = ntuple(i -> xci[i][I[i]], Val(N))
    # phase ratios weights (∑w = 1.0)
    w = phase_ratio_weights(
        getindex.(pxi, I...), phases[I...], cell_center, di, nphases(ratio_centers)
    )
    # update phase ratios array
    for k in 1:numphases(ratio_centers)
        @cell ratio_centers[k, I...] = w[k]
    end

    return nothing
end

function phase_ratio_weights(
    pxi::NTuple{NP,C}, ph::SVector{N1,T}, cell_center, di, ::Val{NC}
) where {N1,NC,NP,T,C}

    # Initiaze phase ratio weights (note: can't use ntuple() here because of the @generated function)
    w = ntuple(_ -> zero(T), Val(NC))
    sumw = zero(T)

    for i in eachindex(ph)
        # bilinear weight (1-(xᵢ-xc)/dx)*(1-(yᵢ-yc)/dy)
        p =  getindex.(pxi, i)
        isnan(first(p)) && continue
        x = @inline bilinear_weight(cell_center, p, di)
        sumw += x # reduce
        ph_local = ph[i]
        # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
        # w = w .+ x .* ntuple(j -> δ(Int(ph_local), j), Val(NC))
        w = w .+ x .* ntuple(j -> (ph_local == j), Val(NC))
    end
    w = w .* inv(sumw)
    return w
end

@generated function bilinear_weight(
    a::NTuple{N,T}, b::NTuple{N,T}, di::NTuple{N,T}
) where {N,T}
    quote
        Base.@_inline_meta
        val = one($T)
        Base.Cartesian.@nexprs $N i ->
            @inbounds val *= muladd(-abs(a[i] - b[i]), inv(di[i]), one($T))
        return val
    end
end
