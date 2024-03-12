
struct Phases{T}
    vertex::T
    center::T
end

struct PhaseRatio{T}
    vertex::T
    center::T

    function PhaseRatio(ni, num_phases)
        center = @fill(0.0, ni..., celldims = (num_phases,))
        vertex = @fill(0.0, ni .+ 1..., celldims = (num_phases,))
        T = typeof(center)
        return new{T}(vertex, center)
    end
end

"""
    nphases(x::PhaseRatio)

Return the number of phases in `x::PhaseRatio`.
"""
@inline nphases(x::PhaseRatio) = nphases(x.center)
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

# ParallelStencil launch kernel for 2D
@parallel_indices (i, j) function phase_ratios_center(x, phases)
    phase_ratios_center(x, phases, i, j)
    return nothing
end

# ParallelStencil launch kernel for 3D
@parallel_indices (i, j, k) function phase_ratios_center(x, phases)
    phase_ratios_center(x, phases, i, j, k)
    return nothing
end

"""
    phase_ratios_center(x::PhaseRatio, cell::Vararg{Int, N})

Compute the phase ratios at the center of the cell `cell` in `x::PhaseRatio`.
"""
function phase_ratios_center(x::PhaseRatio, phases, cell::Vararg{Int,N}) where {N}
    return phase_ratios_center(x.center, phases, cell...)
end

@inline function phase_ratios_center(x::CellArray, phases, cell::Vararg{Int,N}) where {N}
    # total number of material phases
    num_phases = nphases(x)
    # number of active particles in this cell
    n = 0
    for j in cellaxes(phases)
        n += isinteger(@cell(phases[j, cell...])) && @cell(phases[j, cell...]) != 0
    end
    _n = inv(n)
    # compute phase ratios
    ratios = _phase_ratios_center(phases, num_phases, _n, cell...)
    for (i, ratio) in enumerate(ratios)
        @cell x[i, cell...] = ratio
    end
end

@generated function _phase_ratios_center(
    phases, ::Val{N1}, _n, cell::Vararg{Int,N2}
) where {N1,N2}
    quote
        Base.@_inline_meta
        Base.@nexprs $N1 i -> reps_i = begin
            c = 0
            for j in cellaxes(phases)
                c += @cell(phases[j, cell...]) == i
            end
            c * _n
        end
        Base.@ncall $N1 tuple reps
    end
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

# ParallelStencil launch kernel for 2D

function phase_ratios_center!(phase_ratios::PhaseRatio, particles, xci, di, pPhases)
    ni = size(pPhases)
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, particles.coords, xci, di, pPhases)
    return nothing
end

@parallel_indices (I...) function phase_ratios_center_kernel(
    ratio_centers, pxi::NTuple{N,T1}, xci::NTuple{N,T2}, di::NTuple{N,T3}, phases
) where {N,T1,T2,T3}

    # index corresponding to the cell center
    cell_center = ntuple(i -> xci[i][I[i]], Val(N))
    # phase ratios weights (∑w = 1.0)
    w = phase_ratio_weights(
        getindex.(pxi, I...),
        phases[I...],
        cell_center,
        di,
        JustRelax.nphases(ratio_centers),
    )
    # update phase ratios array
    for k in 1:numphases(ratio_centers)
        JustRelax.@cell ratio_centers[k, I...] = w[k]
    end

    return nothing
end

function phase_ratio_weights(
    pxi::NTuple{NP,C}, ph::SVector{N1,T}, cell_center, di, ::Val{NC}
) where {N1,NC,NP,T,C}
    if @generated
        quote
            Base.@_inline_meta
            # Initiaze phase ratio weights (note: can't use ntuple() here because of the @generated function)
            Base.@nexprs $NC i -> w_i = zero($T)
            w = Base.@ncall $NC tuple w

            # initialie sum of weights
            sumw = zero($T)
            Base.@nexprs $N1 i -> begin
                # bilinear weight (1-(xᵢ-xc)/dx)*(1-(yᵢ-yc)/dy)
                x = bilinear_weight(cell_center, getindex.(pxi, i), di)
                sumw += x # reduce
                ph_local = ph[i]
                # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
                # Base.@nexprs $NC j -> tmp_j = w[j] + x * δ(Int(ph_local), j)
                Base.@nexprs $NC j -> tmp_j = w[j] + x * (ph_local == j)
                w = Base.@ncall $NC tuple tmp
            end

            # return phase ratios weights w = sum(w * δij(i, phase)) / sum(w)
            _sumw = inv(sum(w))
            Base.@nexprs $NC i -> w_i = w[i] * _sumw
            w = Base.@ncall $NC tuple w
            return w
        end
    else
        # Initiaze phase ratio weights (note: can't use ntuple() here because of the @generated function)
        w = ntuple(_ -> zero(T), Val(NC))
        # initialie sum of weights
        sumw = zero(T)

        for i in eachindex(pxi)
            # bilinear weight (1-(xᵢ-xc)/dx)*(1-(yᵢ-yc)/dy)
            x = @inline bilinear_weight(cell_center, getindex.(pxi, i), di)
            sumw += x # reduce
            ph_local = ph[i]
            # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
            # w = w .+ x .* ntuple(j -> δ(Int(ph_local), j), Val(NC))
            w = w .+ x .* ntuple(j -> (ph_local == j), Val(NC))
        end
        w = w .* inv(sum(w))
        return w
    end
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
