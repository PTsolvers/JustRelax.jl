
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
    num_phases = Val(nphases(x))
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
        Base.@nexprs $N1 i -> reps_i = (c = 0;
        for j in cellaxes(phases)
            c += @cell(phases[j, cell...]) == i
        end;
        c * _n)
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
        # Base.@nexprs $N i -> x += !(ratio[i] == 0) * fn(rheology[i]) * ratio[i]
        Base.@nexprs $N i -> x += ratio[i] == 0 ? 0.0 : fn(rheology[i]) * ratio[i]
        return x * inv($N)
    end
end

@generated function fn_ratio(
    fn::F, rheology::NTuple{N,AbstractMaterialParamsStruct}, ratio, args::NamedTuple
) where {N,F}
    quote
        Base.@_inline_meta
        x = 0.0
        # Base.@nexprs $N i -> x += !(ratio[i] == 0) * fn(rheology[i], args) * ratio[i]
        Base.@nexprs $N i -> x += ratio[i] == 0 ? 0.0 : fn(rheology[i], args) * ratio[i]
        return x * inv($N)
    end
end
