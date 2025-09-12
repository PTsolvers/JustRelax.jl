@parallel_indices (I...) function visc_AD!(
        η,
        ν, 
        ratios_center,
        εxx, 
        εyy, 
        εxyv, 
        args, 
        rheology,
        air_phase::Integer,
        cutoff,
        Sens,
    )
    
    # convenience closure
    @inline gather(A) = _gather(A, I...)

    @inbounds begin

        # cache dislocation parameters
        dis = (Sens[9][I...], Sens[10][I...], Sens[11][I...], Sens[12][I...], Sens[13][I...], Sens[19][I...], Sens[20][I...],Sens[14][I...],Sens[15][I...],Sens[16][I...],Sens[17][I...],Sens[18][I...])

        # cache
        ε = εxx[I...], εyy[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        εII_0 = allzero(ε...) * eps()

        # argument fields at local index
        args_ij = local_viscosity_args(args, I...)

        # local phase ratio
        ratio_ij = @cell ratios_center[I...]
        # remove phase ratio of the air if necessary & normalize ratios
        ratio_ij = correct_phase_ratioAD(air_phase, ratio_ij)

        # compute second invariant of strain rate tensor
        εij = εII_0 + ε[1], -εII_0 + ε[2], gather(εxyv)
        εII = second_invariant(εij...)

        # compute and update stress viscosity
        ηi = compute_phase_viscosity_εIIAD(rheology, ratio_ij, εII, dis, args_ij)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

@inline function local_viscosity_argsAD(args, I::Vararg{Integer, N}) where {N}
    v = getindex.(values(args), I...)
    local_args = (; zip(keys(args), v)..., dt = args.dt, τII_old = 0.0)
    return local_args
end

function correct_phase_ratioAD(air_phase, ratio::SVector{N, T}) where {N, T}
    if iszero(air_phase)
        return ratio
    elseif ratio[air_phase] ≈ 1
        return SVector{N, T}(zero(T) for _ in 1:N)
    else
        mask = ntuple(i -> (i !== air_phase), Val(N))
        # set air phase ratio to zero
        corrected_ratio = ratio .* mask
        # normalize phase ratios without air
        return corrected_ratio ./ sum(corrected_ratio)
    end
end

@generated function compute_phase_viscosity_εIIAD(
        rheology::NTuple{N, AbstractMaterialParamsStruct}, ratio, εII, dis, args
    ) where {N}
    return quote
        Base.@_inline_meta
        η = 0.0
        Base.@nexprs $N i -> (
            η += if iszero(ratio[i])
                0.0
            else
                inv(compute_viscosity_εIIAD(rheology[i].CompositeRheology[1], εII, dis, args)) * ratio[i]
            end
        )
        inv(η)
    end
end


#### GeoParams
@inline elementsAD(v::Union{CompositeRheology, Parallel}) = v.elements

@inline iselasticAD(v::AbstractElasticity) = true
@inline iselasticAD(v) = false

#@inline isplastic(x::AbstractPlasticity) = true
#@inline isplastic(x) = false


######## Utils
@inline precisionAD(::AbstractConstitutiveLaw{T}) where {T} = T

# fast exponential
@inline fastpowAD(x::Number, n::Integer) = x^n

@inline function fastpowAD(x::Number, n::AbstractFloat)
    isinteger(n) && return x^Int(n)
    x > 0 && return exp(log(x) * n)
    return x^n
end

@inline function fastpowAD(x::Quantity, n::AbstractFloat)
    isinteger(n) && return x^Int(n)
    return x^n
end

@inline function pow_checkAD(x::T, n) where {T}
    return if isone(x) || isone(n)
        x
    elseif iszero(n)
        one(T)
    else
        fastpowAD(x, n)
    end
end

macro powAD(ex)
    substitute_walkAD(ex)
    return esc(:($ex))
end

@inline function substitute_walkAD(ex::Expr)
    for (i, arg) in enumerate(ex.args)
        new_arg = substitute_walkAD(arg)
        if !isnothing(new_arg)
            ex.args[i] = new_arg
        end
    end
    return
end
@inline substitute_walkAD(sym::Symbol) = sym == :(^) ? :(pow_checkAD) : sym
@inline substitute_walkAD(x) = x

########

for fn in (:compute_viscosity_εIIAD,)
    @eval begin

        #=
        @inline function $fn(v::AbstractConstitutiveLaw, xx, yy, xy, args)
            II = second_invariant(xx, yy, xy)
            η = $fn(v, II, args)
            return η
        end

        # For single phase versions MaterialParams
        @inline $fn(v::MaterialParams, args::Vararg{Any, N}) where {N} = $fn(v.CompositeRheology[1], args...)
=#
        # compute effective "creep" viscosity from strain rate tensor given a composite rheology
        @inline function $fn(v::CompositeRheology, II, dis, args::Vararg{T, N}) where {T, N}
            e = elementsAD(v)
            return compute_viscosity_IIAD(e, $fn, II, dis, args...)
        end

    end
end

# compute effective "creep" for a composite rheology where elements are in series
@generated function compute_viscosity_IIAD(v::NTuple{N, AbstractConstitutiveLaw}, fn::F, II::T, dis, args) where {F, N, T}
    return quote
        Base.@_inline_meta
        η = zero(T)
        Base.@nexprs $N i -> (
            v_i = v[i];
            !isplastic(v_i) && !iselasticAD(v_i) && (η += inv(fn(v_i, II, dis, args)))
        )
        return inv(η)
    end
end

#@inline compute_τIIAD(a::AbstractConstitutiveLaw, EpsII, dis, args) = #compute_τIIAD(a, EpsII, dis; args...)

@inline function compute_viscosity_εIIAD(v::AbstractConstitutiveLaw, εII, dis, args)
    #τII = compute_τIIAD(v, εII, dis, args)
    τII = compute_τIIAD(v, εII, dis)
    η = _viscosityAD(τII, εII)
    return η
end

@inline _viscosityAD(τII, εII) = τII / (2 * εII)


@inline function compute_τIIAD(a::LinearViscous, EpsII, dis; kwargs...)
    @unpack_val η = a

    return 2 * (η * EpsII)
end

@inline function compute_τIIAD(
    a::DislocationCreep,
    EpsII,
    dis;
    T = one(precisionAD(a)),
    P = zero(precisionAD(a)),
    f = one(precisionAD(a)),
    #args...,
)
n, r, A, E, V, R = if EpsII isa Quantity
    @unpack_units n, r, A, E, V, R = a
    n, r, A, E, V, R
else
    @unpack_val n, r, A, E, V, R = a
    n, r, A, E, V, R
end

FT, FE = a.FT, a.FE

A  = dis[1]
n  = dis[2]
r  = dis[3]
E  = dis[4]
V  = dis[5]
Pc = dis[6]
Tc = dis[7]

_n = inv(n)

return @powAD A^-_n * (EpsII * FE)^_n * f^(-r * _n) * exp((E + Pc * V) / (n * R * Tc)) / FT
end


@inline function compute_τIIAD(
    a::DiffusionCreep,
    EpsII,
    dis;
    T = one(precisionAD(a)),
    P = zero(precisionAD(a)),
    f = one(precisionAD(a)),
    d = one(precisionAD(a)),
    #kwargs...,
)
@unpack_val n, r, p, A, E, V, R = a
FT, FE = a.FT, a.FE

A  = dis[8]
p  = dis[9]
r  = dis[10]
E  = dis[11]
V  = dis[12]
Pc = dis[6]
Tc = dis[7]

n_inv = inv(n)

τ = @powAD A^-n_inv *
    (EpsII * FE)^n_inv *
    f^(-r * n_inv) *
    d^(-p * n_inv) *
    exp((E + Pc * V) / (n * R * Tc)) / FT

return τ
end