@parallel_indices (I...) function visc_ADDot!(
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
        dp,
        dM,
        AdisDot,
        ndisDot,
        rdisDot,
        EdisDot,
        VdisDot,
        Pdot,
        Tdot,
        visc,
    )
    
    # convenience closure
    @inline gather(A) = _gather(A, I...)

    @inbounds begin

        # cache dislocation parameters
        dis = (dp, dM[I...], AdisDot, ndisDot, rdisDot, EdisDot, VdisDot, Pdot, Tdot,visc)

        # cache
        ε = εxx[I...], εyy[I...]

        # we need strain rate not to be zero, otherwise we get NaNs
        εII_0 = allzero(ε...) * eps()

        # argument fields at local index
        args_ij = local_viscosity_args(args, I...)

        # local phase ratio
        ratio_ij = @cell ratios_center[I...]
        # remove phase ratio of the air if necessary & normalize ratios
        ratio_ij = correct_phase_ratioADDot(air_phase, ratio_ij)

        # compute second invariant of strain rate tensor
        εij = εII_0 + ε[1], -εII_0 + ε[1], gather(εxyv)
        εII = second_invariant(εij...)

        # compute and update stress viscosity
        ηi = compute_phase_viscosity_εIIADDot(rheology, ratio_ij, εII, dis, args_ij)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

@inline function local_viscosity_argsADDot(args, I::Vararg{Integer, N}) where {N}
    v = getindex.(values(args), I...)
    local_args = (; zip(keys(args), v)..., dt = args.dt, τII_old = 0.0)
    return local_args
end

function correct_phase_ratioADDot(air_phase, ratio::SVector{N, T}) where {N, T}
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

@generated function compute_phase_viscosity_εIIADDot(
        rheology::NTuple{N, AbstractMaterialParamsStruct}, ratio, εII, dis, args
    ) where {N}
    return quote
        Base.@_inline_meta
        η = 0.0
        Base.@nexprs $N i -> (
            η += if iszero(ratio[i])
                0.0
            else
                inv(compute_viscosity_εIIADDot(rheology[i].CompositeRheology[1], εII, dis, args)) * ratio[i]
            end
        )
        inv(η)
    end
end


#### GeoParams
@inline elements(v::Union{CompositeRheology, Parallel}) = v.elements

@inline iselastic(v::AbstractElasticity) = true
@inline iselastic(v) = false

#@inline isplastic(x::AbstractPlasticity) = true
#@inline isplastic(x) = false


######## Utils
@inline precision(::AbstractConstitutiveLaw{T}) where {T} = T

# fast exponential
@inline fastpow(x::Number, n::Integer) = x^n

@inline function fastpow(x::Number, n::AbstractFloat)
    isinteger(n) && return x^Int(n)
    x > 0 && return exp(log(x) * n)
    return x^n
end

@inline function fastpow(x::Quantity, n::AbstractFloat)
    isinteger(n) && return x^Int(n)
    return x^n
end

@inline function pow_check(x::T, n) where {T}
    return if isone(x) || isone(n)
        x
    elseif iszero(n)
        one(T)
    else
        fastpow(x, n)
    end
end

macro pow(ex)
    substitute_walk(ex)
    return esc(:($ex))
end

@inline function substitute_walk(ex::Expr)
    for (i, arg) in enumerate(ex.args)
        new_arg = substitute_walk(arg)
        if !isnothing(new_arg)
            ex.args[i] = new_arg
        end
    end
    return
end
@inline substitute_walk(sym::Symbol) = sym == :(^) ? :(pow_check) : sym
@inline substitute_walk(x) = x

########

for fn in (:compute_viscosity_εIIADDot,)
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
            e = elements(v)
            return compute_viscosity_IIADDot(e, $fn, II, dis, args...)
        end

    end
end

# compute effective "creep" for a composite rheology where elements are in series
@generated function compute_viscosity_IIADDot(v::NTuple{N, AbstractConstitutiveLaw}, fn::F, II::T, dis, args) where {F, N, T}
    return quote
        Base.@_inline_meta
        η = zero(T)
        Base.@nexprs $N i -> (
            v_i = v[i];
            !isplastic(v_i) && !iselastic(v_i) && (η += inv(fn(v_i, II, dis, args)))
        )
        return inv(η)
    end
end

@inline compute_τIIADDot(a::AbstractConstitutiveLaw, EpsII, dis, args) = compute_τIIADDot(a, EpsII, dis; args...)

@inline function compute_viscosity_εIIADDot(v::AbstractConstitutiveLaw, εII, dis, args)
    τII = compute_τIIADDot(v, εII, dis, args)
    #τII = compute_τIIADDot(v, εII, dis)
    η = _viscosity(τII, εII)
    return η
end

@inline _viscosity(τII, εII) = τII / (2 * εII)

@inline function compute_τIIADDot(a::LinearViscous, EpsII, dis; kwargs...)
    @unpack_val η = a

    η = η + (dis[2]*dis[1]*dis[10])

    return 2 * (η * EpsII)
end

@inline function compute_τIIADDot(
    a::DislocationCreep,
    EpsII,
    dis;
    T = one(precision(a)),
    P = zero(precision(a)),
    f = one(precision(a)),
    args...,
)
n, r, A, E, V, R = if EpsII isa Quantity
    @unpack_units n, r, A, E, V, R = a
    n, r, A, E, V, R
else
    @unpack_val n, r, A, E, V, R = a
    n, r, A, E, V, R
end

FT, FE = a.FT, a.FE
_n = inv(n)

#A  = dis[1]
#n  = dis[2]
#r  = dis[3]
#E  = dis[4]
#V  = dis[5]
#Pc = dis[6]
#Tc = dis[7]

A = A + (dis[2]*dis[1]*dis[3])
n = n + (dis[2]*dis[1]*dis[4])
r = r + (dis[2]*dis[1]*dis[5])
E = E + (dis[2]*dis[1]*dis[6])
V = V + (dis[2]*dis[1]*dis[7])
P = P + (dis[2]*dis[1]*dis[8])
T = T + (dis[2]*dis[1]*dis[9])

return @pow A^-_n * (EpsII * FE)^_n * f^(-r * _n) * exp((E + P * V) / (n * R * T)) / FT
end


@inline function compute_τIIADDot(
    a::DiffusionCreep,
    EpsII;
    T = one(precision(a)),
    P = zero(precision(a)),
    f = one(precision(a)),
    d = one(precision(a)),
    kwargs...,
)
@unpack_val n, r, p, A, E, V, R = a
FT, FE = a.FT, a.FE

n_inv = inv(n)

τ = @pow A^-n_inv *
    (EpsII * FE)^n_inv *
    f^(-r * n_inv) *
    d^(-p * n_inv) *
    exp((E + P * V) / (n * R * T)) / FT

return τ
end