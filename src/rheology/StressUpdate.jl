# inner kernel to compute the plastic stress update within Pseudo-Transient stress continuation
function _compute_τ_nonlinear!(
    τ::NTuple{N1,T},
    τII,
    τ_old::NTuple{N1,T},
    ε::NTuple{N1,T},
    P,
    ηij,
    η_vep,
    λ,
    dτ_r,
    _Gdt,
    plastic_parameters,
    idx::Vararg{Integer,N2},
) where {N1,N2,T}

    # cache tensors
    τij, τij_o, εij = cache_tensors(τ, τ_old, ε, idx...)

    # Stress increment and trial stress
    dτij, τII_trial = compute_stress_increment_and_trial(τij, τij_o, ηij, εij, _Gdt, dτ_r)

    # visco-elastic strain rates
    εij_ve = ntuple(Val(N1)) do i
        @inbounds muladd(0.5 * τij_o[i], _Gdt, εij[i])
    end
    # get plastic paremeters (if any...)
    (; is_pl, C, sinϕ, cosϕ, η_reg, volume) = plastic_parameters
    τy = C * cosϕ + P[idx...] * sinϕ

    dτij = if isyielding(is_pl, τII_trial, τy)
        # derivatives plastic stress correction
        dτ_pl, λ[idx...] = compute_dτ_pl(
            τij, dτij, τy, τII_trial, ηij, λ[idx...], η_reg, dτ_r, volume
        )
        dτ_pl

    else
        dτij
    end

    τij = τij .+ dτij
    correct_stress!(τ, τij, idx...)
    τII[idx...] = τII_ij = second_invariant(τij...)
    η_vep[idx...] = τII_ij * 0.5 * inv(second_invariant(εij_ve...))

    return nothing
end

# check if plasticity is active
@inline isyielding(is_pl, τII_trial, τy) = is_pl && τII_trial > τy

@inline compute_dτ_r(θ_dτ, ηij, _Gdt) = inv(θ_dτ + muladd(ηij, _Gdt, 1.0))

function compute_stress_increment_and_trial(
    τij::NTuple{N,T}, τij_o::NTuple{N,T}, ηij, εij::NTuple{N,T}, _Gdt, dτ_r
) where {N,T}
    dτij = ntuple(Val(N)) do i
        Base.@_inline_meta
        @inbounds dτ_r * muladd(
            2.0 * ηij, εij[i], muladd(-((τij[i] - τij_o[i])) * ηij, _Gdt, -τij[i])
        )
    end
    return dτij, second_invariant((τij .+ dτij)...)
end

function compute_dτ_pl(
    τij::NTuple{N,T}, dτij, τy, τII_trial, ηij, λ0, η_reg, dτ_r, volume
) where {N,T}
    # yield function
    F = τII_trial - τy
    # Plastic multiplier
    ν = 0.5
    λ = ν * λ0 + (1 - ν) * (F > 0.0) * F * inv(ηij * dτ_r  + η_reg + volume)
    λ_τII = λ * 0.5 * inv(τII_trial)

    dτ_pl = ntuple(Val(N)) do i
        Base.@_inline_meta
        # derivatives of the plastic potential
        λdQdτ = (τij[i] + dτij[i]) * λ_τII
        # corrected stress
        muladd(-dτ_r * 2.0, ηij * λdQdτ, dτij[i])
    end
    return dτ_pl, λ
end

# update the global arrays τ::NTuple{N, AbstractArray} with the local τij::NTuple{3, Float64} at indices idx::Vararg{Integer, N}
@generated function correct_stress!(
    τ, τij::NTuple{N1,T}, idx::Vararg{Integer,N2}
) where {N1,N2,T}
    quote
        Base.@_inline_meta
        Base.@nexprs $N1 i -> @inbounds τ[i][idx...] = τij[i]
    end
end

@inline function correct_stress!(τxx, τyy, τxy, τij, i, j)
    return correct_stress!((τxx, τyy, τxy), τij, i, j)
end

@inline function correct_stress!(τxx, τyy, τzz, τyz, τxz, τxy, τij, i, j, k)
    return correct_stress!((τxx, τyy, τzz, τyz, τxz, τxy), τij, i, j, k)
end

@inline isplastic(x::AbstractPlasticity) = true
@inline isplastic(x) = false

@inline plastic_params(v) = plastic_params(v.CompositeRheology[1].elements)

@generated function plastic_params(v::NTuple{N,Any}) where {N}
    quote
        Base.@_inline_meta
        Base.@nexprs $N i ->
            isplastic(v[i]) && return true,
            v[i].C.val, v[i].sinϕ.val, v[i].cosϕ.val, v[i].sinΨ.val,
            v[i].η_vp.val
        (false, 0.0, 0.0, 0.0, 0.0, 0.0)
    end
end

function plastic_params_phase(
    rheology::NTuple{N,AbstractMaterialParamsStruct}, ratio
) where {N}
    data = _plastic_params_phase(rheology, ratio)
    # average over phases
    is_pl = false
    C = sinϕ = cosϕ = sinψ = η_reg = 0.0
    for n in 1:N
        data[n][1] && (is_pl = true)
        C += data[n][2] * ratio[n]
        sinϕ += data[n][3] * ratio[n]
        cosϕ += data[n][4] * ratio[n]
        sinψ += data[n][5] * ratio[n]
        η_reg += data[n][6] * ratio[n]
    end
    return is_pl, C, sinϕ, cosϕ, sinψ, η_reg
end

@generated function _plastic_params_phase(
    rheology::NTuple{N,AbstractMaterialParamsStruct}, ratio
) where {N}
    quote
        Base.@_inline_meta
        empty_args = false, 0.0, 0.0, 0.0, 0.0, 0.0
        Base.@nexprs $N i -> a_i = ratio[i] == 0 ? empty_args : plastic_params(rheology[i])
        Base.@ncall $N tuple a
    end
end

# cache tensors
function cache_tensors(
    τ::NTuple{3,Any}, τ_old::NTuple{3,Any}, ε::NTuple{3,Any}, idx::Vararg{Integer,2}
)
    @inline av_shear(A) = 0.25 * sum(_gather(A, idx...))

    εij = ε[1][idx...], ε[2][idx...], av_shear(ε[3])
    # τij_o = τ_old[1][idx...], τ_old[2][idx...], av_shear(τ_old[3])
    τij = getindex.(τ, idx...)
    τij_o = getindex.(τ_old, idx...)

    return τij, τij_o, εij
end

function cache_tensors(
    τ::NTuple{6,Any}, τ_old::NTuple{6,Any}, ε::NTuple{6,Any}, idx::Vararg{Integer,3}
)
    @inline av_yz(A) = _av_yz(A, idx...)
    @inline av_xz(A) = _av_xz(A, idx...)
    @inline av_xy(A) = _av_xy(A, idx...)

    # normal components of the strain rate and old-stress tensors
    ε_normal = ntuple(i -> ε[i][idx...], Val(3))
    # τ_old_normal = ntuple(i -> τ_old[i][idx...], Val3)
    # shear components of the strain rate and old-stress tensors
    ε_shear = av_yz(ε[4]), av_xz(ε[5]), av_xy(ε[6])
    # τ_old_shear = av_yz(τ_old[4]), av_xz(τ_old[5]), av_xy(τ_old[6])
    # cache ij-th components of the tensors into a tuple in Voigt notation 
    εij = (ε_normal..., ε_shear...)
    # τij_o = (τ_old_normal..., τ_old_shear...)
    τij_o = getindex.(τ_old, idx...)
    τij = getindex.(τ, idx...)

    return τij, τij_o, εij
end
