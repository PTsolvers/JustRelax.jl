# inner kernel to compute the plastic stress update within Pseudo-Transient stress continuation
function _compute_τ_nonlinear!(
    τ::NTuple{N1, T},
    τII,
    τ_old::NTuple{N1, T},
    ε::NTuple{N1, T},
    P,
    ηij,
    η_vep,
    λ,
    dτ_r,
    _Gdt,
    plastic_parameters,
    idx::Vararg{Integer, N2}
) where {N1, N2, T}

    # cache tensors
    τij, τij_p_o, εij_p = cache_tensors(τ, τ_old, ε, idx...)

    # Stress increment and trial stress
    dτij, τII_trial = compute_stress_increment_and_trial(τij, τij_p_o, ηij, εij_p, _Gdt, dτ_r)

    # get plastic paremeters (if any...)
    (; is_pl, C, sinϕ, η_reg) = plastic_parameters
    Pij = P[idx...] 
    τy = C + Pij * sinϕ

    if isyielding(is_pl, τII_trial, τy, Pij) 
        # derivatives plastic stress correction
        dτ_pl, λ[idx...] = compute_dτ_pl(τij, dτij, τij_p_o, εij_p, τy, τII_trial, ηij, λ[idx...], η_reg, _Gdt, dτ_r)
        τij = τij .+ dτ_pl
        correct_stress!(τ, τij, idx...)
        # visco-elastic strain rates
        εij_ve = ntuple(Val(N1)) do i
            εij_p[i] + 0.5 * τij_p_o[i] * _Gdt
        end
        τII[idx...] = τII_ij = second_invariant(τij...)
        η_vep[idx...] = τII_ij * 0.5 * inv(second_invariant(εij_ve...))

    else
        τij = τij .+ dτij
        correct_stress!(τ, τij, idx...)
        τII[idx...] = second_invariant(τij...)
        η_vep[idx...] = ηij
    end

    return nothing
end

# check if plasticity is active
@inline isyielding(is_pl, τII_trial, τy, Pij) =  is_pl && τII_trial > τy && Pij > 0

@inline compute_dτ_r(θ_dτ, ηij, _Gdt) = inv(θ_dτ + ηij * _Gdt + 1.0) 

# cache tensors
function cache_tensors(τ::NTuple{3, Any}, τ_old::NTuple{3, Any}, ε::NTuple{3, Any}, idx::Vararg{Integer, N}) where N

    @inline av_shear(A) = 0.25 * sum(_gather(A, idx...))

    εij_p = ε[1][idx...], ε[2][idx...], av_shear(ε[3])
    τij_p_o = τ_old[1][idx...], τ_old[2][idx...], av_shear(τ_old[3])
    τij = getindex.(τ, idx...)
    
    return τij, τij_p_o, εij_p
end

function compute_stress_increment_and_trial(τij::NTuple{N, T}, τij_p_o, ηij, εij_p, _Gdt, dτ_r) where {N, T}
    dτij = ntuple(Val(N)) do i
        Base.@_inline_meta
        dτ_r * (-(τij[i] - τij_p_o[i]) * ηij * _Gdt - τij[i] + 2.0 * ηij * εij_p[i])
    end
    return dτij, second_invariant((τij .+ dτij)...)
end

function compute_dτ_pl(τij::NTuple{N, T}, dτij, τij_p_o, εij_p, τy, τII_trial, ηij, λ0, η_reg, _Gdt, dτ_r) where {N, T}
    # yield function
    F = τII_trial - τy
    # Plastic multiplier
    ν = 0.9
    λ = ν * λ0 + (1-ν) * (F > 0.0) * F * inv(ηij + η_reg)
    λ_τII = λ * 0.5 * inv(τII_trial)

    dτ_pl = ntuple(Val(N)) do i
        Base.@_inline_meta
        # derivatives of the plastic potential
        λdQdτ = (τij[i] + dτij[i]) * λ_τII
        # corrected stress
        dτ_r * (
            -(τij[i] - τij_p_o[i]) * ηij * _Gdt - τij[i] +
            2.0 * ηij * (εij_p[i] - λdQdτ)
        )
    end
    return dτ_pl, λ
end

@inline correct_stress!(τxx, τyy, τxy, τij, idx::Vararg{Integer, 2}) = correct_stress!((τxx, τyy, τxy), τij, idx...)
@inline correct_stress!(τxx, τyy, τzz, τyz, τxz, τxy, τij, idx::Vararg{Integer, 3}) = correct_stress!((τxx, τyy, τzz, τyz, τxz, τxy), τij, idx...)
@inline correct_stress!(τ, τij, idx::Vararg{Integer, 2}) = Base.@nexprs 3 i -> τ[i][idx...] = τij[i]
@inline correct_stress!(τ, τij, idx::Vararg{Integer, 3}) = Base.@nexprs 6 i -> τ[i][idx...] = τij[i]

@inline isplastic(x::AbstractPlasticity) = true
@inline isplastic(x) = false

@inline plastic_params(v) = plastic_params(v.CompositeRheology[1].elements)

@generated function plastic_params(v::NTuple{N, Any}) where N
    quote
        Base.@_inline_meta
        Base.@nexprs $N i -> isplastic(v[i]) && return true, v[i].C.val, v[i].sinϕ.val, v[i].η_vp.val
        (false, 0.0, 0.0, 0.0)
    end
end
