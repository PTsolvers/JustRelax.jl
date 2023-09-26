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
    εij_ve = @. εij + 0.5 * τij_o * _Gdt

    # get plastic paremeters (if any...)
    (; is_pl, C, sinϕ, η_reg) = plastic_parameters
    τy = C + P[idx...] * sinϕ

    dτij = if isyielding(is_pl, τII_trial, τy)
        # derivatives plastic stress correction
        dτ_pl, λ[idx...] = compute_dτ_pl(
            τij, dτij, τij_o, εij, τy, τII_trial, ηij, λ[idx...], η_reg, _Gdt, dτ_r
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

@inline compute_dτ_r(θ_dτ, ηij, _Gdt) = inv(θ_dτ + ηij * _Gdt + 1.0)

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
    @inline av_yz(A) = 0.125 * sum(_gather_yz(A, idx...))
    @inline av_xz(A) = 0.125 * sum(_gather_xz(A, idx...))
    @inline av_xy(A) = 0.125 * sum(_gather_xy(A, idx...))

    Val3 = Val(3)

    # normal components of the strain rate and old-stress tensors
    ε_normal = ntuple(i -> ε[i][idx...], Val3)
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

function compute_stress_increment_and_trial(
    τij::NTuple{N,T}, τij_o::NTuple{N,T}, ηij, εij::NTuple{N,T}, _Gdt, dτ_r
) where {N,T}
    dτij = ntuple(Val(N)) do i
        Base.@_inline_meta
        @inbounds dτ_r * (-(τij[i] - τij_o[i]) * ηij * _Gdt - τij[i] + 2.0 * ηij * εij[i])
    end
    return dτij, second_invariant((τij .+ dτij)...)
end

function compute_dτ_pl(
    τij::NTuple{N,T}, dτij, τij_o, εij, τy, τII_trial, ηij, λ0, η_reg, _Gdt, dτ_r
) where {N,T}
    # yield function
    F = τII_trial - τy
    # Plastic multiplier
    ν = 1e-2
    # λ = ν * λ0 + (1 - ν) * (F > 0.0) * F * inv(ηij + η_reg)
    # λ = (F > 0.0) * F * inv(ηij + η_reg)
    # λ = (F > 0.0) * F * inv(dτ_r + η_reg)
    λ = ν * λ0 + (1 - ν) * (F > 0.0) * F * inv(dτ_r + η_reg)
    λ_τII = λ * 0.5 * inv(τII_trial)

    dτ_pl = ntuple(Val(N)) do i
        Base.@_inline_meta
        # derivatives of the plastic potential
        λdQdτ = (τij[i] + dτij[i]) * λ_τII
        # corrected stress
        # dτ_r * (-(τij[i] - τij_o[i]) * ηij * _Gdt - τij[i] + 2.0 * ηij * (εij[i] - λdQdτ))
        dτij[i] - dτ_r * 2.0 * ηij * λdQdτ
    end
    return dτ_pl, λ
end

# update the global arrays τ::NTuple{N, AbstractArray} with the local τij::NTuple{3, Float64} at indices idx::Vararg{Integer, N}
@inline function correct_stress!(τ, τij, idx::Vararg{Integer,2})
    Base.@nexprs 3 i -> τ[i][idx...] = τij[i]
end
@inline function correct_stress!(τ, τij, idx::Vararg{Integer,3})
    Base.@nexprs 6 i -> τ[i][idx...] = τij[i]
end
@inline function correct_stress!(τxx, τyy, τxy, τij, idx::Vararg{Integer,2})
    return correct_stress!((τxx, τyy, τxy), τij, idx...)
end
@inline function correct_stress!(τxx, τyy, τzz, τyz, τxz, τxy, τij, idx::Vararg{Integer,3})
    return correct_stress!((τxx, τyy, τzz, τyz, τxz, τxy), τij, idx...)
end

@inline isplastic(x::AbstractPlasticity) = true
@inline isplastic(x) = false

@inline plastic_params(v) = plastic_params(v.CompositeRheology[1].elements)

@generated function plastic_params(v::NTuple{N,Any}) where {N}
    quote
        Base.@_inline_meta
        Base.@nexprs $N i ->
            isplastic(v[i]) && return true, v[i].C.val, v[i].sinϕ.val, v[i].η_vp.val
        (false, 0.0, 0.0, 0.0)
    end
end

function plastic_params_phase(
    rheology::NTuple{N,AbstractMaterialParamsStruct}, ratio
) where {N}
    data = _plastic_params_phase(rheology, ratio)
    # average over phases
    is_pl = false
    C = 0.0
    sinϕ = 0.0
    η_reg = 0.0
    for n in 1:N
        data[n][1] && (is_pl = true)
        C += data[n][2] * ratio[n]
        sinϕ += data[n][3] * ratio[n]
        η_reg += data[n][4] * ratio[n]
    end
    return is_pl, C, sinϕ, η_reg
end

@generated function _plastic_params_phase(
    rheology::NTuple{N,AbstractMaterialParamsStruct}, ratio
) where {N}
    quote
        Base.@_inline_meta
        empty_args = false, 0.0, 0.0, 0.0
        Base.@nexprs $N i -> a_i = ratio[i] == 0 ? empty_args : plastic_params(rheology[i])
        Base.@ncall $N tuple a
    end
end
