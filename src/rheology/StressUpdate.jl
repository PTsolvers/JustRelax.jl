# inner kernel to compute the plastic stress update within Pseudo-Transient stress continuation
function _compute_τ_nonlinear!(
        τ::NTuple{N1, T},
        τII,
        τ_old::NTuple{N1, T},
        ε::NTuple{N1, T},
        ε_pl::NTuple{N1, T},
        P,
        ηij,
        η_vep,
        λ,
        dτ_r,
        _Gdt,
        plastic_parameters,
        I::Vararg{Integer, N2},
    ) where {N1, N2, T}

    # cache tensors
    τij, τij_o, εij, εij_pl = cache_tensors(τ, τ_old, ε, ε_pl, I...)

    # Stress increment and trial stress
    dτij, τII_trial = compute_stress_increment_and_trial(τij, τij_o, ηij, εij, _Gdt, dτ_r)

    # visco-elastic strain rates
    εij_ve = ntuple(i -> fma(0.5 * τij_o[i], _Gdt, εij[i]), Val(N1))

    # get plastic parameters (if any...)
    (; is_pl, C, sinϕ, cosϕ, η_reg, volume) = plastic_parameters

    # yield stress (GeoParams could be used here...)
    τy = @inbounds max(C * cosϕ + P[I...] * sinϕ, 0)

    # check if yielding; if so, compute plastic strain rate (λdQdτ),
    # plastic stress increment (dτ_pl), and update the plastic
    # multiplier (λ)
    dτij, λdQdτ = if isyielding(is_pl, τII_trial, τy)
        # derivatives plastic stress correction
        dτ_pl, λ[I...], λdQdτ = compute_dτ_pl(
            τij, dτij, τy, τII_trial, ηij, λ[I...], η_reg, dτ_r, volume
        )
        dτ_pl, λdQdτ

    else
        # in this case the plastic strain rate is a tuples of zeros
        dτij, ntuple(_ -> zero(eltype(T)), Val(N1))
    end

    # fill plastic strain rate tensor
    update_plastic_strain_rate!(ε_pl, λdQdτ, I)
    # update and correct stress
    correct_stress!(τ, τij .+ dτij, I...)

    τII[I...] = τII_ij = second_invariant((τij .+ dτij)...)
    η_vep[I...] = τII_ij * 0.5 * inv(second_invariant(εij...))

    return nothing
end

# fill plastic strain rate tensor
@generated function update_plastic_strain_rate!(ε_pl::NTuple{N, T}, λdQdτ, I) where {N, T}
    return quote
        Base.@inline
        Base.@nexprs $N i -> ε_pl[i][I...] = !isinf(λdQdτ[i]) * λdQdτ[i]
    end
end

# check if plasticity is active
@inline isyielding(is_pl, τII_trial, τy) = is_pl * (τII_trial > τy)

@inline compute_dτ_r(θ_dτ, ηij, _Gdt) = inv(θ_dτ + fma(ηij, _Gdt, 1.0))

function compute_stress_increment_and_trial(
        τij::NTuple{N, T}, τij_o::NTuple{N, T}, ηij, εij::NTuple{N, T}, _Gdt, dτ_r
    ) where {N, T}
    dτij = ntuple(Val(N)) do i
        Base.@inline
        return dτ_r *
            fma(2.0 * ηij, εij[i], fma(-((τij[i] - τij_o[i])) * ηij, _Gdt, -τij[i]))
    end
    return dτij, second_invariant((τij .+ dτij)...)
end

function compute_dτ_pl(
        τij::NTuple{N, T}, dτij, τy, τII_trial, ηij, λ0, η_reg, dτ_r, volume
    ) where {N, T}
    # yield function
    F = τII_trial - τy
    # Plastic multiplier
    ν = 0.5
    λ = ν * λ0 + (1 - ν) * (F > 0.0) * F * inv(ηij * dτ_r + η_reg + volume)
    λ_τII = λ * 0.5 * inv(τII_trial)

    λdQdτ = ntuple(Val(N)) do i
        Base.@inline
        # derivatives of the plastic potential
        return (τij[i] + dτij[i]) * λ_τII
    end

    dτ_pl = ntuple(Val(N)) do i
        Base.@inline
        # corrected stress
        return fma(-dτ_r * 2.0, ηij * λdQdτ[i], dτij[i])
    end
    return dτ_pl, λ, λdQdτ
end

# update the global arrays τ::NTuple{N, AbstractArray} with the local τij::NTuple{3, Float64} at indices I::Vararg{Integer, N}
@generated function correct_stress!(
        τ, τij::NTuple{N1}, I::Vararg{Integer, N2}
    ) where {N1, N2}
    return quote
        Base.@inline
        Base.@nexprs $N1 i -> τ[i][I...] = τij[i]
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

@inline plastic_params(v) = plastic_params(v.CompositeRheology[1].elements, 0.0e0)
@inline plastic_params(v, EII) = plastic_params(v.CompositeRheology[1].elements, EII)

@generated function plastic_params(v::NTuple{N, Any}, EII) where {N}
    return quote
        Base.@inline
        Base.@nexprs $N i -> begin
            vᵢ = v[i]
            if isplastic(vᵢ)
                C = soften_cohesion(vᵢ, EII)
                sinϕ, cosϕ = soften_friction_angle(vᵢ, EII)
                return (true, C, sinϕ, cosϕ, vᵢ.sinΨ.val, vᵢ.η_vp.val)
            end
        end
        (false, 0.0, 0.0, 0.0, 0.0, 0.0)
    end
end

function plastic_params_phase(
        rheology::NTuple{N, AbstractMaterialParamsStruct}, EII, ratio, kwargs
    ) where {N}
    return plastic_params_phase(rheology, EII, ratio; kwargs...)
end

function plastic_params_phase(
        rheology::NTuple{N, AbstractMaterialParamsStruct},
        EII,
        ratio;
        perturbation_C = nothing,
        kwargs...,
    ) where {N}
    @inline perturbation(::Nothing) = 1.0
    @inline perturbation(x::Real) = x

    data = _plastic_params_phase(rheology, EII, ratio)
    # average over phases
    is_pl = false
    C = sinϕ = cosϕ = sinψ = η_reg = 0.0
    for n in 1:N
        data_n = data[n] .* ratio[n]
        data[n][1] && (is_pl = true)
        C += data_n[2] * perturbation(perturbation_C)
        sinϕ += data_n[3]
        cosϕ += data_n[4]
        sinψ += data_n[5]
        η_reg += data_n[6]
    end
    return is_pl, C, sinϕ, cosϕ, sinψ, η_reg
end

@generated function _plastic_params_phase(
        rheology::NTuple{N, AbstractMaterialParamsStruct}, EII, ratio
    ) where {N}
    return quote
        Base.@inline
        empty_args = false, 0.0, 0.0, 0.0, 0.0, 0.0
        Base.@nexprs $N i ->
        a_i = ratio[i] == 0 ? empty_args : plastic_params(rheology[i], EII)
        Base.@ncall $N tuple a
    end
end

@inline function cache_tensors(
        τ::NTuple{3, Any}, τ_old::NTuple{3, Any}, ε::NTuple{3, Any}, I::Vararg{Integer, 2}
    )
    Base.@propagate_inbounds @inline av_shear(A) = sum(_gather(A, I...)) / 4

    # unpack
    εxx, εyy, εxy = ε
    τxx, τyy, τxy = τ
    τxx_old, τyy_old, τxy_old = τ_old
    # index
    εij = @inbounds εxx[I...], εyy[I...], av_shear(εxy)
    τij = @inbounds τxx[I...], τyy[I...], τxy[I...]
    τij_o = @inbounds τxx_old[I...], τyy_old[I...], τxy_old[I...]

    return τij, τij_o, εij
end

@inline function cache_tensors(
        τ::NTuple{3, Any}, τ_old::NTuple{3, Any}, ε::NTuple{3, Any}, ε_pl::NTuple{3, Any}, I::Vararg{Integer, 2}
    )
    Base.@propagate_inbounds @inline av_shear(A) = sum(_gather(A, I...)) / 4

    # unpack
    εxx, εyy, εxy = ε
    εxx_pl, εyy_pl, εxy_pl = ε_pl
    τxx, τyy, τxy = τ
    τxx_old, τyy_old, τxy_old = τ_old
    # index
    εij = @inbounds εxx[I...], εyy[I...], av_shear(εxy)
    εij_pl = @inbounds εxx_pl[I...], εyy_pl[I...], av_shear(εxy_pl)
    τij = @inbounds τxx[I...], τyy[I...], τxy[I...]
    τij_o = @inbounds τxx_old[I...], τyy_old[I...], τxy_old[I...]

    return τij, τij_o, εij, εij_pl
end

@inline function cache_tensors(
        τ::NTuple{3, Any}, τ_old::NTuple{3, Any}, ε::NTuple{3, Any}, ε_pl::NTuple{3, Any}, Δε::NTuple{3, Any}, I::Vararg{Integer, 2}
    )
    Base.@propagate_inbounds @inline av_shear(A) = sum(_gather(A, I...)) / 4

    # unpack
    εxx, εyy, εxy = ε
    εxx_pl, εyy_pl, εxy_pl = ε_pl
    Δεxx, Δεyy, Δεxy = Δε
    τxx, τyy, τxy = τ
    τxx_old, τyy_old, τxy_old = τ_old
    # index
    εij = @inbounds εxx[I...], εyy[I...], av_shear(εxy)
    εij_pl = @inbounds εxx_pl[I...], εyy_pl[I...], av_shear(εxy_pl)
    Δεij = @inbounds Δεxx[I...], Δεyy[I...], av_shear(Δεxy)
    τij = @inbounds τxx[I...], τyy[I...], τxy[I...]
    τij_o = @inbounds τxx_old[I...], τyy_old[I...], τxy_old[I...]

    return τij, τij_o, εij, εij_pl, Δεij
end


@inline function cache_tensors(
        τ::NTuple{6, Any}, τ_old::NTuple{6, Any}, ε::NTuple{6, Any}, I::Vararg{Integer, 3}
    )
    Base.@propagate_inbounds @inline av_yz(A) = _av_yz(A, I...)
    Base.@propagate_inbounds @inline av_xz(A) = _av_xz(A, I...)
    Base.@propagate_inbounds @inline av_xy(A) = _av_xy(A, I...)

    # unpack
    εxx, εyy, εzz, εyz, εxz, εxy = ε
    τxx, τyy, τzz, τyz, τxz, τxy = τ
    τxx_old, τyy_old, τzz_old, τyz_old, τxz_old, τxy_old = τ_old

    # normal components of the strain rate and old-stress tensors
    ε_normal = @inbounds εxx[I...], εyy[I...], εzz[I...]
    # shear components of the strain rate and old-stress tensors
    ε_shear = @inbounds av_yz(εyz), av_xz(εxz), av_xy(εxy)
    # cache ij-th components of the tensors into a tuple in Voigt notation
    εij = (ε_normal..., ε_shear...)
    τij = @inbounds τxx[I...], τyy[I...], τzz[I...], τyz[I...], τxz[I...], τxy[I...]
    τij_o = @inbounds τxx_old[I...], τyy_old[I...], τzz_old[I...], τyz_old[I...], τxz_old[I...], τxy_old[I...]

    return τij, τij_o, εij
end

@inline function cache_tensors(
        τ::NTuple{6, Any}, τ_old::NTuple{6, Any}, ε::NTuple{6, Any}, ε_pl::NTuple{6, Any}, I::Vararg{Integer, 3}
    )
    Base.@propagate_inbounds @inline av_yz(A) = _av_yz(A, I...)
    Base.@propagate_inbounds @inline av_xz(A) = _av_xz(A, I...)
    Base.@propagate_inbounds @inline av_xy(A) = _av_xy(A, I...)

    # unpack
    εxx, εyy, εzz, εyz, εxz, εxy = ε
    εxx_pl, εyy_pl, εzz_pl, εyz_pl, εxz_pl, εxy_pl = ε_pl
    τxx, τyy, τzz, τyz, τxz, τxy = τ
    τxx_old, τyy_old, τzz_old, τyz_old, τxz_old, τxy_old = τ_old

    # normal components of the strain rate and old-stress tensors
    ε_normal = @inbounds εxx[I...], εyy[I...], εzz[I...]
    # shear components of the strain rate and old-stress tensors
    ε_shear = @inbounds av_yz(εyz), av_xz(εxz), av_xy(εxy)

    # normal components of the plastic strain rate
    ε_pl_normal = @inbounds εxx_pl[I...], εyy_pl[I...], εzz_pl[I...]
    # shear components of the plastic strain rate
    ε_pl_shear = @inbounds av_yz(εyz_pl), av_xz(εxz_pl), av_xy(εxy_pl)
    # cache ij-th components of the tensors into a tuple in Voigt notation
    εij = (ε_normal..., ε_shear...)
    εij_pl = (ε_pl_normal..., ε_pl_shear...)
    τij = @inbounds τxx[I...], τyy[I...], τzz[I...], τyz[I...], τxz[I...], τxy[I...]
    τij_o = @inbounds τxx_old[I...], τyy_old[I...], τzz_old[I...], τyz_old[I...], τxz_old[I...], τxy_old[I...]

    return τij, τij_o, εij, εij_pl
end

## softening kernels

@inline function soften_cohesion(
        v::DruckerPrager{T, U, U1, S1, NoSoftening}, ::T
    ) where {T, U, U1, S1}
    return v.C.val
end

@inline function soften_cohesion(
        v::DruckerPrager_regularised{T, U, U1, U2, S1, NoSoftening}, ::T
    ) where {T, U, U1, U2, S1}
    return v.C.val
end

@inline function soften_cohesion(
        v::DruckerPrager{T, U, U1, S1, S2}, EII::T
    ) where {T, U, U1, S1, S2}
    return v.softening_C(EII, v.C.val)
end

@inline function soften_cohesion(
        v::DruckerPrager_regularised{T, U, U1, U2, S1, S2}, EII::T
    ) where {T, U, U1, U2, S1, S2}
    return v.softening_C(EII, v.C.val)
end

@inline function soften_friction_angle(
        v::DruckerPrager{T, U, U1, NoSoftening, S2}, ::T
    ) where {T, U, U1, S2}
    return (v.sinϕ.val, v.cosϕ.val)
end

@inline function soften_friction_angle(
        v::DruckerPrager_regularised{T, U, U1, U2, NoSoftening, S2}, ::T
    ) where {T, U, U1, U2, S2}
    return (v.sinϕ.val, v.cosϕ.val)
end

@inline function soften_friction_angle(
        v::DruckerPrager{T, U, U1, S1, S2}, EII::T
    ) where {T, U, U1, S1, S2}
    ϕ = v.softening_ϕ(EII, v.ϕ.val)
    return sincosd(ϕ)
end

@inline function soften_friction_angle(
        v::DruckerPrager_regularised{T, U, U1, U2, S1, S2}, EII::T
    ) where {T, U, U1, U2, S1, S2}
    ϕ = v.softening_ϕ(EII, v.ϕ.val)
    return sincosd(ϕ)
end
