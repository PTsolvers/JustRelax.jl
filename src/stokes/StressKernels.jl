# Viscous
function compute_stress_increment(τij::Real, τij_o::Real, ηij, εij::Real, _Gdt, dτ_r)
    dτij = dτ_r * fma(2.0 * ηij, εij, fma(-(τij - τij_o) * ηij, _Gdt, -τij))
    return dτij
end

function compute_stress_increment(
        τij::NTuple{N}, τij_o::NTuple{N}, ηij, εij::NTuple{N}, _Gdt, dτ_r
    ) where {N}
    dτij = ntuple(Val(N)) do i
        Base.@_inline_meta
        return dτ_r *
            fma(2.0 * ηij, εij[i], fma(-((τij[i] - τij_o[i])) * ηij, _Gdt, -τij[i]))
    end
    return dτij
end

function compute_stress_increment(τij::Real, τij_o::Real, ηij, Δεij::Real, _G, dτ_r, dt)
    dτij = dτ_r * fma(2.0 * ηij, Δεij, fma(-(τij - τij_o) * ηij, _G, -τij * dt))
    return dτij
end

function compute_stress_increment(
        τij::NTuple{N}, τij_o::NTuple{N}, ηij, Δεij::NTuple{N}, _G, dτ_r, dt
    ) where {N}
    dτij = ntuple(Val(N)) do i
        Base.@_inline_meta
        return dτ_r *
            fma(2.0 * ηij, Δεij[i], fma(-((τij[i] - τij_o[i])) * ηij, _G, -τij[i] * dt))
    end
    return dτij
end

@parallel_indices (i, j) function compute_τ!(
        τxx::AbstractArray{T, 2}, τyy, τxy, εxx, εyy, εxy, η, θ_dτ
    ) where {T}
    @inline av(A) = _av_a(A, i, j)
    @inline harm(A) = _harm_a(A, i, j)

    _Gdt = 0
    ηij = η[i, j]
    dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)

    Δτxx = compute_stress_increment(τxx[i, j], 0.0e0, ηij, εxx[i, j], _Gdt, dτ_r)
    τxx[i, j] += Δτxx

    Δτyy = compute_stress_increment(τyy[i, j], 0.0e0, ηij, εyy[i, j], _Gdt, dτ_r)
    τyy[i, j] += Δτyy

    if all((i, j) .< size(τxy) .- 1)
        ηij = av(η)
        dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)
        Δτxy = compute_stress_increment(
            τxy[i + 1, j + 1], 0.0e0, ηij, εxy[i + 1, j + 1], _Gdt, dτ_r
        )
        τxy[i + 1, j + 1] += Δτxy
    end
    return nothing
end

# Visco-elastic

@parallel_indices (i, j) function compute_τ!(
        τxx::AbstractArray{T, 2}, τyy, τxy, τxx_o, τyy_o, τxy_o, εxx, εyy, εxy, η, G, θ_dτ, dt
    ) where {T}
    @inline av(A) = _av_a(A, i, j)
    @inline harm(A) = _harm_a(A, i, j)

    _Gdt = inv(G[i, j] * dt)
    ηij = η[i, j]
    dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)

    Δτxx = compute_stress_increment(τxx[i, j], τxx_o[i, j], ηij, εxx[i, j], _Gdt, dτ_r)
    τxx[i, j] += Δτxx

    Δτyy = compute_stress_increment(τyy[i, j], τyy_o[i, j], ηij, εyy[i, j], _Gdt, dτ_r)
    τyy[i, j] += Δτyy

    if all((i, j) .< size(τxy) .- 1)
        ηij = av(η)
        _Gdt = inv(av(G) * dt)
        dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)
        Δτxy = compute_stress_increment(
            τxy[i + 1, j + 1], τxy_o[i + 1, j + 1], ηij, εxy[i + 1, j + 1], _Gdt, dτ_r
        )
        τxy[i + 1, j + 1] += Δτxy
    end

    return nothing
end

@parallel_indices (i, j) function compute_τ!(
        τxx::AbstractArray{T, 2}, # centers
        τyy, # centers
        τxy, # centers
        τxx_o, # centers
        τyy_o, # centers
        τxy_o, # centers
        εxx, # centers
        εyy, # centers
        εxy, # vertices
        η, # centers
        θ_dτ,
        dt,
        phase_center,
        rheology,
    ) where {T}
    @inline av(A) = _av_a(A, i, j)

    # Normal components
    phase = phase_center[i, j]
    _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
    η_ij = η[i, j]

    multiplier = inv(θ_dτ + η_ij * _Gdt + 1.0)

    τxx[i, j] +=
        (-(τxx[i, j] - τxx_o[i, j]) * η_ij * _Gdt - τxx[i, j] + 2.0 * η_ij * εxx[i, j]) *
        multiplier
    τyy[i, j] +=
        (-(τyy[i, j] - τyy_o[i, j]) * η_ij * _Gdt - τyy[i, j] + 2.0 * η_ij * εyy[i, j]) *
        multiplier
    τxy[i, j] +=
        (-(τxy[i, j] - τxy_o[i, j]) * η_ij * _Gdt - τxy[i, j] + 2.0 * η_ij * av(εxy)) *
        multiplier

    return nothing
end

@parallel_indices (i, j) function compute_τ_vertex!(
        τxy::AbstractArray{T, 2}, εxy, η, θ_dτ
    ) where {T}
    @inline av(A) = _av_a(A, i, j)
    @inline harm(A) = _harm_a(A, i, j)

    # Shear components
    if all((i, j) .< size(τxy) .- 1)
        I = i + 1, j + 1
        av_η_ij = harm(η)
        denominator = inv(θ_dτ + 1.0)

        τxy[I...] += (-τxy[I...] + 2.0 * av_η_ij * εxy[I...]) * denominator
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_τ!(
        τxx::AbstractArray{T, 3},
        τyy,
        τzz,
        τyz,
        τxz,
        τxy,
        τxx_o,
        τyy_o,
        τzz_o,
        τyz_o,
        τxz_o,
        τxy_o,
        εxx,
        εyy,
        εzz,
        εyz,
        εxz,
        εxy,
        η,
        G,
        dt,
        θ_dτ,
    ) where {T}
    harm_xy(A) = _harm_xyi(A, i, j, k)
    harm_xz(A) = _harm_xzi(A, i, j, k)
    harm_yz(A) = _harm_yzi(A, i, j, k)
    av_xy(A) = _av_xyi(A, i, j, k)
    av_xz(A) = _av_xzi(A, i, j, k)
    av_yz(A) = _av_yzi(A, i, j, k)
    get(x) = x[i, j, k]
    av_xy(::Nothing) = Inf
    av_xz(::Nothing) = Inf
    av_yz(::Nothing) = Inf
    get(::Nothing) = Inf

    @inbounds begin
        if all((i, j, k) .≤ size(τxx))
            _Gdt = inv(get(G) * dt)
            ηij = get(η)
            dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)

            # Compute τ_xx
            Δτxx = compute_stress_increment(get(τxx), get(τxx_o), ηij, get(εxx), _Gdt, dτ_r)
            τxx[i, j, k] += Δτxx
            # Compute τ_yy
            Δτyy = compute_stress_increment(get(τyy), get(τyy_o), ηij, get(εyy), _Gdt, dτ_r)
            τyy[i, j, k] += Δτyy
            # Compute τ_zz
            Δτzz = compute_stress_increment(get(τzz), get(τzz_o), ηij, get(εzz), _Gdt, dτ_r)
            τzz[i, j, k] += Δτzz
        end
        # Compute τ_xy
        if (1 < i < size(τxy, 1)) && (1 < j < size(τxy, 2)) && k ≤ size(τxy, 3)
            ηij = av_xy(η)
            _Gdt = inv(av_xy(G) * dt)
            dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)
            Δτxy = compute_stress_increment(
                τxy[i, j, k], τxy_o[i, j, k], ηij, εxy[i, j, k], _Gdt, dτ_r
            )
            τxy[i, j, k] += Δτxy
        end
        # Compute τ_xz
        if (1 < i < size(τxz, 1)) && j ≤ size(τxz, 2) && (1 < k < size(τxz, 3))
            ηij = av_xz(η)
            _Gdt = inv(av_xz(G) * dt)
            dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)
            Δτxz = compute_stress_increment(
                τxz[i, j, k], τxz_o[i, j, k], ηij, εxz[i, j, k], _Gdt, dτ_r
            )
            τxz[i, j, k] += Δτxz
        end
        # Compute τ_yz
        if i ≤ size(τyz, 1) && (1 < j < size(τyz, 2)) && (1 < k < size(τyz, 3))
            ηij = av_yz(η)
            _Gdt = inv(av_yz(G) * dt)
            dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)
            Δτyz = compute_stress_increment(
                τyz[i, j, k], τyz_o[i, j, k], ηij, εyz[i, j, k], _Gdt, dτ_r
            )
            τyz[i, j, k] += Δτyz
        end
    end
    return nothing
end

@parallel_indices (i, j, k) function compute_τ_vertex!(
        τyz, τxz, τxy, εyz, εxz, εxy, ηvep, θ_dτ
    )
    harm_xy(A) = _harm_xyi(A, i, j, k)
    harm_xz(A) = _harm_xzi(A, i, j, k)
    harm_yz(A) = _harm_yzi(A, i, j, k)
    av_xy(A) = _av_xyi(A, i, j, k)
    av_xz(A) = _av_xzi(A, i, j, k)
    av_yz(A) = _av_yzi(A, i, j, k)
    get(x) = x[i, j, k]

    @inbounds begin
        # Compute τ_xy
        if (1 < i < size(τxy, 1)) && (1 < j < size(τxy, 2)) && k ≤ size(τxy, 3)
            η_ij = harm_xy(ηvep)
            denominator = inv(θ_dτ + 1.0)
            τxy[i, j, k] += (-get(τxy) + 2.0 * η_ij * get(εxy)) * denominator
        end

        # Compute τ_xz
        if (1 < i < size(τxz, 1)) && j ≤ size(τxz, 2) && (1 < k < size(τxz, 3))
            η_ij = harm_xz(ηvep)
            denominator = inv(θ_dτ + 1.0)
            τxz[i, j, k] += (-get(τxz) + 2.0 * η_ij * get(εxz)) * denominator
        end
        # Compute τ_yz
        if i ≤ size(τyz, 1) && (1 < j < size(τyz, 2)) && (1 < k < size(τyz, 3))
            η_ij = harm_yz(ηvep)
            denominator = inv(θ_dτ + 1.0)
            τyz[i, j, k] += (-get(τyz) + 2.0 * η_ij * get(εyz)) * denominator
        end
    end
    return nothing
end

# Single phase visco-elasto-plastic flow

@parallel_indices (I...) function compute_τ_nonlinear!(
        τ,     # @ centers
        τII,   # @ centers
        τ_old, # @ centers
        ε,     # @ vertices
        ε_pl,  # @ centers
        EII,   # accumulated plastic strain rate @ centers
        P,
        θ,
        η,
        η_vep,
        λ,
        rheology,
        dt,
        θ_dτ,
        args,
    )

    # numerics
    ηij = η[I...]
    _Gdt = inv(get_shear_modulus(rheology[1]) * dt)
    dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)

    # get plastic parameters (if any...)
    is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(
        rheology, EII[I...], 1, ntuple_idx(args, I...)
    )

    # plastic volumetric change K * dt * sinϕ * sinψ
    K = get_bulk_modulus(rheology[1])
    volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ
    plastic_parameters = (; is_pl, C, sinϕ, cosϕ, η_reg, volume)

    _compute_τ_nonlinear!(
        τ, τII, τ_old, ε, ε_pl, P, ηij, η_vep, λ, dτ_r, _Gdt, plastic_parameters, I...
    )

    # augmented pressure with plastic volumetric strain over pressure
    θ[I...] = P[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)

    return nothing
end

# multi phase visco-elasto-plastic flow, where phases are defined in the cell center
@parallel_indices (I...) function compute_τ_nonlinear!(
        τ,      # @ centers
        τII,    # @ centers
        τ_old,  # @ centers
        ε,      # @ vertices
        ε_pl,   # @ centers
        EII,    # accumulated plastic strain rate @ centers
        P,
        θ,
        η,
        η_vep,
        λ,
        phase_center,
        rheology,
        dt,
        θ_dτ,
        args,
    )
    # numerics
    ηij = @inbounds η[I...]
    phase = @inbounds phase_center[I...]
    _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
    dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)

    # get plastic parameters (if any...)
    is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)

    # plastic volumetric change K * dt * sinϕ * sinψ
    K = fn_ratio(get_bulk_modulus, rheology, phase)
    volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ
    plastic_parameters = (; is_pl, C, sinϕ, cosϕ, η_reg, volume)

    _compute_τ_nonlinear!(
        τ, τII, τ_old, ε, ε_pl, P, ηij, η_vep, λ, dτ_r, _Gdt, plastic_parameters, I...
    )
    # augmented pressure with plastic volumetric strain over pressure
    @inbounds θ[I...] = P[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)

    return nothing
end

## Accumulate tensor
function accumulate_tensor!(II, A::JustRelax.SymmetricTensor, dt)
    return accumulate_tensor!(backend(A), II, A, dt)
end

function accumulate_tensor!(::CPUBackendTrait, II, A::JustRelax.SymmetricTensor, dt)
    _accumulate_tensor!(II, A, dt)
    return nothing
end

function _accumulate_tensor!(II, A::JustRelax.SymmetricTensor, dt)
    ni = size(II)
    @parallel (@idx ni) accumulate_tensor_kernel!(II, @tensor(A)..., dt)
    return nothing
end

@parallel_indices (I...) function accumulate_tensor_kernel!(
        II, xx, yy, xy, dt
    )

    # convenience closures
    @inline gather(A) = _gather(A, I...)

    @inbounds begin
        ε_pl = xx[I...], yy[I...], gather(xy)
        II[I...] += second_invariant_staggered(ε_pl...) * dt
    end

    return nothing
end

@parallel_indices (I...) function accumulate_tensor_kernel!(
        II, xx, yy, zz, yz, xz, xy, dt
    )
    # convenience closures
    @inline gather_yz(A) = _gather_yz(A, I...)
    @inline gather_xz(A) = _gather_xz(A, I...)
    @inline gather_xy(A) = _gather_xy(A, I...)

    @inbounds begin
        ε_pl = xx[I...], yy[I...], zz[I...], gather_yz(yz), gather_xz(xz), gather_xy(xy)
        II[I...] += second_invariant_staggered(ε_pl...) * dt
    end

    return nothing
end

# Accumulate the volumetric plastic strain: EVol_pl += dt * ε_vol_pl
# ε_vol_pl is a scalar field (= λ*(-dQ/dp)), distinct from the deviatoric
function accumulate_vol!(EVol_pl::AbstractArray, ε_vol_pl::AbstractArray, dt)
    _accumulate_vol!(EVol_pl, ε_vol_pl, dt)
    return nothing
end

function _accumulate_vol!(EVol_pl::AbstractArray, ε_vol_pl::AbstractArray, dt)
    ni = size(EVol_pl)
    @parallel (@idx ni) accumulate_vol_kernel!(EVol_pl, ε_vol_pl, dt)
    return nothing
end

@parallel_indices (I...) function accumulate_vol_kernel!(
        EVol_pl, ε_vol_pl, dt
    )
    @inbounds EVol_pl[I...] += dt * ε_vol_pl[I...]
    return nothing
end

@parallel_indices (I...) function tensor_invariant_center!(
        II, tensor::NTuple{N, T}
    ) where {N, T}
    @inbounds II[I...] = second_invariant_staggered(getindex.(tensor, I...)...)
    return nothing
end

"""
    tensor_invariant!(A::JustRelax.SymmetricTensor)

Compute the tensor invariant of the given symmetric tensor `A`.

# Arguments
- `A::JustRelax.SymmetricTensor`: The input symmetric tensor.
"""
function tensor_invariant!(A::JustRelax.SymmetricTensor)
    tensor_invariant!(backend(A), A)
    return nothing
end

function tensor_invariant!(::CPUBackendTrait, A::JustRelax.SymmetricTensor)
    return _tensor_invariant!(A)
end

function _tensor_invariant!(A::JustRelax.SymmetricTensor)
    ni = size(A.II)
    @parallel (@idx ni) tensor_invariant_kernel!(A.II, @tensor(A)...)
    return nothing
end

@parallel_indices (I...) function tensor_invariant_kernel!(II, xx, yy, xy)
    # convenience closure
    @inline gather(A) = _gather(A, I...)

    @inbounds begin
        τ = xx[I...], yy[I...], gather(xy)
        II[I...] = second_invariant_staggered(τ...)
    end

    return nothing
end

@parallel_indices (I...) function tensor_invariant_kernel!(II, xx, yy, zz, yz, xz, xy)

    # convenience closures
    @inline gather_yz(A) = _gather_yz(A, I...)
    @inline gather_xz(A) = _gather_xz(A, I...)
    @inline gather_xy(A) = _gather_xy(A, I...)

    @inbounds begin
        τ = xx[I...], yy[I...], zz[I...], gather_yz(yz), gather_xz(xz), gather_xy(xy)
        II[I...] = second_invariant_staggered(τ...)
    end

    return nothing
end

####

function update_stress!(stokes, θ, λ, phase_ratios, rheology, dt, θ_dτ, args)
    return update_stress!(
        islinear(rheology), stokes, θ, λ, phase_ratios, rheology, dt, θ_dτ, args
    )
end

function update_stress!(
        ::LinearRheologyTrait, stokes, ::Any, ::Any, phase_ratios, rheology, dt, θ_dτ, args
    )
    dim(::AbstractArray{T, N}) where {T, N} = Val(N)

    function f!(stokes, ::Val{2})
        center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
        update_halo!(stokes.τ.xy)
        return nothing
    end

    function f!(stokes, ::Val{3})
        center2vertex!(
            stokes.τ.yz,
            stokes.τ.xz,
            stokes.τ.xy,
            stokes.τ.yz_c,
            stokes.τ.xz_c,
            stokes.τ.xy_c,
        )
        update_halo!(stokes.τ.yz, stokes.τ.xz, stokes.τ.xy)
        return nothing
    end

    ni = size(phase_ratios.center)
    nDim = dim(stokes.viscosity.η)

    @parallel (@idx ni) compute_τ!(
        @tensor_center(stokes.τ)...,
        @tensor_center(stokes.τ_o)...,
        @strain(stokes)...,
        stokes.viscosity.η,
        θ_dτ,
        dt,
        phase_ratios.center,
        tupleize(rheology), # needs to be a tuple
    )

    f!(stokes, nDim)

    return nothing
end

function update_stress!(
        ::NonLinearRheologyTrait,
        stokes,
        θ,
        λ::AbstractArray{T, N},
        phase_ratios,
        rheology,
        dt,
        θ_dτ,
        args,
    ) where {N, T}
    ni = size(phase_ratios.center)
    nDim = Val(N)

    function f!(stokes, ::Val{2})
        center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
        update_halo!(stokes.τ.xy)
        return nothing
    end

    function f!(stokes, ::Val{3})
        center2vertex!(
            stokes.τ.yz,
            stokes.τ.xz,
            stokes.τ.xy,
            stokes.τ.yz_c,
            stokes.τ.xz_c,
            stokes.τ.xy_c,
        )
        update_halo!(stokes.τ.yz, stokes.τ.xz, stokes.τ.xy)
        return nothing
    end

    @parallel (@idx ni) compute_τ_nonlinear!(
        @tensor_center(stokes.τ),
        stokes.τ.II,
        @tensor_center(stokes.τ_o),
        @strain(stokes),
        @plastic_strain(stokes.ε_pl),
        stokes.EII_pl,
        stokes.P,
        θ,
        stokes.viscosity.η,
        stokes.viscosity.η_vep,
        λ,
        phase_ratios.center,
        tupleize(rheology), # needs to be a tuple
        dt,
        θ_dτ,
    )

    f!(stokes, nDim)

    return nothing
end

#####

Base.@propagate_inbounds @inline function clamped_indices(ni::NTuple{3, Integer}, i, j, k)
    nx, ny, nz = ni
    i0 = clamp(i - 1, 1, nx)
    ic = clamp(i, 1, nx)
    i1 = clamp(i + 1, 1, nx)
    j0 = clamp(j - 1, 1, ny)
    jc = clamp(j, 1, ny)
    j1 = clamp(j + 1, 1, ny)
    k0 = clamp(k - 1, 1, nz)
    kc = clamp(k, 1, nz)
    k1 = clamp(k + 1, 1, nz)
    return i0, j0, k0, ic, jc, kc, i1, j1, k1
end

Base.@propagate_inbounds @inline function av_clamped_yz(A, i0, j0, k0, ic, jc, kc, ::Vararg{Integer, N}) where {N}
    return 0.25 * (A[ic, j0, k0] + A[ic, jc, k0] + A[ic, j0, kc] + A[ic, jc, kc])
end

Base.@propagate_inbounds @inline function av_clamped_xz(A, i0, j0, k0, ic, jc, kc, ::Vararg{Integer, N}) where {N}
    return 0.25 * (A[i0, jc, k0] + A[ic, jc, k0] + A[i0, jc, kc] + A[ic, jc, kc])
end

Base.@propagate_inbounds @inline function av_clamped_xy(A, i0, j0, k0, ic, jc, kc, ::Vararg{Integer, N}) where {N}
    return 0.25 * (A[i0, j0, kc] + A[ic, j0, kc] + A[i0, jc, kc] + A[ic, jc, kc])
end

Base.@propagate_inbounds @inline function harm_clamped_yz(A, i0, j0, k0, ic, jc, kc, ::Vararg{Integer, N}) where {N}
    return 4 / (1 / A[ic, j0, k0] + 1 / A[ic, jc, k0] + 1 / A[ic, j0, kc] + 1 / A[ic, jc, kc])
end

Base.@propagate_inbounds @inline function harm_clamped_xz(A, i0, j0, k0, ic, jc, kc, ::Vararg{Integer, N}) where {N}
    return 4 / (1 / A[i0, jc, k0] + 1 / A[ic, jc, k0] + 1 / A[i0, jc, kc] + 1 / A[ic, jc, kc])
end

Base.@propagate_inbounds @inline function harm_clamped_xy(A, i0, j0, k0, ic, jc, kc, ::Vararg{Integer, N}) where {N}
    return 4 / (1 / A[i0, j0, kc] + 1 / A[ic, j0, kc] + 1 / A[i0, jc, kc] + 1 / A[ic, jc, kc])
end

# on yz
Base.@propagate_inbounds @inline function av_clamped_yz_z(A, i0, j0, k0, ic, jc, kc, i1, j1, k1)
    return 0.25 * (A[ic, jc, k0] + A[i1, jc, k0] + A[ic, jc, kc] + A[i1, jc, kc])
end

Base.@propagate_inbounds @inline function av_clamped_yz_y(A, i0, j0, k0, ic, jc, kc, i1, j1, k1)
    return 0.25 * (A[ic, j0, kc] + A[i1, j0, kc] + A[ic, jc, kc] + A[i1, jc, kc])
end

# on xz
Base.@propagate_inbounds @inline function av_clamped_xz_z(A, i0, j0, k0, ic, jc, kc, i1, j1, k1)
    return 0.25 * (A[ic, jc, k0] + A[ic, j1, k0] + A[ic, jc, kc] + A[ic, j1, kc])
end

Base.@propagate_inbounds @inline function av_clamped_xz_x(A, i0, j0, k0, ic, jc, kc, i1, j1, k1)
    return 0.25 * (A[i0, jc, kc] + A[ic, jc, kc] + A[ic, j1, kc] + A[i0, j1, kc])
end

# on xy
Base.@propagate_inbounds @inline function av_clamped_xy_y(A, i0, j0, k0, ic, jc, kc, i1, j1, k1)
    return 0.25 * (A[ic, j0, kc] + A[ic, jc, kc] + A[ic, j0, k1] + A[ic, jc, k1])
end

Base.@propagate_inbounds @inline function av_clamped_xy_x(A, i0, j0, k0, ic, jc, kc, i1, j1, k1)
    return 0.25 * (A[i0, jc, kc] + A[ic, jc, kc] + A[i0, jc, k1] + A[ic, jc, k1])
end

# 3D kernel
@parallel_indices (I...) function update_stresses_center_vertex_ps!(
        ε::NTuple{6},         # normal components @ centers; shear components @ vertices
        ε_pl::NTuple{6},      # whole Voigt tensor @ centers
        EII,                  # accumulated plastic strain rate @ centers
        τ::NTuple{6},         # whole Voigt tensor @ centers
        τshear_v::NTuple{3},  # shear tensor components @ vertices
        τ_o::NTuple{6},
        τshear_ov::NTuple{3}, # shear tensor components @ vertices
        Pr,
        Pr_c,
        η,
        λ,
        λv::NTuple{3},
        τII,
        η_vep,
        relλ,
        dt,
        θ_dτ,
        rheology,
        phase_center,
        phase_vertex,
        phase_xy,
        phase_yz,
        phase_xz,
    )
    τyzv, τxzv, τxyv = τshear_v
    τyzv_old, τxzv_old, τxyv_old = τshear_ov

    ni = size(Pr)
    Ic = clamped_indices(ni, I...)

    ## yz
    @inbounds if all(I .≤ size(ε[4]))
        # interpolate to ith vertex
        ηv_ij = harm_clamped_yz(η, Ic...)
        Pv_ij = av_clamped_yz(Pr, Ic...)
        EIIv_ij = av_clamped_yz(EII, Ic...)
        εxxv_ij = av_clamped_yz(ε[1], Ic...)
        εyyv_ij = av_clamped_yz(ε[2], Ic...)
        εzzv_ij = av_clamped_yz(ε[3], Ic...)
        εyzv_ij = ε[4][I...]
        εxzv_ij = av_clamped_yz_y(ε[5], Ic...)
        εxyv_ij = av_clamped_yz_z(ε[6], Ic...)

        ε_plyzv_ij = ε_pl[4][I...]

        τxxv_ij = av_clamped_yz(τ[1], Ic...)
        τyyv_ij = av_clamped_yz(τ[2], Ic...)
        τzzv_ij = av_clamped_yz(τ[3], Ic...)
        τyzv_ij = τyzv[I...]
        τxzv_ij = av_clamped_yz_y(τxzv, Ic...)
        τxyv_ij = av_clamped_yz_z(τxyv, Ic...)

        τxxv_old_ij = av_clamped_yz(τ_o[1], Ic...)
        τyyv_old_ij = av_clamped_yz(τ_o[2], Ic...)
        τzzv_old_ij = av_clamped_yz(τ_o[3], Ic...)
        τyzv_old_ij = τyzv_old[I...]
        τxzv_old_ij = av_clamped_yz_y(τxzv_old, Ic...)
        τxyv_old_ij = av_clamped_yz_z(τxyv_old, Ic...)

        # vertex parameters
        phase = @inbounds phase_yz[I...]
        is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(
            rheology, EIIv_ij, phase
        )
        _Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        Kv = fn_ratio(get_bulk_modulus, rheology, phase)
        volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

        # stress increments @ vertex
        dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
        dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
        dτzzv = compute_stress_increment(τzzv_ij, τzzv_old_ij, ηv_ij, εzzv_ij, _Gvdt, dτ_rv)
        dτyzv = compute_stress_increment(τyzv_ij, τyzv_old_ij, ηv_ij, εyzv_ij, _Gvdt, dτ_rv)
        dτxzv = compute_stress_increment(τxzv_ij, τxzv_old_ij, ηv_ij, εxzv_ij, _Gvdt, dτ_rv)
        dτxyv = compute_stress_increment(τxyv_ij, τxyv_old_ij, ηv_ij, εxyv_ij, _Gvdt, dτ_rv)

        dτijv = dτxxv, dτyyv, dτzzv, dτyzv, dτxzv, dτxyv
        τijv = τxxv_ij, τyyv_ij, τzzv_ij, τyzv_ij, τxzv_ij, τxyv_ij
        τIIv_ij = second_invariant(τijv .+ dτijv)

        # yield function @ vertex
        Fv = τIIv_ij - Cv * cosϕv - Pv_ij * sinϕv
        # Fv = if Pv_ij ≥ 0
        #     τIIv_ij - Cv * cosϕv - Pv_ij * sinϕv
        # else
        #     τIIv_ij - Cv
        # end
        if is_pl && !iszero(τIIv_ij) && Fv > 0
            # stress correction @ vertex
            λv[1][I...] =
                (1.0 - relλ) * λv[1][I...] +
                relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))

            dQdτyz = 0.5 * (τyzv_ij + dτyzv) / τIIv_ij
            ε_plyzv_ij = λv[1][I...] * dQdτyz
            τyzv[I...] += @muladd dτyzv - 2.0 * ηv_ij * ε_plyzv_ij * dτ_rv
            ε_pl[4][I...] = ε_plyzv_ij
        else
            # stress correction @ vertex
            τyzv[I...] += dτyzv
            ε_pl[4][I...] = 0.0
        end
    end

    ## xz
    @inbounds if all(I .≤ size(ε[5]))
        # interpolate to ith vertex
        ηv_ij = harm_clamped_xz(η, Ic...)
        EIIv_ij = av_clamped_xz(EII, Ic...)
        Pv_ij = av_clamped_xz(Pr, Ic...)
        εxxv_ij = av_clamped_xz(ε[1], Ic...)
        εyyv_ij = av_clamped_xz(ε[2], Ic...)
        εzzv_ij = av_clamped_xz(ε[3], Ic...)
        εyzv_ij = av_clamped_xz_x(ε[4], Ic...)
        εxzv_ij = ε[5][I...]
        εxyv_ij = av_clamped_xz_z(ε[6], Ic...)
        τxxv_ij = av_clamped_xz(τ[1], Ic...)
        τyyv_ij = av_clamped_xz(τ[2], Ic...)
        τzzv_ij = av_clamped_xz(τ[3], Ic...)
        τyzv_ij = av_clamped_xz_x(τyzv, Ic...)
        τxzv_ij = τxzv[I...]
        τxyv_ij = av_clamped_xz_z(τxyv, Ic...)
        τxxv_old_ij = av_clamped_xz(τ_o[1], Ic...)
        τyyv_old_ij = av_clamped_xz(τ_o[2], Ic...)
        τzzv_old_ij = av_clamped_xz(τ_o[3], Ic...)
        τyzv_old_ij = av_clamped_xz_x(τyzv_old, Ic...)
        τxzv_old_ij = τxzv_old[I...]
        τxyv_old_ij = av_clamped_xz_z(τxyv_old, Ic...)
        ε_plxzv_ij = ε_pl[5][I...]

        # vertex parameters
        phase = @inbounds phase_xz[I...]
        is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(
            rheology, EIIv_ij, phase
        )
        _Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        Kv = fn_ratio(get_bulk_modulus, rheology, phase)
        volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

        # stress increments @ vertex
        dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
        dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
        dτzzv = compute_stress_increment(τzzv_ij, τzzv_old_ij, ηv_ij, εzzv_ij, _Gvdt, dτ_rv)
        dτyzv = compute_stress_increment(τyzv_ij, τyzv_old_ij, ηv_ij, εyzv_ij, _Gvdt, dτ_rv)
        dτxzv = compute_stress_increment(τxzv_ij, τxzv_old_ij, ηv_ij, εxzv_ij, _Gvdt, dτ_rv)
        dτxyv = compute_stress_increment(τxyv_ij, τxyv_old_ij, ηv_ij, εxyv_ij, _Gvdt, dτ_rv)

        dτijv = dτxxv, dτyyv, dτzzv, dτyzv, dτxzv, dτxyv
        τijv = τxxv_ij, τyyv_ij, τzzv_ij, τyzv_ij, τxzv_ij, τxyv_ij
        τIIv_ij = second_invariant(τijv .+ dτijv)

        # yield function @ vertex
        Fv = τIIv_ij - Cv * cosϕv - Pv_ij * sinϕv
        # Fv = if Pv_ij ≥ 0
        #     τIIv_ij - Cv * cosϕv - Pv_ij * sinϕv
        # else
        #     τIIv_ij - Cv
        # end
        if is_pl && !iszero(τIIv_ij) && Fv > 0
            # stress correction @ vertex
            λv[2][I...] =
                (1.0 - relλ) * λv[2][I...] +
                relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))

            dQdτxz = 0.5 * (τxzv_ij + dτxzv) / τIIv_ij
            ε_plxzv_ij = λv[2][I...] * dQdτxz
            τxzv[I...] += @muladd dτxzv - 2.0 * ηv_ij * ε_plxzv_ij * dτ_rv
            ε_pl[5][I...] = ε_plxzv_ij
        else
            # stress correction @ vertex
            τxzv[I...] += dτxzv
            ε_pl[5][I...] = 0.0
        end
    end

    ## xy
    if all(I .≤ size(ε[6]))
        # interpolate to ith vertex
        ηv_ij = harm_clamped_xy(η, Ic...)
        EIIv_ij = av_clamped_xy(EII, Ic...)
        Pv_ij = av_clamped_xy(Pr, Ic...)
        εxxv_ij = av_clamped_xy(ε[1], Ic...)
        εyyv_ij = av_clamped_xy(ε[2], Ic...)
        εzzv_ij = av_clamped_xy(ε[3], Ic...)
        εyzv_ij = av_clamped_xy_x(ε[4], Ic...)
        εxzv_ij = av_clamped_xy_y(ε[5], Ic...)
        εxyv_ij = ε[6][I...]
        ε_plxyv_ij = ε_pl[6][I...]

        τxxv_ij = av_clamped_xy(τ[1], Ic...)
        τyyv_ij = av_clamped_xy(τ[2], Ic...)
        τzzv_ij = av_clamped_xy(τ[3], Ic...)
        τyzv_ij = av_clamped_xy_x(τyzv, Ic...)
        τxzv_ij = av_clamped_xy_y(τxzv, Ic...)
        τxyv_ij = τxyv[I...]

        τxxv_old_ij = av_clamped_xy(τ_o[1], Ic...)
        τyyv_old_ij = av_clamped_xy(τ_o[2], Ic...)
        τzzv_old_ij = av_clamped_xy(τ_o[3], Ic...)
        τyzv_old_ij = av_clamped_xy_x(τyzv_old, Ic...)
        τxzv_old_ij = av_clamped_xy_y(τxzv_old, Ic...)
        τxyv_old_ij = τxyv_old[I...]

        # vertex parameters
        phase = @inbounds phase_xy[I...]
        is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(
            rheology, EIIv_ij, phase
        )
        _Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        Kv = fn_ratio(get_bulk_modulus, rheology, phase)
        volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

        # stress increments @ vertex
        dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
        dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
        dτzzv = compute_stress_increment(τzzv_ij, τzzv_old_ij, ηv_ij, εzzv_ij, _Gvdt, dτ_rv)
        dτyzv = compute_stress_increment(τyzv_ij, τyzv_old_ij, ηv_ij, εyzv_ij, _Gvdt, dτ_rv)
        dτxzv = compute_stress_increment(τxzv_ij, τxzv_old_ij, ηv_ij, εxzv_ij, _Gvdt, dτ_rv)
        dτxyv = compute_stress_increment(τxyv_ij, τxyv_old_ij, ηv_ij, εxyv_ij, _Gvdt, dτ_rv)
        dτijv = dτxxv, dτyyv, dτzzv, dτyzv, dτxzv, dτxyv
        τijv = τxxv_ij, τyyv_ij, τzzv_ij, τyzv_ij, τxzv_ij, τxyv_ij
        τIIv_ij = second_invariant(τijv .+ dτijv)

        # yield function @ vertex
        Fv = τIIv_ij - Cv * cosϕv - Pv_ij * sinϕv
        # Fv = if Pv_ij ≥ 0
        #     τIIv_ij - Cv * cosϕv - Pv_ij * sinϕv
        # else
        #     τIIv_ij - Cv
        # end
        if is_pl && !iszero(τIIv_ij) && Fv > 0
            # stress correction @ vertex
            λv[3][I...] =
                (1.0 - relλ) * λv[3][I...] +
                relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))

            dQdτxy = 0.5 * (τxyv_ij + dτxyv) / τIIv_ij
            ε_plxyv_ij = λv[3][I...] * dQdτxy
            τxyv[I...] += @muladd  dτxyv - 2.0 * ηv_ij * ε_plxyv_ij * dτ_rv
            ε_pl[6][I...] = ε_plxyv_ij
        else
            # stress correction @ vertex
            τxyv[I...] += dτxyv
            ε_pl[6][I...] = 0.0
        end
    end

    ## center
    if all(I .≤ ni)
        # Material properties
        phase = @inbounds phase_center[I...]
        _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)
        K = fn_ratio(get_bulk_modulus, rheology, phase)
        volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
        ηij = η[I...]
        dτ_r = inv(θ_dτ + ηij * _Gdt + 1.0)

        # cache strain rates for center calculations
        τij, τij_o, εij, εij_pl = cache_tensors(τ, τ_o, ε, ε_pl, I...)

        # visco-elastic strain rates @ center
        εij_ve = @. εij + 0.5 * τij_o * _Gdt
        εII_ve = second_invariant(εij_ve)
        # stress increments @ center
        dτij = @. (-(τij - τij_o) * ηij * _Gdt - τij + 2.0 * ηij * εij) * dτ_r
        τII_ij = second_invariant(dτij .+ τij)
        # yield function @ center
        F = τII_ij - C * cosϕ - Pr[I...] * sinϕ
        # F = if Pr[I...] ≥ 0
        #     τII_ij - C * cosϕ - Pr[I...] * sinϕ
        # else
        #     τII_ij - C
        # end
        τII_ij = if is_pl && !iszero(τII_ij) && F > 0
            # stress correction @ center
            λ[I...] =
                (1.0 - relλ) * λ[I...] +
                relλ * (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
            dQdτij = @. 0.5 * (τij + dτij) / τII_ij
            εij_pl = λ[I...] .* dQdτij
            dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
            τij = dτij .+ τij
            Base.@nexprs 6 i -> begin
                @inbounds τ[i][I...] = dτij[i] + τij[i]
            end
            Base.@nexprs 3 i -> begin
                @inbounds ε_pl[i][I...] = εij_pl[i]
            end
            τII[I...] = τII_ij = second_invariant(τij)
        else
            # stress correction @ center
            Base.@nexprs 6 i -> begin
                @inbounds τ[i][I...] = dτij[i] .+ τij[i]
            end
            Base.@nexprs 3 i -> begin
                @inbounds ε_pl[i][I...] = 0.0
            end
            τII[I...] = τII_ij
        end
        η_vep[I...] = τII_ij * 0.5 * inv(second_invariant(εij))
        Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
    end

    return nothing
end

# 2D kernel
@parallel_indices (I...) function update_stresses_center_vertex_ps!(
        ε::NTuple{3},         # normal components @ centers; shear components @ vertices
        ε_pl::NTuple{3},      # whole Voigt tensor @ centers
        EII,                  # accumulated plastic strain rate @ centers
        τ::NTuple{3},         # whole Voigt tensor @ centers
        τshear_v::NTuple{1},  # shear tensor components @ vertices
        τ_o::NTuple{3},
        τshear_ov::NTuple{1}, # shear tensor components @ vertices
        Pr,
        Pr_c,
        η,
        λ,
        λv,
        τII,
        η_vep,
        relλ,
        dt,
        θ_dτ,
        rheology,
        phase_center,
        phase_vertex,
    )
    τxyv = τshear_v[1]
    τxyv_old = τshear_ov[1]
    εij_plv = ε_pl[3]
    ni = size(Pr)
    Ic = clamped_indices(ni, I...)

    # interpolate to ith vertex
    Pv_ij = @inbounds av_clamped(Pr, Ic...)
    εxxv_ij = @inbounds av_clamped(ε[1], Ic...)
    εyyv_ij = @inbounds av_clamped(ε[2], Ic...)
    τxxv_ij = @inbounds av_clamped(τ[1], Ic...)
    τyyv_ij = @inbounds av_clamped(τ[2], Ic...)
    τxxv_old_ij = @inbounds av_clamped(τ_o[1], Ic...)
    τyyv_old_ij = @inbounds av_clamped(τ_o[2], Ic...)
    EIIv_ij = @inbounds av_clamped(EII, Ic...)

    ## vertex
    phase = @inbounds phase_vertex[I...]
    is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
    _Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
    Kv = fn_ratio(get_bulk_modulus, rheology, phase)
    volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
    ηv_ij = @inbounds harm_clamped(η, Ic...)
    dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

    # stress increments @ vertex
    dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
    dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
    dτxyv = @inbounds compute_stress_increment(
        τxyv[I...], τxyv_old[I...], ηv_ij, ε[3][I...], _Gvdt, dτ_rv
    )
    τijv = τxxv_ij, τyyv_ij, τxyv[I...]
    dτijv = dτxxv, dτyyv, dτxyv
    τIIv_ij = second_invariant(dτijv .+ τijv)

    # yield function @ center
    Fv = τIIv_ij - Cv * cosϕv - Pv_ij * sinϕv
    # Fv = if Pv_ij ≥ 0
    #     τIIv_ij - Cv * cosϕv - Pv_ij * sinϕv
    # else
    #     τIIv_ij - Cv
    # end
    @inbounds if is_pl && !iszero(τIIv_ij)  && Fv > 0
        # stress correction @ vertex
        λv[I...] =
            @muladd (1.0 - relλ) * λv[I...] +
            relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
        dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij
        εij_plv = λv[I...] * dQdτxy
        τxyv[I...] += @muladd dτxyv - 2.0 * ηv_ij * εij_plv * dτ_rv
        ε_pl[3][I...] = εij_plv
    else
        # stress correction @ vertex
        τxyv[I...] += dτxyv
        ε_pl[3][I...] = 0.0
    end

    ## center
    if all(I .≤ ni)
        # Material properties
        phase = @inbounds phase_center[I...]
        _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        is_pl, C, sinϕ, cosϕ, sinψ, η_reg = @inbounds plastic_params_phase(rheology, EII[I...], phase)
        K = fn_ratio(get_bulk_modulus, rheology, phase)
        volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
        ηij = η[I...]
        dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

        # cache strain rates for center calculations
        τij, τij_o, εij, εij_pl = cache_tensors(τ, τ_o, ε, ε_pl, I...)

        # visco-elastic strain rates @ center
        εij_ve = @. εij + 0.5 * τij_o * _Gdt
        εII_ve = second_invariant(εij_ve)
        # stress increments @ center
        dτij = compute_stress_increment(τij, τij_o, ηij, εij, _Gdt, dτ_r)
        τII_ij = second_invariant(dτij .+ τij)
        # yield function @ center
        F = @inbounds τII_ij - C * cosϕ - Pr[I...] * sinϕ
        # F = if Pr[I...] ≥ 0
        #     @inbounds τII_ij - C * cosϕ - Pr[I...] * sinϕ
        # else
        #     @inbounds τII_ij - C
        # end
        τII_ij = @inbounds if is_pl && !iszero(τII_ij) && F > 0
            # stress correction @ center
            λ[I...] =
                @muladd (1.0 - relλ) * λ[I...] +
                relλ * (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
            dQdτij = @. 0.5 * (τij + dτij) / τII_ij
            εij_pl = λ[I...] .* dQdτij
            dτij = @muladd @. dτij - 2.0 * ηij * εij_pl * dτ_r
            τij = dτij .+ τij

            Base.@nexprs 3 i -> begin
                @inbounds τ[i][I...] = τij[i]
            end
            Base.@nexprs 2 i -> begin
                @inbounds ε_pl[i][I...] = εij_pl[i]
            end
            τII_ij = second_invariant(τij)
        else
            # stress correction @ center

            Base.@nexprs 3 i -> begin
                @inbounds τ[i][I...] = dτij[i] .+ τij[i]
            end
            Base.@nexprs 2 i -> begin
                @inbounds ε_pl[i][I...] = 0.0
            end
            τII_ij
        end
        @inbounds τII[I...] = τII_ij
        @inbounds η_vep[I...] = τII_ij * 0.5 * inv(second_invariant(εij))
        @inbounds Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
    end

    return nothing
end

## 2D kernel with strain increment Δε
@parallel_indices (I...) function update_stresses_center_vertex_ps!(
        ε::NTuple{3},         # normal components @ centers; shear components @ vertices
        Δε::NTuple{3},         # normal components @ centers; shear components @ vertices
        ε_pl::NTuple{3},      # whole Voigt tensor @ centers
        EII,                  # accumulated plastic strain rate @ centers
        τ::NTuple{3},         # whole Voigt tensor @ centers
        τshear_v::NTuple{1},  # shear tensor components @ vertices
        τ_o::NTuple{3},
        τshear_ov::NTuple{1}, # shear tensor components @ vertices
        Pr,
        Pr_c,
        η,
        λ,
        λv,
        τII,
        η_vep,
        relλ,
        dt,
        θ_dτ,
        rheology,
        phase_center,
        phase_vertex,
        phase_xy,
        phase_yz,
        phase_xz,
    )
    τxyv = τshear_v[1]
    τxyv_old = τshear_ov[1]
    εij_plv = ε_pl[3]
    ni = size(Pr)
    Ic = clamped_indices(ni, I...)

    # interpolate to ith vertex
    Pv_ij = av_clamped(Pr, Ic...)
    εxxv_ij = av_clamped(ε[1], Ic...)
    εyyv_ij = av_clamped(ε[2], Ic...)
    Δεxxv_ij = av_clamped(Δε[1], Ic...)
    Δεyyv_ij = av_clamped(Δε[2], Ic...)
    τxxv_ij = av_clamped(τ[1], Ic...)
    τyyv_ij = av_clamped(τ[2], Ic...)
    τxxv_old_ij = av_clamped(τ_o[1], Ic...)
    τyyv_old_ij = av_clamped(τ_o[2], Ic...)
    EIIv_ij = av_clamped(EII, Ic...)

    ## vertex
    phase = @inbounds phase_vertex[I...]
    is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
    _Gv = inv(fn_ratio(get_shear_modulus, rheology, phase))
    _Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
    Kv = fn_ratio(get_bulk_modulus, rheology, phase)
    volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
    ηv_ij = harm_clamped(η, Ic...)
    dτ_rv = inv(θ_dτ * dt + ηv_ij * _Gv + dt)
    dτ_rv2 = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)
    # stress increments @ vertex
    dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, Δεxxv_ij, _Gv, dτ_rv, dt)
    dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, Δεyyv_ij, _Gv, dτ_rv, dt)
    dτxyv = compute_stress_increment(
        τxyv[I...], τxyv_old[I...], ηv_ij, Δε[3][I...], _Gv, dτ_rv, dt
    )

    τijv = τxxv_ij, τyyv_ij, τxyv[I...]
    dτijv = dτxxv, dτyyv, dτxyv
    τIIv_ij = second_invariant(dτijv .+ τijv)

    # yield function @ center
    Fv = τIIv_ij - Cv * cosϕv - Pv_ij * sinϕv
    # Fv = if Pv_ij ≥ 0
    #     τIIv_ij - Cv * cosϕv - Pv_ij * sinϕv
    # else
    #     τIIv_ij - Cv
    # end
    @inbounds if is_pl && !iszero(τIIv_ij) && Fv > 0
        # stress correction @ vertex
        λv[I...] =
            @muladd (1.0 - relλ) * λv[I...] +
            relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv * dt + η_regv + volumev))
        dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij
        εij_plv = λv[I...] * dQdτxy
        τxyv[I...] += @muladd dτxyv - 2.0 * ηv_ij * dt * εij_plv * dτ_rv
        ε_pl[3][I...] = εij_plv
    else
        # stress correction @ vertex
        τxyv[I...] += dτxyv
        ε_pl[3][I...] = 0.0
    end

    ## center
    if all(I .≤ ni)
        # Material properties
        phase = @inbounds phase_center[I...]
        _G = inv(fn_ratio(get_shear_modulus, rheology, phase))
        _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        is_pl, C, sinϕ, cosϕ, sinψ, η_reg = @inbounds plastic_params_phase(rheology, EII[I...], phase)
        K = fn_ratio(get_bulk_modulus, rheology, phase)
        volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
        ηij = η[I...]
        dτ_r = 1.0 / (θ_dτ * dt + ηij * _G + dt)
        dτ_r2 = inv(θ_dτ + ηij * _Gdt + 1.0)
        # cache strain rates for center calculations
        τij, τij_o, εij, εij_pl, Δεij = cache_tensors(τ, τ_o, ε, ε_pl, Δε, I...)

        # visco-elastic strain rates @ center
        εij_ve = @. εij + 0.5 * τij_o * _Gdt
        εII_ve = second_invariant(εij_ve)
        # stress increments @ center
        # dτij = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij) * dτ_r
        dτij = compute_stress_increment(τij, τij_o, ηij, Δεij, _G, dτ_r, dt)
        τII_ij = second_invariant(dτij .+ τij)
        # yield function @ center
        F = @inbounds τII_ij - C * cosϕ - Pr[I...] * sinϕ
        # F = if Pr[I...] ≥ 0
        #     @inbounds τII_ij - C * cosϕ - Pr[I...] * sinϕ
        # else
        #     τII_ij - C
        # end
        τII_ij = @inbounds if is_pl && !iszero(τII_ij) && F > 0
            # stress correction @ center
            λ[I...] =
                @muladd (1.0 - relλ) * λ[I...] +
                relλ * (max(F, 0.0) / (η[I...] * dτ_r * dt + η_reg + volume))
            dQdτij = @. 0.5 * (τij + dτij) / τII_ij
            # dτij        = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * (εij  - λ[I...] *dQdτij )) * dτ_r
            εij_pl = λ[I...] .* dQdτij
            dτij = @muladd @. dτij - 2.0 * ηij * dt * εij_pl * dτ_r
            τij = dτij .+ τij

            Base.@nexprs 3 i -> begin
                @inbounds τ[i][I...] = τij[i]
            end
            Base.@nexprs 2 i -> begin
                @inbounds ε_pl[i][I...] = εij_pl[i]
            end
            τII_ij = second_invariant(τij)
        else
            Base.@nexprs 3 i -> begin
                @inbounds τ[i][I...] = dτij[i] .+ τij[i]
            end
            Base.@nexprs 2 i -> begin
                @inbounds ε_pl[i][I...] = 0.0
            end
            τII_ij
        end

        @inbounds τII[I...] = τII_ij
        @inbounds η_vep[I...] = τII_ij * 0.5 * inv(second_invariant(εij))
        @inbounds Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
    end

    return nothing
end

Base.@propagate_inbounds @inline function clamped_indices(ni::NTuple{2, Integer}, i, j)
    nx, ny = ni
    i0 = clamp(i - 1, 1, nx)
    ic = clamp(i, 1, nx)
    j0 = clamp(j - 1, 1, ny)
    jc = clamp(j, 1, ny)
    return i0, j0, ic, jc
end

Base.@propagate_inbounds @inline function av_clamped(A, i0, j0, ic, jc)
    return 0.25 * (A[i0, j0] + A[ic, jc] + A[i0, jc] + A[ic, j0])
end

Base.@propagate_inbounds @inline function harm_clamped(A, i0, j0, ic, jc)
    return 4 / (1 / A[i0, j0] + 1 / A[ic, jc] + 1 / A[i0, jc] + 1 / A[ic, j0])
end

# ============================================================
# Tensile-cap plasticity helpers (Popov, Berlie & Kaus, GMD 2025)
# https://doi.org/10.5194/gmd-18-7035-2025
#
# The composite yield surface combines a linear Drucker–Prager (DP) shear
# failure envelope with a circular tensile cap so that both the yield surface
# and the flow potential are globally C¹-continuous. No Jacobian or local
# Newton iteration is required here: we reuse the existing pseudo-transient
# (PT) Perzyna-type explicit update, replacing the plain DP yield function
# with the smooth composite one and using the matching flow-potential gradient.
#
# Material parameters (all scalar):
#   sinϕ, cosϕ  – DP friction angle (from GeoParams)
#   sinψ         – DP dilation angle
#   C            – DP cohesion
#   pT           – tensile strength (positive, [Pa])
#
# Derived geometry (Eqs. 12–17 of the paper):
#   k = sinϕ,  c = C*cosϕ              – DP slope and intercept
#   kq = sinψ                           – DP dilatation slope
#   Delimiter pressure p_d (smooth intersection of DP and cap):
#        p_d = (c - pT*(1-k²) ) / (k*(1+1/k²))    ← simplified from paper
#   Cap center  p_c  and radius Ry are computed so that the two segments
#   join with a common tangent (C¹) at the delimiter point.
#
# Yield function (scalar, Eq. 18):
#   In DP regime  (p ≤ p_d):  F = τII - k*p - c
#   In tensile regime (p > p_d):  F = a * sqrt((p-pc)² + qc²*τII²/τII²) - Ry
#     where `a` is a scaling coefficient so that outside the cap the function
#     is globally continuous (see paper Fig. 2b).
#
# Flow potential gradient (Eqs. 20-21):
#   DP regime:   ∂Q/∂τij = τij/(2τII),  ∂Q/∂p = -sinψ
#   Tensile cap: ∂Q/∂τij = τij/(2τII) * (qc² * b / Rq),  ∂Q/∂p = (p-pqc)*b/Rq
#     where Rq = sqrt((p-pqc)²+qc²*τII²) is the cap-flow-potential radius.
# ============================================================

"""
    tensile_cap_params(sinϕ, cosϕ, sinψ, C, pT)

Compute the geometric parameters of the smooth composite Drucker–Prager +
circular tensile-cap yield surface from Popov et al. (2025, GMD).
Formulas follow `get_yield_param` in the paper's reference Python script.

Returns a named tuple with fields:
  k, kf, c   – DP friction/dilation slopes and projected cohesion
  a, b        – cap and flow-potential scaling coefficients
  pd, τd      – delimiter pressure and shear stress (boundary of regimes)
  py          – cap center pressure
  Ry           – cap radius
  pq          – flow-potential center pressure
  Rf          – flow-potential radius
  pdf, sdf    – flow-potential delimiter point coordinates
"""
@inline function tensile_cap_params(sinϕ::T, cosϕ::T, sinψ::T, C::T, pT::T) where {T}
    k   = sinϕ              # DP friction coefficient  (= sin φ)
    kf  = sinψ              # DP dilatation coefficient (= sin ψ)
    c   = C * cosϕ          # projected cohesion intercept

    # Cap scaling: a = sqrt(1 + k²)  (from paper / plot2D.py)
    a    = sqrt(one(T) + k^2)
    b    = sqrt(one(T) + kf^2)
    cosa = inv(a)
    sina = k * cosa          # = k/a

    # Cap center (on the p-axis) and radius
    py   = (pT + c * cosa) / (one(T) - sina)
    Ry    = py - pT

    # Delimiter: smooth C¹ junction of DP and cap
    pd   = py - Ry * sina
    τd   = c + k * pd        # shear stress at delimiter (on DP surface)

    # Flow potential center and scaling  (b = sqrt(1 + kf²))
    pq   = pd + kf * (c + k * pd)
    Rf   = pq - pT           # flow-potential radius

    # Delimiter point on the flow-potential curve
    norm_pf = hypot(pd - pq, τd)
    pdf     = pq + Rf * (pd - pq) / norm_pf
    sdf     = Rf * τd / norm_pf

    return (; k, kf, c, a, b, pd, τd, py, Ry, pq, Rf, pdf, sdf)
end

"""
    composite_yield(τII, p, k, c, py, Ry, a, pd, τd)

Evaluate the smooth composite yield function F(τII, p).
  - F < 0 : elastic
  - F ≥ 0 : yielding

Regime selection (from plot2D.py):
  Tensile-cap regime when: τII < (py - p)*τd / (py - pd)
  DP regime otherwise:
    F = τII - k*p - c
  Tensile-cap regime:
    F = a * (√(τII² + (p - py)²) - Ry)
"""
@inline function composite_yield(τII::T, p::T, k::T, c::T, py::T, Ry::T, a::T, pd::T, τd::T) where {T}
    if τII < (py - p) * τd / (py - pd)
        # Tensile-cap regime: circular cap scaled by a
        return a * (hypot(τII, p - py) - Ry)
    else
        # Drucker–Prager regime
        return τII - k * p - c
    end
end

"""
    composite_flow_gradient(τII, τij, p, k, kf, c, py, a, pd, τd, pq, b, Rf, pdf, sdf)

Return `(dQdτij, dQdp)` – the deviatoric and volumetric components of the
(non-associated) flow-potential gradient ∂Q/∂(τij, p) at the current stress state.

Regime selection mirrors the yield surface (using flow-potential parameters):
  Tensile-cap regime when: τII < (pq - p)*τd / (pq - pd)
  DP regime:    ∂Q/∂τij = τij/(2τII),   ∂Q/∂p = -kf
  Cap regime:   ∂Q/∂τij = b*τij/(Rq*2τII)*τII,   ∂Q/∂p = b*(p-pq)/Rq
    where Rq = sqrt(τII² + (p-pq)²)
"""
@inline function composite_flow_gradient(
        τII::T, τij::NTuple{N,T}, p::T,
        k::T, kf::T, c::T, py::T, a::T, pd::T, τd::T, pq::T, b::T, Rf::T, pdf::T, sdf::T
    ) where {T, N}
    if τII < (pq - p) * τd / (pq - pd)
        # Circular cap flow potential: Q = b*(sqrt(τII² + (p-pq)²) - Rf)
        Rq     = hypot(τII, p - pq)
        inv_Rq = inv(Rq)
        dQdτij = ntuple(i -> 0.5 * b * τij[i] * inv_Rq, Val(N))
        dQdp   = b * (p - pq) * inv_Rq
    else
        # DP flow potential: Q = τII - kf*p
        dQdτij = ntuple(i -> τij[i] / (2 * τII), Val(N))
        dQdp   = -kf
    end
    return dQdτij, dQdp
end

@inline function composite_flow_gradient(
        τII::T, τij::T, p::T,
        k::T, kf::T, c::T, py::T, a::T, pd::T, τd::T, pq::T, b::T, Rf::T, pdf::T, sdf::T
    ) where {T}
    if τII < (pq - p) * τd / (pq - pd)
        Rq     = hypot(τII, p - pq)
        dQdτij = 0.5 * b * τij / Rq
        dQdp   = b * (p - pq) / Rq
    else
        dQdτij = τij / (2 * τII)
        dQdp   = -kf
    end
    return dQdτij, dQdp
end

# ============================================================
# 2D stress kernel with smooth tensile-cap plasticity
# (Popov, Berlie & Kaus, GMD 2025) – pseudo-transient adaptation
#
# This is a drop-in replacement for update_stresses_center_vertex_ps! that
# uses the smooth composite DP + tensile-cap yield surface instead of the
# plain Drucker–Prager one. The PT update rule for the viscoplastic multiplier
# λ is kept identical; no local Newton iterations or Jacobian are needed.
#
# Extra argument compared to the plain kernel:
#   pT_center   – tensile strength field @ cell centers [Pa]
#   pT_vertex   – tensile strength field @ vertices     [Pa]
#
# The volumetric plastic strain (pressure correction) is handled via the
# flow-potential volumetric gradient dQ/dp, which naturally collapses to the
# standard sinψ term in the DP regime.
# ============================================================

# 2D kernel
@parallel_indices (I...) function update_stresses_center_vertex_ps_tensile!(
        ε::NTuple{3},         # normal components @ centers; shear components @ vertices
        ε_pl::NTuple{3},      # whole Voigt tensor @ centers
        EII,                  # accumulated plastic strain rate @ centers
        τ::NTuple{3},         # whole Voigt tensor @ centers
        τshear_v::NTuple{1},  # shear tensor components @ vertices
        τ_o::NTuple{3},
        τshear_ov::NTuple{1}, # shear tensor components @ vertices
        Pr,
        Pr_c,
        η,
        λ,
        λv,
        τII,
        η_vep,
        relλ,
        dt,
        θ_dτ,
        rheology,
        phase_center,
        phase_vertex,
        pT_center,            # tensile strength @ cell centers [Pa]
        pT_vertex,            # tensile strength @ vertices     [Pa]
        pl_domain,            # plasticity regime @ centers: 0=elastic, 1=DP, 2=cap
        ε_vol_pl,             # volumetric plastic strain rate @ centers (overwritten each iter)
    )
    τxyv     = τshear_v[1]
    τxyv_old = τshear_ov[1]
    ni       = size(Pr)
    Ic       = clamped_indices(ni, I...)

    # ─────────────────────────────────────────────────────────
    ## vertex update
    # ─────────────────────────────────────────────────────────
    Pv_ij        = @inbounds av_clamped(Pr,   Ic...)
    εxxv_ij      = @inbounds av_clamped(ε[1], Ic...)
    εyyv_ij      = @inbounds av_clamped(ε[2], Ic...)
    τxxv_ij      = @inbounds av_clamped(τ[1], Ic...)
    τyyv_ij      = @inbounds av_clamped(τ[2], Ic...)
    τxxv_old_ij  = @inbounds av_clamped(τ_o[1], Ic...)
    τyyv_old_ij  = @inbounds av_clamped(τ_o[2], Ic...)
    EIIv_ij      = @inbounds av_clamped(EII, Ic...)
    pTv_ij       = @inbounds av_clamped(pT_vertex, Ic...)

    phase = @inbounds phase_vertex[I...]
    is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
    _Gvdt  = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
    Kv     = fn_ratio(get_bulk_modulus, rheology, phase)
    ηv_ij  = @inbounds harm_clamped(η, Ic...)
    dτ_rv  = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

    # stress trial increments @ vertex
    dτxxv = compute_stress_increment(τxxv_ij,    τxxv_old_ij,  ηv_ij, εxxv_ij,      _Gvdt, dτ_rv)
    dτyyv = compute_stress_increment(τyyv_ij,    τyyv_old_ij,  ηv_ij, εyyv_ij,      _Gvdt, dτ_rv)
    dτxyv = @inbounds compute_stress_increment(
        τxyv[I...], τxyv_old[I...], ηv_ij, ε[3][I...], _Gvdt, dτ_rv
    )
    τijv     = τxxv_ij, τyyv_ij, τxyv[I...]
    dτijv    = dτxxv, dτyyv, dτxyv
    τijv_tr  = τijv .+ dτijv
    τIIv_ij  = second_invariant(τijv_tr)

    # tensile-cap parameters @ vertex
    cp       = tensile_cap_params(sinϕv, cosϕv, sinψv, Cv, pTv_ij)

    # composite yield function @ vertex
    Fv = composite_yield(τIIv_ij, Pv_ij, cp.k, cp.c, cp.py, cp.Ry, cp.a, cp.pd, cp.τd)

    @inbounds if is_pl && !iszero(τIIv_ij) && Fv > 0
        dQdτijv, dQdpv = composite_flow_gradient(
            τIIv_ij, τijv_tr, Pv_ij,
            cp.k, cp.kf, cp.c, cp.py, cp.a, cp.pd, cp.τd, cp.pq, cp.b, cp.Rf, cp.pdf, cp.sdf
        )
        volumev = isinf(Kv) ? 0.0 : Kv * dt * (-dQdpv)  # volumetric plastic work
        λv[I...] =
            @muladd (1.0 - relλ) * λv[I...] +
            relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
        # only the xy-shear component is stored at the vertex; use its gradient
        εij_plxyv = λv[I...] * dQdτijv[3]
        τxyv[I...] += @muladd dτxyv - 2.0 * ηv_ij * εij_plxyv * dτ_rv
        ε_pl[3][I...] = εij_plxyv
    else
        τxyv[I...] += dτxyv
        ε_pl[3][I...] = 0.0
    end

    # ─────────────────────────────────────────────────────────
    ## center update
    # ─────────────────────────────────────────────────────────
    if all(I .≤ ni)
        pT_ij  = @inbounds pT_center[I...]
        phase  = @inbounds phase_center[I...]
        _Gdt   = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        is_pl, C, sinϕ, cosϕ, sinψ, η_reg = @inbounds plastic_params_phase(rheology, EII[I...], phase)
        K      = fn_ratio(get_bulk_modulus, rheology, phase)
        ηij    = η[I...]
        dτ_r   = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

        # cache visco-elastic tensors @ center
        τij, τij_o, εij, εij_pl = cache_tensors(τ, τ_o, ε, ε_pl, I...)

        # stress trial increments @ center
        dτij    = compute_stress_increment(τij, τij_o, ηij, εij, _Gdt, dτ_r)
        τij_tr  = dτij .+ τij
        τII_ij  = second_invariant(τij_tr)

        # tensile-cap parameters @ center
        cp = tensile_cap_params(sinϕ, cosϕ, sinψ, C, pT_ij)

        # composite yield function @ center
        F = @inbounds composite_yield(τII_ij, Pr[I...], cp.k, cp.c, cp.py, cp.Ry, cp.a, cp.pd, cp.τd)

        # determine and record plasticity regime from trial stress state
        @inbounds begin
            is_active   = is_pl && !iszero(τII_ij) && F > 0
            in_cap      = τII_ij < (cp.py - Pr[I...]) * cp.τd / (cp.py - cp.pd)
            pl_domain[I...] = is_active ? (in_cap ? 2.0 : 1.0) : 0.0
        end

        τII_ij = @inbounds if is_pl && !iszero(τII_ij) && F > 0
            # flow potential gradient @ center
            dQdτij, dQdp = composite_flow_gradient(
                τII_ij, τij_tr, Pr[I...],
                cp.k, cp.kf, cp.c, cp.py, cp.a, cp.pd, cp.τd, cp.pq, cp.b, cp.Rf, cp.pdf, cp.sdf
            )
            volume = isinf(K) ? 0.0 : K * dt * (-dQdp)  # volumetric plastic work

            # viscoplastic multiplier (Perzyna-type PT update)
            λ[I...] =
                @muladd (1.0 - relλ) * λ[I...] +
                relλ * (max(F, 0.0) / (ηij * dτ_r + η_reg + volume))

            εij_pl   = λ[I...] .* dQdτij
            dτij     = @muladd @. dτij - 2.0 * ηij * εij_pl * dτ_r
            τij      = dτij .+ τij

            Base.@nexprs 3 i -> begin
                @inbounds τ[i][I...] = τij[i]
            end
            Base.@nexprs 2 i -> begin
                @inbounds ε_pl[i][I...] = εij_pl[i]
            end
            τII_ij = second_invariant(τij)
        else
            Base.@nexprs 3 i -> begin
                @inbounds τ[i][I...] = dτij[i] .+ τij[i]
            end
            Base.@nexprs 2 i -> begin
                @inbounds ε_pl[i][I...] = 0.0
            end
            τII_ij
        end

        @inbounds τII[I...] = τII_ij
        @inbounds η_vep[I...] = τII_ij * 0.5 * inv(second_invariant(εij))
        # pressure correction: use volumetric flow-potential component
        # In DP regime this reproduces the K*dt*λ*sinψ of the original kernel.
        @inbounds begin
            dQdτij_c, dQdp_c = composite_flow_gradient(
                τII_ij, ntuple(i -> τ[i][I...], Val(3)), Pr[I...],
                cp.k, cp.kf, cp.c, cp.py, cp.a, cp.pd, cp.τd, cp.pq, cp.b, cp.Rf, cp.pdf, cp.sdf
            )
            Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * (-dQdp_c))
            # volumetric plastic strain rate = λ * (-dQ/dp); zero when elastic
            ε_vol_pl[I...] = pl_domain[I...] > 0 ? λ[I...] * (-dQdp_c) : 0.0
        end
    end

    return nothing
end
