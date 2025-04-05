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
@parallel_indices (I...) function accumulate_tensor!(
        II, tensor::NTuple{N, T}, dt
    ) where {N, T}
    @inbounds II[I...] += second_invariant(getindex.(tensor, I...)...) * dt
    return nothing
end

## Stress invariants
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
        @tensor_center(stokes.ε_pl),
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

function clamped_indices(ni::NTuple{3, Integer}, i, j, k)
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

function av_clamped_yz(A, i0, j0, k0, ic, jc, kc, ::Vararg{Integer, N}) where {N}
    return 0.25 * (A[ic, j0, k0] + A[ic, jc, k0] + A[ic, j0, kc] + A[ic, jc, kc])
end

function av_clamped_xz(A, i0, j0, k0, ic, jc, kc, ::Vararg{Integer, N}) where {N}
    return 0.25 * (A[i0, jc, k0] + A[ic, jc, k0] + A[i0, jc, kc] + A[ic, jc, kc])
end

function av_clamped_xy(A, i0, j0, k0, ic, jc, kc, ::Vararg{Integer, N}) where {N}
    return 0.25 * (A[i0, j0, kc] + A[ic, j0, kc] + A[i0, jc, kc] + A[ic, jc, kc])
end

# on yz
function av_clamped_yz_z(A, i0, j0, k0, ic, jc, kc, i1, j1, k1)
    return 0.25 * (A[ic, jc, k0] + A[i1, jc, k0] + A[ic, jc, kc] + A[i1, jc, kc])
end

function av_clamped_yz_y(A, i0, j0, k0, ic, jc, kc, i1, j1, k1)
    return 0.25 * (A[ic, j0, kc] + A[i1, j0, kc] + A[ic, jc, kc] + A[i1, jc, kc])
end

# on xz
function av_clamped_xz_z(A, i0, j0, k0, ic, jc, kc, i1, j1, k1)
    return 0.25 * (A[ic, jc, k0] + A[ic, j1, k0] + A[ic, jc, kc] + A[ic, j1, kc])
end

function av_clamped_xz_x(A, i0, j0, k0, ic, jc, kc, i1, j1, k1)
    return 0.25 * (A[i0, jc, kc] + A[ic, jc, kc] + A[ic, j1, kc] + A[i0, j1, kc])
end

# on xy
function av_clamped_xy_y(A, i0, j0, k0, ic, jc, kc, i1, j1, k1)
    return 0.25 * (A[ic, j0, kc] + A[ic, jc, kc] + A[ic, j0, k1] + A[ic, jc, k1])
end

function av_clamped_xy_x(A, i0, j0, k0, ic, jc, kc, i1, j1, k1)
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
    if all(I .≤ size(ε[4]))
        # interpolate to ith vertex
        ηv_ij = av_clamped_yz(η, Ic...)
        Pv_ij = av_clamped_yz(Pr, Ic...)
        EIIv_ij = av_clamped_yz(EII, Ic...)
        εxxv_ij = av_clamped_yz(ε[1], Ic...)
        εyyv_ij = av_clamped_yz(ε[2], Ic...)
        εzzv_ij = av_clamped_yz(ε[3], Ic...)
        εyzv_ij = ε[4][I...]
        εxzv_ij = av_clamped_yz_y(ε[5], Ic...)
        εxyv_ij = av_clamped_yz_z(ε[6], Ic...)

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
        if is_pl && !iszero(τIIv_ij) && Fv > 0
            # stress correction @ vertex
            λv[1][I...] =
                (1.0 - relλ) * λv[1][I...] +
                relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))

            dQdτyz = 0.5 * (τyzv_ij + dτyzv) / τIIv_ij
            τyzv[I...] += dτyzv - 2.0 * ηv_ij * λv[1][I...] * dQdτyz * dτ_rv
        else
            # stress correction @ vertex
            τyzv[I...] += dτyzv
        end
    end

    ## xz
    if all(I .≤ size(ε[5]))
        # interpolate to ith vertex
        ηv_ij = av_clamped_xz(η, Ic...)
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
        if is_pl && !iszero(τIIv_ij) && Fv > 0
            # stress correction @ vertex
            λv[2][I...] =
                (1.0 - relλ) * λv[2][I...] +
                relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))

            dQdτxz = 0.5 * (τxzv_ij + dτxzv) / τIIv_ij
            τxzv[I...] += dτxzv - 2.0 * ηv_ij * λv[2][I...] * dQdτxz * dτ_rv
        else
            # stress correction @ vertex
            τxzv[I...] += dτxzv
        end
    end

    ## xy
    if all(I .≤ size(ε[6]))
        # interpolate to ith vertex
        ηv_ij = av_clamped_xy(η, Ic...)
        EIIv_ij = av_clamped_xy(EII, Ic...)
        Pv_ij = av_clamped_xy(Pr, Ic...)
        εxxv_ij = av_clamped_xy(ε[1], Ic...)
        εyyv_ij = av_clamped_xy(ε[2], Ic...)
        εzzv_ij = av_clamped_xy(ε[3], Ic...)
        εyzv_ij = av_clamped_xy_x(ε[4], Ic...)
        εxzv_ij = av_clamped_xy_y(ε[5], Ic...)
        εxyv_ij = ε[6][I...]

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
        if is_pl && !iszero(τIIv_ij) && Fv > 0
            # stress correction @ vertex
            λv[3][I...] =
                (1.0 - relλ) * λv[3][I...] +
                relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))

            dQdτxy = 0.5 * (τxyv_ij + dτxyv) / τIIv_ij
            τxyv[I...] += dτxyv - 2.0 * ηv_ij * λv[3][I...] * dQdτxy * dτ_rv
        else
            # stress correction @ vertex
            τxyv[I...] += dτxyv
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
        τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

        # visco-elastic strain rates @ center
        εij_ve = @. εij + 0.5 * τij_o * _Gdt
        εII_ve = second_invariant(εij_ve)
        # stress increments @ center
        dτij = @. (-(τij - τij_o) * ηij * _Gdt - τij + 2.0 * ηij * εij) * dτ_r
        τII_ij = second_invariant(dτij .+ τij)
        # yield function @ center
        F = τII_ij - C * cosϕ - Pr[I...] * sinϕ

        if is_pl && !iszero(τII_ij) && F > 0
            # stress correction @ center
            λ[I...] =
                (1.0 - relλ) * λ[I...] +
                relλ * (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
            dQdτij = @. 0.5 * (τij + dτij) / τII_ij
            εij_pl = λ[I...] .* dQdτij
            dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
            τij = dτij .+ τij
            setindex!.(τ, τij, I...)
            setindex!.(ε_pl, εij_pl, I...)
            τII[I...] = τII_ij = second_invariant(τij)
        else
            # stress correction @ center
            setindex!.(τ, dτij .+ τij, I...)
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
    ni = size(Pr)
    Ic = clamped_indices(ni, I...)

    # interpolate to ith vertex
    Pv_ij = av_clamped(Pr, Ic...)
    εxxv_ij = av_clamped(ε[1], Ic...)
    εyyv_ij = av_clamped(ε[2], Ic...)
    τxxv_ij = av_clamped(τ[1], Ic...)
    τyyv_ij = av_clamped(τ[2], Ic...)
    τxxv_old_ij = av_clamped(τ_o[1], Ic...)
    τyyv_old_ij = av_clamped(τ_o[2], Ic...)
    EIIv_ij = av_clamped(EII, Ic...)

    ## vertex
    phase = @inbounds phase_vertex[I...]
    is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
    _Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
    Kv = fn_ratio(get_bulk_modulus, rheology, phase)
    volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
    ηv_ij = av_clamped(η, Ic...)
    dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

    # stress increments @ vertex
    dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
    dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
    dτxyv = compute_stress_increment(
        τxyv[I...], τxyv_old[I...], ηv_ij, ε[3][I...], _Gvdt, dτ_rv
    )
    τIIv_ij = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)

    # yield function @ center
    Fv = τIIv_ij - Cv * cosϕv - Pv_ij * sinϕv

    if is_pl && !iszero(τIIv_ij)  && Fv > 0
        # stress correction @ vertex
        λv[I...] =
            (1.0 - relλ) * λv[I...] +
            relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
        dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij
        εij_pl = λv[I...] * dQdτxy
        τxyv[I...] += dτxyv - 2.0 * ηv_ij * εij_pl * dτ_rv
    else
        # stress correction @ vertex
        τxyv[I...] += dτxyv
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
        dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

        # cache strain rates for center calculations
        τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

        # visco-elastic strain rates @ center
        εij_ve = @. εij + 0.5 * τij_o * _Gdt
        εII_ve = GeoParams.second_invariant(εij_ve)
        # stress increments @ center
        dτij = compute_stress_increment(τij, τij_o, ηij, εij, _Gdt, dτ_r)
        τII_ij = GeoParams.second_invariant(dτij .+ τij)
        # yield function @ center
        F = τII_ij - C * cosϕ - Pr[I...] * sinϕ

        τII_ij = if is_pl && !iszero(τII_ij) && F > 0
            # stress correction @ center
            λ[I...] =
                (1.0 - relλ) * λ[I...] +
                relλ * (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
            dQdτij = @. 0.5 * (τij + dτij) / τII_ij
            εij_pl = λ[I...] .* dQdτij
            dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
            τij = dτij .+ τij
            setindex!.(τ, τij, I...)
            setindex!.(ε_pl, εij_pl, I...)
            τII_ij = GeoParams.second_invariant(τij)
        else
            # stress correction @ center
            setindex!.(τ, dτij .+ τij, I...)
            τII_ij
        end
        τII[I...] = τII_ij

        η_vep[I...] = τII_ij * 0.5 * inv(second_invariant(εij))
        Pr_c[I...] = Pr[I...] + volume * λ[I...]
    end

    return nothing
end

function clamped_indices(ni::NTuple{2, Integer}, i, j)
    nx, ny = ni
    i0 = clamp(i - 1, 1, nx)
    ic = clamp(i, 1, nx)
    j0 = clamp(j - 1, 1, ny)
    jc = clamp(j, 1, ny)
    return i0, j0, ic, jc
end

function av_clamped(A, i0, j0, ic, jc)
    return 0.25 * (A[i0, j0] + A[ic, jc] + A[i0, jc] + A[ic, j0])
end