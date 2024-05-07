# Viscous

@parallel_indices (i, j) function compute_τ!(
    τxx::AbstractArray{T,2}, τyy, τxy, εxx, εyy, εxy, η, θ_dτ
) where {T}
    @inline av(A) = _av_a(A, i, j)

    denominator = inv(θ_dτ + 1.0)
    η_ij = η[i, j]

    # Normal components
    τxx[i, j] += (-τxx[i, j] + 2.0 * η_ij * εxx[i, j]) * denominator
    τyy[i, j] += (-τyy[i, j] + 2.0 * η_ij * εyy[i, j]) * denominator
    # Shear components
    if all((i, j) .< size(τxy) .- 1)
        τxy[i + 1, j + 1] +=
            (-τxy[i + 1, j + 1] + 2.0 * av(η) * εxy[i + 1, j + 1]) * denominator
    end

    return nothing
end

# Visco-elastic

@parallel_indices (i, j) function compute_τ!(
    τxx::AbstractArray{T,2}, τyy, τxy, τxx_o, τyy_o, τxy_o, εxx, εyy, εxy, η, G, θ_dτ, dt
) where {T}
    @inline av(A) = _av_a(A, i, j)

    # Normal components
    _Gdt = inv(G[i, j] * dt)
    η_ij = η[i, j]
    denominator = inv(θ_dτ + η_ij * _Gdt + 1.0)
    τxx[i, j] +=
        (-(τxx[i, j] - τxx_o[i, j]) * η_ij * _Gdt - τxx[i, j] + 2.0 * η_ij * εxx[i, j]) *
        denominator
    τyy[i, j] +=
        (-(τyy[i, j] - τyy_o[i, j]) * η_ij * _Gdt - τyy[i, j] + 2.0 * η_ij * εyy[i, j]) *
        denominator

    # Shear components
    if all((i, j) .< size(τxy) .- 1)
        av_η_ij = av(η)
        _av_Gdt = inv(av(G) * dt)
        denominator = inv(θ_dτ + av_η_ij * _av_Gdt + 1.0)
        τxy[i + 1, j + 1] +=
            (
                -(τxy[i + 1, j + 1] - τxy_o[i + 1, j + 1]) * av_η_ij * _av_Gdt +
                2.0 * av_η_ij * εxy[i + 1, j + 1]
            ) * denominator
    end

    return nothing
end

@parallel_indices (i, j) function compute_τ!(
    τxx::AbstractArray{T,2}, # centers
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
    τxy::AbstractArray{T,2}, εxy, η, θ_dτ
) where {T}
    @inline av(A) = _av_a(A, i, j)

    # Shear components
    if all((i, j) .< size(τxy) .- 1)
        I = i + 1, j + 1
        av_η_ij = av(η)
        denominator = inv(θ_dτ + 1.0)

        τxy[I...] += (-τxy[I...] + 2.0 * av_η_ij * εxy[I...]) * denominator
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_τ!(
    τxx::AbstractArray{T,3},
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

    @inbounds begin
        if all((i, j, k) .≤ size(τxx))
            _Gdt = inv(get(G) * dt)
            η_ij = get(η)
            denominator = inv(θ_dτ + η_ij * _Gdt + 1.0)
            # Compute τ_xx
            τxx[i, j, k] +=
                (
                    -(get(τxx) - get(τxx_o)) * η_ij * _Gdt - get(τxx) +
                    2.0 * η_ij * get(εxx)
                ) * denominator
            # Compute τ_yy
            τyy[i, j, k] +=
                (
                    -(get(τyy) - get(τyy_o)) * η_ij * _Gdt - get(τyy) +
                    2.0 * η_ij * get(εyy)
                ) * denominator
            # Compute τ_zz
            τzz[i, j, k] +=
                (
                    -(get(τzz) - get(τzz_o)) * η_ij * _Gdt - get(τzz) +
                    2.0 * η_ij * get(εzz)
                ) * denominator
        end
        # Compute τ_xy
        if (1 < i < size(τxy, 1)) && (1 < j < size(τxy, 2)) && k ≤ size(τxy, 3)
            _Gdt = inv(harm_xy(G) * dt)
            η_ij = harm_xy(η)
            denominator = inv(θ_dτ + η_ij * _Gdt + 1.0)
            τxy[i, j, k] +=
                (
                    -(get(τxy) - get(τxy_o)) * η_ij * _Gdt - get(τxy) +
                    2.0 * η_ij * get(εxy)
                ) * denominator
        end
        # Compute τ_xz
        if (1 < i < size(τxz, 1)) && j ≤ size(τxz, 2) && (1 < k < size(τxz, 3))
            _Gdt = inv(harm_xz(G) * dt)
            η_ij = harm_xz(η)
            denominator = inv(θ_dτ + η_ij * _Gdt + 1.0)
            τxz[i, j, k] +=
                (
                    -(get(τxz) - get(τxz_o)) * η_ij * _Gdt - get(τxz) +
                    2.0 * η_ij * get(εxz)
                ) * denominator
        end
        # Compute τ_yz
        if i ≤ size(τyz, 1) && (1 < j < size(τyz, 2)) && (1 < k < size(τyz, 3))
            _Gdt = inv(harm_yz(G) * dt)
            η_ij = harm_yz(η)
            denominator = inv(θ_dτ + η_ij * _Gdt + 1.0)
            τyz[i, j, k] +=
                (
                    -(get(τyz) - get(τyz_o)) * η_ij * _Gdt - get(τyz) +
                    2.0 * η_ij * get(εyz)
                ) * denominator
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
)

    # numerics
    ηij = η[I...]
    _Gdt = inv(get_shear_modulus(rheology[1]) * dt)
    dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)

    # get plastic parameters (if any...)
    is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], 1)

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
    II, tensor::NTuple{N,T}, dt
) where {N,T}
    @inbounds II[I...] += second_invariant(getindex.(tensor, I...)...) * dt
    return nothing
end

## Stress invariants
@parallel_indices (I...) function tensor_invariant_center!(
    II, tensor::NTuple{N,T}
) where {N,T}
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

function update_stress!(stokes, θ, λ, phase_ratios, rheology, dt, θ_dτ)
    return update_stress!(
        islinear(rheology), stokes, θ, λ, phase_ratios, rheology, dt, θ_dτ
    )
end

function update_stress!(
    ::LinearRheologyTrait, stokes, ::Any, ::Any, phase_ratios, rheology, dt, θ_dτ
)
    dim(::AbstractArray{T,N}) where {T,N} = N

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
    nDim = Val(dim(stokes.viscosity.η))

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
    λ::AbstractArray{T,N},
    phase_ratios,
    rheology,
    dt,
    θ_dτ,
) where {N,T}
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
