struct ThermalParameters2{T}
    K::T # thermal conductivity
    ρCp::T # density * heat capacity
end

@inline function _update_ΔT!(ΔT, T, Told, I::Vararg{Int,N}) where {N}
    return ΔT[I...] = T[I...] - Told[I...]
end

@parallel_indices (i) function update_ΔT!(ΔT, T, Told)
    _update_ΔT!(ΔT, T, Told, i)
    return nothing
end

@parallel_indices (i, j) function update_ΔT!(ΔT, T, Told)
    _update_ΔT!(ΔT, T, Told, i, j)
    return nothing
end

@parallel_indices (i, j, k) function update_ΔT!(ΔT, T, Told)
    _update_ΔT!(ΔT, T, Told, i, j, k)
    return nothing
end

## GeoParams

@inline function compute_diffusivity(rheology, args)
    return compute_conductivity(rheology, args) *
           inv(compute_heatcapacity(rheology, args) * compute_density(rheology, args))
end

@inline function compute_diffusivity(rheology, phase::Int, args)
    return compute_conductivity(rheology, phase, args) * inv(
        compute_heatcapacity(rheology, phase, args) * compute_density(rheology, phase, args)
    )
end

@inline function compute_diffusivity(rheology, ρ, args)
    return compute_conductivity(rheology, args) *
           inv(compute_heatcapacity(rheology, args) * ρ)
end

@inline function compute_diffusivity(rheology, ρ, phase::Int, args)
    return compute_conductivity(rheology, phase, args) *
           inv(compute_heatcapacity(rheology, phase, args) * ρ)
end

@inline function compute_ρCp(rheology, args)
    return compute_heatcapacity(rheology, args) * compute_density(rheology, args)
end

@inline function compute_ρCp(rheology, phase::Int, args)
    return compute_heatcapacity(rheology, phase, args) *
           compute_density(rheology, phase, args)
end

@inline function compute_ρCp(rheology, ρ, args)
    return compute_heatcapacity(rheology, args) * ρ
end

@inline function compute_ρCp(rheology, ρ, phase::Int, args)
    return compute_heatcapacity(rheology, phase, args) * ρ
end

## 3D KERNELS 

@parallel_indices (i, j, k) function compute_flux!(
    qTx, qTy, qTz, qTx2, qTy2, qTz2, T, K, θr_dτ, _dx, _dy, _dz
)
    
    d_xa(A) = _dx(A, i, j, k)
    d_ya(A) = _dy(A, i, j, k)
    d_za(A) = _dz(A, i, j, k)
    av_xy(A) = _av_xy(A, i, j, k)
    av_xz(A) = _av_xz(A, i, j, k)
    av_yz(A) = _av_yz(A, i, j, k)

    I = i, j, k

    @inbounds if all(I .≤ size(qTx))
        qx = qTx2[I...] = -av_yz(K) * d_xa(T) * _dx
        qTx[I...] = (qTx[I...] * av_yz(θr_dτ) + qx) / (1.0 + av_yz(θr_dτ))
    end

    @inbounds if all(I .≤ size(qTy))
        qy = qTy2[I...] = -av_xz(K) * d_ya(T) * _dy
        qTy[I...] = (qTy[I...] * av_xz(θr_dτ) + qy) / (1.0 + av_xz(θr_dτ))
    end

    @inbounds if all(I .≤ size(qTz))
        qz = qTz2[I...] = -av_xy(K) * d_za(T) * _dz
        qTz[I...] = (qTz[I...] * av_xy(θr_dτ) + qz) / (1.0 + av_xy(θr_dτ))
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_flux!(
    qTx, qTy, qTz, qTx2, qTy2, qTz2, T, rheology, θr_dτ, _dx, _dy, _dz, args
)
    
    d_xa(A) = _dx(A, i, j, k) * _dx
    d_ya(A) = _dy(A, i, j, k) * _dy
    d_za(A) = _dz(A, i, j, k) * _dz
    av_xy(A) = _av_xy(A, i, j, k)
    av_xz(A) = _av_xz(A, i, j, k)
    av_yz(A) = _av_yz(A, i, j, k)

    I = i, j, k
    K = compute_conductivity(rheology, args)

    @inbounds if all(I .≤ size(qTx))
        qx = qTx2[I...] = -K * d_xa(T)
        qTx[I...] = (qTx[I...] * av_yz(θr_dτ) + qx) / (1.0 + av_yz(θr_dτ))
    end

    @inbounds if all(I .≤ size(qTy))
        qy = qTy2[I...] = -K * d_ya(T)
        qTy[I...] = (qTy[I...] * av_xz(θr_dτ) + qy) / (1.0 + av_xz(θr_dτ))
    end

    @inbounds if all(I .≤ size(qTz))
        qz = qTz2[I...] = -K * d_za(T)
        qTz[I...] = (qTz[I...] * av_xy(θr_dτ) + qz) / (1.0 + av_xy(θr_dτ))
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_update!(
    T, Told, qTx, qTy, qTz, ρCp, dτ_ρ, _dt, _dx, _dy, _dz
)
    av(A)   = _av(A, i, j, k)
    d_xa(A) = _dx(A, i, j, k) * _dx
    d_ya(A) = _dy(A, i, j, k) * _dy
    d_za(A) = _dz(A, i, j, k) * _dz

    T[i + 1, j + 1, k + 1] +=
        av(dτ_ρ) * (
            (-(d_xa(qTx) + d_ya(qTy) + d_ya(qTz))) -
            av(ρCp) * (T[i + 1, j + 1, k + 1] - Told[i + 1, j + 1, k + 1]) * _dt
        )
    return nothing
end

@parallel_indices (i, j, k) function compute_update!(
    T, Told, qTx, qTy, qTz, rheology, dτ_ρ, _dt, _dx, _dy, _dz, args
)
    av(A)   = _av(A, i, j, k)
    d_xa(A) = _dx(A, i, j, k) * _dx
    d_ya(A) = _dy(A, i, j, k) * _dy
    d_za(A) = _dz(A, i, j, k) * _dz

    T_ijk = T[i + 1, j + 1, k + 1]
    args_ijk = (; T = T_ijk, P = av(args.P))

    T[i + 1, j + 1, k + 1] +=
        av(dτ_ρ) * (
            (-(d_xa(qTx) + d_ya(qTy) + d_ya(qTz))) -
            compute_ρCp(rheology, args_ijk) * (T_ijk - Told[i + 1, j + 1, k + 1]) * _dt
        )
    return nothing
end

## 2D KERNELS

@parallel_indices (i, j) function compute_flux!(qTx, qTy, qTx2, qTy2, T, K, θr_dτ, _dx, _dy)
    nx = size(θr_dτ, 1)

    d_xa(A) = _d_xa(A, i, j, _dx)
    d_ya(A) = _d_ya(A, i, j, _dy)
    av_xa(A) = (A[clamp(i - 1, 1, nx), j + 1] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av_ya(A) = (A[clamp(i, 1, nx), j] + A[clamp(i - 1, 1, nx), j]) * 0.5

    @inbounds if all((i, j) .≤ size(qTx))
        qx = qTx2[i, j] = -av_xa(K) * d_xa(T)
        qTx[i, j] = (qTx[i, j] * av_xa(θr_dτ) + qx) / (1.0 + av_xa(θr_dτ))
    end

    @inbounds if all((i, j) .≤ size(qTy))
        qy = qTy2[i, j] = -av_ya(K) * d_ya(T)
        qTy[i, j] = (qTy[i, j] * av_ya(θr_dτ) + qy) / (1.0 + av_ya(θr_dτ))
    end
    return nothing
end

@parallel_indices (i, j) function compute_flux!(
    qTx, qTy, qTx2, qTy2, T, rheology, θr_dτ, _dx, _dy, args
)
    nx = size(θr_dτ, 1)

    d_xa(A) = _d_xa(A, i, j, _dx)
    d_ya(A) = _d_ya(A, i, j, _dy)
    av_xa(A) = (A[clamp(i - 1, 1, nx), j + 1] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av_ya(A) = (A[clamp(i, 1, nx), j] + A[clamp(i - 1, 1, nx), j]) * 0.5

    K = compute_conductivity(rheology, args)

    @inbounds if all((i, j) .≤ size(qTx))
        qx = qTx2[i, j] = -K * d_xa(T)
        qTx[i, j] = (qTx[i, j] * av_xa(θr_dτ) + qx) / (1.0 + av_xa(θr_dτ))
    end

    @inbounds if all((i, j) .≤ size(qTy))
        qy = qTy2[i, j] = -K * d_ya(T)
        qTy[i, j] = (qTy[i, j] * av_ya(θr_dτ) + qy) / (1.0 + av_ya(θr_dτ))
    end
    return nothing
end

@parallel_indices (i, j) function compute_update!(
    T, Told, qTx, qTy, ρCp, dτ_ρ, _dt, _dx, _dy
)
    nx, = size(ρCp)

    d_xa(A) = _d_xa(A, i, j, _dx)
    d_ya(A) = _d_ya(A, i, j, _dy)
    av_xa(A) = (A[clamp(i - 1, 1, nx), j + 1] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av_ya(A) = (A[clamp(i, 1, nx), j] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av(A) = av_ya(av_xa(A))

    T[i + 1, j + 1] +=
        av(dτ_ρ) * (
            (-(d_xa(qTx) + d_ya(qTy))) -
            av(ρCp) * (T[i + 1, j + 1] - Told[i + 1, j + 1]) * _dt
        )
    return nothing
end

@parallel_indices (i, j) function compute_update!(
    T, Told, qTx, qTy, rheology, dτ_ρ, _dt, _dx, _dy, args
)
    nx, = size(args.P)

    d_xa(A) = _d_xa(A, i, j, _dx)
    d_ya(A) = _d_ya(A, i, j, _dy)
    av_xa_cl(A) = (A[clamp(i - 1, 1, nx), j + 1] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av_ya_cl(A) = (A[clamp(i, 1, nx), j] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av(A) = av_ya_cl(av_xa_cl(A))

    T_ij = T[i + 1, j + 1]
    args_ij = (; T = T_ij, P = av(args.P))

    T[i + 1, j + 1] +=
        av(dτ_ρ) * (
            (-(d_xa(qTx) + d_ya(qTy))) -
            compute_ρCp(rheology, args_ij) * (T_ij - Told[i + 1, j + 1]) * _dt
        )
    return nothing
end

@parallel_indices (i, j) function check_res!(ResT, T, Told, qTx2, qTy2, ρCp, _dt, _dx, _dy)
    nx, = size(ρCp)

    d_xa(A) = _d_xa(A, i, j, _dx)
    d_ya(A) = _d_ya(A, i, j, _dy)
    av_xa(A) = (A[clamp(i - 1, 1, nx), j + 1] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av_ya(A) = (A[clamp(i, 1, nx), j] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av(A) = av_ya(av_xa(A))

    ResT[i, j] =
        -av(ρCp) * (T[i + 1, j + 1] - Told[i + 1, j + 1]) * _dt - (d_xa(qTx2) + d_ya(qTy2))
    return nothing
end

@parallel_indices (i, j) function check_res!(ResT, T, Told, qTx2, qTy2, rheology, _dt, _dx, _dy, args)
    nx, = size(ρCp)

    d_xa(A) = _d_xa(A, i, j, _dx)
    d_ya(A) = _d_ya(A, i, j, _dy)
    av_xa(A) = (A[clamp(i - 1, 1, nx), j + 1] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av_ya(A) = (A[clamp(i, 1, nx), j] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av(A) = av_ya(av_xa(A))

    args_ij = (; T = T[i + 1, j + 1], P = av(args.P))

    ResT[i, j] =
        -compute_ρCp(rheology, args_ij) * (T[i + 1, j + 1] - Told[i + 1, j + 1]) * _dt - (d_xa(qTx2) + d_ya(qTy2))
    return nothing
end

### SOLVERS COLLECTION BELOW - THEY SHOULD BE DIMENSION AGNOSTIC

"""
    diffusion_PT!(thermal, pt_thermal, K, ρCp, dt, di; iterMax, nout, verbose)

Heat diffusion solver using Pseudo-Transient iterations. Both `K` and `ρCp` are n-dimensional arrays.
"""
function diffusion_PT!(
    thermal::ThermalArrays,
    pt_thermal::PTThermalCoeffs,
    thermal_bc::TemperatureBoundaryConditions,
    K::AbstractArray,
    ρCp::AbstractArray,
    dt,
    di;
    iterMax=50e3,
    nout = 1e3,
    verbose = true
)
    idx_range(nx, ny) = @idx (nx + 1, ny - 1)

    # Compute some constant stuff
    _dt = inv(dt)
    _di = inv.(di)
    _sq_len_RT = inv(sqrt(length(thermal.ResT)))
    ϵ = pt_thermal.ϵ
    nx, ny = size(thermal.Tc)
    # errors 
    iter_count = Int64[]
    norm_ResT = Float64[]
    sizehint!(iter_count, Int(iterMax))
    sizehint!(norm_ResT, Int(iterMax))

    # Pseudo-transient iteration
    iter = 0
    wtime0 = 0e0
    err = 2 * ϵ
    iterMax = 10e3

    println("\n ====================================\n")
    println("Starting thermal diffusion solver...\n")

    while err > ϵ && iter < iterMax
        wtime0 += @elapsed begin
            @parallel compute_flux!(
                @qT(thermal)..., @qT2(thermal)..., thermal.T, K, pt_thermal.θr_dτ, _di...
            )

            @parallel idx_range(ni...) compute_update!(
                thermal.T, thermal.Told, @qT(thermal)..., ρCp, pt_thermal.dτ_ρ, _dt, _di...
            )

            thermal_bcs!(thermal.T, thermal_bc)
        end

        iter += 1

        if iter % nout == 0
            wtime0 += @elapsed begin
                @parallel idx_range(ni...) check_res!(
                    thermal.ResT,
                    thermal.T,
                    thermal.Told,
                    @qT2(thermal)...,
                    ρCp,
                    _dt,
                    _di...,
                )
            end

            err = norm(thermal.ResT) * _sq_len_RT

            push!(norm_ResT, err)
            push!(iter_count, iter)

            if verbose && (err < ϵ) 
                @printf("iter = %d, err = %1.3e \n", iter, err)
            end
        end
    end

    println("\n ...solver finished in $(round(wtime0, sigdigits=5)) seconds \n")
    println("====================================\n")
    
    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)

    @parallel (@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)

    return nothing
end

"""
    diffusion_PT!(thermal, pt_thermal, rheology, dt, di; iterMax, nout, verbose)

Heat diffusion solver using Pseudo-Transient iterations.
"""
function diffusion_PT!(
    thermal::ThermalArrays,
    pt_thermal::PTThermalCoeffs,
    thermal_bc::TemperatureBoundaryConditions,
    rheology,
    args::NamedTuple,
    dt,
    di;
    iterMax=50e3,
    nout = 1e3,
    verbose = true
)
    idx_range(nx, ny) = @idx (nx + 1, ny - 1)

    # Compute some constant stuff
    _dt = inv(dt)
    _di = inv.(di)
    _sq_len_RT = inv(sqrt(length(thermal.ResT)))
    ϵ = pt_thermal.ϵ
    nx, ny = size(thermal.Tc)
    # errors 
    iter_count = Int64[]
    norm_ResT = Float64[]
    sizehint!(iter_count, Int(iterMax))
    sizehint!(norm_ResT, Int(iterMax))

    # Pseudo-transient iteration
    iter = 0
    wtime0 = 0e0
    err = 2 * ϵ
    iterMax = 10e3

    println("\n ====================================\n")
    println("Starting thermal diffusion solver...\n")

    while err > ϵ && iter < iterMax
        wtime0 += @elapsed begin
            @parallel (@idx size(thermal.T).-1) compute_flux!(
                @qT(thermal)..., @qT2(thermal)..., thermal.T, rheology, pt_thermal.θr_dτ, _di..., args
            )

            @parallel idx_range(ni...) compute_update!(
                thermal.T, thermal.Told, @qT(thermal)..., rheology, pt_thermal.dτ_ρ, _dt, _di..., args
            )

            thermal_bcs!(thermal.T, thermal_bc)
        end

        iter += 1

        if iter % nout == 0
            wtime0 += @elapsed begin
                @parallel idx_range(ni...) check_res!(
                    thermal.ResT,
                    thermal.T,
                    thermal.Told,
                    @qT2(thermal)...,
                    rheology,
                    _dt,
                    _di...,
                    args
                )
            end

            err = norm(thermal.ResT) * _sq_len_RT

            push!(norm_ResT, err)
            push!(iter_count, iter)

            if verbose && (err < ϵ) 
                @printf("iter = %d, err = %1.3e \n", iter, err)
            end
        end
    end

    println("\n ...solver finished in $(round(wtime0, sigdigits=5)) seconds \n")
    println("====================================\n")
    
    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)
    @parallel (@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)

    return nothing
end