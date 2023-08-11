## GeoParams

@inline compute_phase(fn::F, rheology, phase::Int, args) where F = fn(rheology, phase, args)
@inline compute_phase(fn::F, rheology, ::Nothing, args) where F = fn(rheology, args)

@inline Base.@propagate_inbounds getindex_phase(phase::AbstractArray, I::Vararg{Int,N}) where {N} = phase[I...]
@inline getindex_phase(::Nothing, I::Vararg{Int,N}) where {N} = nothing

@inline function compute_diffusivity(rheology, args)
    return compute_conductivity(rheology, args) *
           inv(compute_heatcapacity(rheology, args) * compute_density(rheology, args))
end

@inline function compute_diffusivity(rheology, phase::Union{Nothing, Int}, args)
    return compute_phase(compute_conductivity, rheology, phase, args) * inv(
        compute_phase(compute_heatcapacity, rheology, phase, args) * compute_phase(compute_density, rheology, phase, args)
    )
end

@inline function compute_diffusivity(rheology, ρ, args)
    return compute_conductivity(rheology, args) *
           inv(compute_heatcapacity(rheology, args) * ρ)
end

@inline function compute_diffusivity(rheology, ρ, phase::Union{Nothing, Int}, args)
    return compute_phase(compute_conductivity, rheology, phase, args) *
           inv(compute_phase(compute_heatcapacity, rheology, phase, args) * ρ)
end

@inline function compute_ρCp(rheology, args)
    return compute_heatcapacity(rheology, args) * compute_density(rheology, args)
end

@inline function compute_ρCp(rheology, phase::Union{Nothing, Int}, args)
    return compute_phase(compute_heatcapacity, rheology, phase, args) *
        compute_phase(compute_density, rheology, phase, args)
end

@inline function compute_ρCp(rheology, ρ, args)
    return compute_heatcapacity(rheology, args) * ρ
end

@inline function compute_ρCp(rheology, ρ, phase::Union{Nothing, Int}, args)
    return compute_phase(compute_heatcapacity, rheology, phase, args) * ρ
end

## 3D KERNELS 

@parallel_indices (i, j, k) function compute_flux!(
    qTx::AbstractArray{_T, 3}, qTy, qTz, qTx2, qTy2, qTz2, T, K, θr_dτ, _dx, _dy, _dz
) where _T
    
    d_xa(A) = _d_xa(A, i, j, k, _dx)
    d_ya(A) = _d_ya(A, i, j, k, _dy)
    d_za(A) = _d_za(A, i, j, k, _dz)
    av_xy(A) = _av_xy(A, i, j, k)
    av_xz(A) = _av_xz(A, i, j, k)
    av_yz(A) = _av_yz(A, i, j, k)

    I = i, j, k

    if all(I .≤ size(qTx))
        qx = qTx2[I...] = -av_yz(K) * d_xa(T)
        qTx[I...] = (qTx[I...] * av_yz(θr_dτ) + qx) / (1.0 + av_yz(θr_dτ))
    end

    if all(I .≤ size(qTy))
        qy = qTy2[I...] = -av_xz(K) * d_ya(T)
        qTy[I...] = (qTy[I...] * av_xz(θr_dτ) + qy) / (1.0 + av_xz(θr_dτ))
    end

    if all(I .≤ size(qTz))
        qz = qTz2[I...] = -av_xy(K) * d_za(T)
        qTz[I...] = (qTz[I...] * av_xy(θr_dτ) + qz) / (1.0 + av_xy(θr_dτ))
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_flux!(
    qTx::AbstractArray{_T, 3}, qTy, qTz, qTx2, qTy2, qTz2, T, rheology, phase, θr_dτ, _dx, _dy, _dz, args
) where _T

    d_xa(A) = _d_xa(A, i, j, k, _dx)
    d_ya(A) = _d_ya(A, i, j, k, _dy)
    d_za(A) = _d_za(A, i, j, k, _dz)
    av_xy(A) = _av_xy(A, i, j, k)
    av_xz(A) = _av_xz(A, i, j, k)
    av_yz(A) = _av_yz(A, i, j, k)

    I = i, j, k
    phase_ijk = getindex_phase(phase, I...)
    K = compute_phase(compute_conductivity, rheology, phase_ijk, args)

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

@parallel_indices (i, j, k) function update_T!(
    T::AbstractArray{_T, 3}, Told, qTx, qTy, qTz, ρCp, dτ_ρ, _dt, _dx, _dy, _dz
) where _T
    av(A)   = _av(A, i, j, k)
    d_xa(A) = _d_xa(A, i, j, k, _dx)
    d_ya(A) = _d_ya(A, i, j, k, _dy)
    d_za(A) = _d_za(A, i, j, k, _dz)

    T[i + 1, j + 1, k + 1] += 
        av(dτ_ρ) * (
            (-(d_xa(qTx) + d_ya(qTy) + d_za(qTz))) -
            av(ρCp) * (T[i + 1, j + 1, k + 1] - Told[i + 1, j + 1, k + 1]) * _dt
        )

    return nothing
end

@parallel_indices (i, j, k) function update_T!(
    T::AbstractArray{_T, 3}, Told, qTx, qTy, qTz, rheology, phase, dτ_ρ, _dt, _dx, _dy, _dz, args
) where _T
    av(A)   = _av(A, i, j, k)
    d_xa(A) = _d_xa(A, i, j, k, _dx)
    d_ya(A) = _d_ya(A, i, j, k, _dy)
    d_za(A) = _d_za(A, i, j, k, _dz)

    T_ijk = T[i + 1, j + 1, k + 1]
    args_ijk = (; T = T_ijk, P = av(args.P))
    phase_ijk = getindex_phase(phase, i, j, k)
    
    T[i + 1, j + 1, k + 1] +=
        av(dτ_ρ) * (
            (-(d_xa(qTx) + d_ya(qTy) + d_za(qTz))) -
            compute_ρCp(rheology, phase_ijk, args_ijk) * (T_ijk - Told[i + 1, j + 1, k + 1]) * _dt
        )
    return nothing
end

@parallel_indices (i, j, k) function check_res!(ResT::AbstractArray{_T, 3}, T, Told, qTx2, qTy2, qTz2, ρCp, _dt, _dx, _dy, _dz) where _T

    d_xa(A) = _d_xa(A, i, j, k, _dx)
    d_ya(A) = _d_ya(A, i, j, k, _dy)
    d_za(A) = _d_za(A, i, j, k, _dz)
    av(A) = _av(A, i, j, k)

    ResT[i, j, k] =
        -av(ρCp) * (T[i + 1, j + 1, k + 1] - Told[i + 1, j + 1, k + 1]) * _dt - (d_xa(qTx2) + d_ya(qTy2) + d_za(qTz2))

    return nothing
end

@parallel_indices (i, j, k) function check_res!(ResT::AbstractArray{_T, 3}, T, Told, qTx2, qTy2, qTz2, rheology, phase, _dt, _dx, _dy, _dz, args) where _T

    d_xa(A) = _d_xa(A, i, j, k, _dx)
    d_ya(A) = _d_ya(A, i, j, k, _dy)
    d_za(A) = _d_za(A, i, j, k, _dz)
    av(A) = _av(A, i, j, k)

    T_ijk = T[i + 1, j + 1, k + 1]
    args_ijk = (; T = T_ijk, P = av(args.P))
    phase_ijk = getindex_phase(phase, i, j, k)

    ResT[i, j, k] =
        -compute_ρCp(rheology, phase_ijk, args_ijk) * (T_ijk - Told[i + 1, j + 1, k + 1]) * _dt - (d_xa(qTx2) + d_ya(qTy2) + d_za(qTz2))

    return nothing
end

## 2D KERNELS

@parallel_indices (i, j) function compute_flux!(qTx::AbstractArray{_T, 2}, qTy, qTx2, qTy2, T, K, θr_dτ, _dx, _dy) where _T
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
    qTx::AbstractArray{_T, 2}, qTy, qTx2, qTy2, T, rheology, phase, θr_dτ, _dx, _dy, args
) where _T
    nx = size(θr_dτ, 1)

    d_xa(A) = _d_xa(A, i, j, _dx)
    d_ya(A) = _d_ya(A, i, j, _dy)
    av_xa(A) = (A[clamp(i - 1, 1, nx), j + 1] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av_ya(A) = (A[clamp(i, 1, nx), j] + A[clamp(i - 1, 1, nx), j]) * 0.5

    phase_ij = getindex_phase(phase, i, j)
    K = compute_phase(compute_conductivity, rheology, phase_ij, args)

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

@parallel_indices (i, j) function update_T!(
    T::AbstractArray{_T, 2}, Told, qTx, qTy, ρCp, dτ_ρ, _dt, _dx, _dy
) where _T
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

@parallel_indices (i, j) function update_T!(
    T::AbstractArray{_T, 2}, Told, qTx, qTy, rheology, phase, dτ_ρ, _dt, _dx, _dy, args::NamedTuple
) where _T
    nx, = size(args.P)

    d_xa(A) = _d_xa(A, i, j, _dx)
    d_ya(A) = _d_ya(A, i, j, _dy)
    av_xa_cl(A) = (A[clamp(i - 1, 1, nx), j + 1] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av_ya_cl(A) = (A[clamp(i, 1, nx), j] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av(A) = av_ya_cl(av_xa_cl(A))

    T_ij = T[i + 1, j + 1]
    args_ij = (; T = T_ij, P = av(args.P))
    phase_ij = getindex_phase(phase, i, j)
    
    T[i + 1, j + 1] +=
        av(dτ_ρ) * (
            (-(d_xa(qTx) + d_ya(qTy))) -
            compute_ρCp(rheology, phase_ij, args_ij) * (T_ij - Told[i + 1, j + 1]) * _dt
        )
    return nothing
end

@parallel_indices (i, j) function check_res!(ResT::AbstractArray{_T, 2}, T, Told, qTx2, qTy2, ρCp, _dt, _dx, _dy) where _T
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

@parallel_indices (i, j) function check_res!(ResT::AbstractArray{_T, 2}, T, Told, qTx2, qTy2, rheology, phase, _dt, _dx, _dy, args) where _T
    nx, = size(args.P)

    d_xa(A) = _d_xa(A, i, j, _dx)
    d_ya(A) = _d_ya(A, i, j, _dy)
    av_xa(A) = (A[clamp(i - 1, 1, nx), j + 1] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av_ya(A) = (A[clamp(i, 1, nx), j] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av(A) = av_ya(av_xa(A))

    args_ij = (; T = T[i + 1, j + 1], P = av(args.P))
    phase_ij = getindex_phase(phase, i, j)

    ResT[i, j] =
        -compute_ρCp(rheology, phase_ij, args_ij) * (T[i + 1, j + 1] - Told[i + 1, j + 1]) * _dt - (d_xa(qTx2) + d_ya(qTy2))
    return nothing
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

@inline function _update_ΔT!(ΔT, T, Told, I::Vararg{Int,N}) where {N}
    return ΔT[I...] = T[I...] - Told[I...]
end

### SOLVERS COLLECTION BELOW - THEY SHOULD BE DIMENSION AGNOSTIC

@inline flux_range(nx, ny) = @idx (nx + 3, ny + 1)
@inline flux_range(nx, ny, nz) = @idx (nx, ny, nz)

@inline update_range(nx, ny) = @idx (nx + 1, ny - 1)
@inline update_range(nx, ny, nz) = residual_range(nx, ny, nz)

@inline residual_range(nx, ny) = update_range(nx, ny)
@inline residual_range(nx, ny, nz) = @idx (nx - 1, ny - 1, nz - 1)

function update_T(::Nothing, b_width, thermal, ρCp, pt_thermal, _dt, _di, ni)
    @parallel update_range(ni...) update_T!(
        thermal.T, thermal.Told, @qT(thermal)..., ρCp, pt_thermal.dτ_ρ, _dt, _di...,
    )
end

function update_T(::Nothing, b_width, thermal, rheology, phase, pt_thermal, _dt, _di, ni, args)
    @parallel update_range(ni...) update_T!(
        thermal.T, thermal.Told, @qT(thermal)..., rheology, phase, pt_thermal.dτ_ρ, _dt, _di..., args
    )
end

function update_T(igg, b_width, thermal, ρCp, pt_thermal, _dt, _di, ni)
    # @hide_communication b_width begin # communication/computation overlap
        @parallel update_range(ni...) update_T!(
            thermal.T, thermal.Told, @qT(thermal)..., ρCp, pt_thermal.dτ_ρ, _dt, _di...,
        )
        update_halo!(thermal.T)
    # end
end

function update_T(igg, b_width, thermal, rheology, phase, pt_thermal, _dt, _di, ni, args)
    # @hide_communication b_width begin # communication/computation overlap
        @parallel update_range(ni...) update_T!(
            thermal.T, thermal.Told, @qT(thermal)..., rheology, phase, pt_thermal.dτ_ρ, _dt, _di..., args
        )
        update_halo!(thermal.T)
    # end
end

"""
    heatdiffusion_PT!(thermal, pt_thermal, K, ρCp, dt, di; iterMax, nout, verbose)

Heat diffusion solver using Pseudo-Transient iterations. Both `K` and `ρCp` are n-dimensional arrays.
"""
function heatdiffusion_PT!(
    thermal::ThermalArrays,
    pt_thermal::PTThermalCoeffs,
    thermal_bc::TemperatureBoundaryConditions,
    K::AbstractArray,
    ρCp::AbstractArray,
    dt,
    di;
    igg = nothing,
    b_width = (4, 4, 4),
    iterMax = 50e3,
    nout = 1e3,
    verbose = true
)

    @show igg, nout
    # Compute some constant stuff
    _dt = inv(dt)
    _di = inv.(di)
    _sq_len_RT = inv(sqrt(length(thermal.ResT)))
    ϵ = pt_thermal.ϵ
    ni = size(thermal.Tc)
    @copy thermal.Told thermal.T

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
            @parallel flux_range(ni...) compute_flux!(
                @qT(thermal)..., @qT2(thermal)..., thermal.T, K, pt_thermal.θr_dτ, _di...
            )
            update_T(igg, b_width, thermal, ρCp, pt_thermal, _dt, _di, ni)
            thermal_bcs!(thermal.T, thermal_bc)
        end

        iter += 1

        if iter % nout == 0
            wtime0 += @elapsed begin
                @parallel residual_range(ni...) check_res!(
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
    heatdiffusion_PT!(thermal, pt_thermal, rheology, dt, di; iterMax, nout, verbose)

Heat diffusion solver using Pseudo-Transient iterations.
"""
function heatdiffusion_PT!(
    thermal::ThermalArrays,
    pt_thermal::PTThermalCoeffs,
    thermal_bc::TemperatureBoundaryConditions,
    rheology,
    args::NamedTuple,
    dt,
    di;
    igg = nothing,
    phase = nothing,
    b_width = (4, 4, 4),
    iterMax = 50e3,
    nout = 1e3,
    verbose = true
)

    # Compute some constant stuff
    _dt = inv(dt)
    _di = inv.(di)
    _sq_len_RT = inv(sqrt(length(thermal.ResT)))
    ϵ = pt_thermal.ϵ
    ni = size(thermal.Tc)
    @copy thermal.Told thermal.T

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
            @parallel flux_range(ni...) JustRelax.compute_flux!(
                @qT(thermal)..., @qT2(thermal)..., thermal.T, rheology, phase, pt_thermal.θr_dτ, _di..., args
            )
            update_T(igg, b_width, thermal, rheology, phase, pt_thermal, _dt, _di, ni, args)
            thermal_bcs!(thermal.T, thermal_bc)
        end

        iter += 1

        if iter % nout == 0
            wtime0 += @elapsed begin
                @parallel residual_range(ni...) check_res!(
                    thermal.ResT,
                    thermal.T,
                    thermal.Told,
                    @qT2(thermal)...,
                    rheology,
                    phase,
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