# using JustRelax, JustRelax.JustRelax2D
using CellArrays
# using Test 
# using ParallelStencil
# @init_parallel_stencil(Threads, Float64, 2)
abstract type AbstractMask end

struct RockRatio{T, N} <: AbstractMask
    center::T
    vertex::T
    Vx::T
    Vy::T
    Vz::Union{Nothing, T}
    yz::Union{Nothing, T}
    xz::Union{Nothing, T}
    xy::Union{Nothing, T}
    
    function RockRatio(center::AbstractArray{F, N}, vertex::T, Vx::T, Vy::T, Vz::Union{Nothing, T}, yz::Union{Nothing, T}, xz::Union{Nothing, T}, xy::Union{Nothing, T}) where {F, N, T}
        new{T, N}(center, vertex, Vx, Vy, Vz, yz, xz, xy)
    end
end

RockRatio(ni::NTuple{N, Integer}) where N = RockRatio(ni...)

function RockRatio(nx, ny)
    ni      = nx, ny
    center  = @zeros(ni...)
    vertex  = @zeros(ni.+1...)
    Vx      = @zeros(nx+1, ny) # no ghost nodes!
    Vy      = @zeros(nx, ny+1) # no ghost nodes!

    return RockRatio(center, vertex, Vx, Vy, nothing, nothing, nothing, nothing)
end

function RockRatio(nx, ny, nz)
    ni      = nx, ny, nz
    center  = @zeros(ni...)
    vertex  = @zeros(ni.+1...)
    Vx      = @zeros(nx+1, ny, nz) # no ghost nodes!
    Vy      = @zeros(nx, ny+1, nz) # no ghost nodes!
    Vz      = @zeros(nx, ny, nz+1) # no ghost nodes!
    yz      = @zeros(nx, ny + 1, nz + 1)
    xz      = @zeros(nx + 1, ny, nz + 1)
    xy      = @zeros(nx + 1, ny + 1, nz)

    return RockRatio(center, vertex, Vx, Vy, Vz, yz, xz, xy)
end

@inline size_c(x::AbstractMask)  = size(x.center)
@inline size_v(x::AbstractMask)  = size(x.vertex)
@inline size_vx(x::AbstractMask) = size(x.Vx)
@inline size_vy(x::AbstractMask) = size(x.Vy)
@inline size_vz(x::AbstractMask) = size(x.Vz)
@inline size_yz(x::AbstractMask) = size(x.yz)
@inline size_xz(x::AbstractMask) = size(x.xz)
@inline size_xy(x::AbstractMask) = size(x.xy)

@inline compute_rock_ratio(phase_ratio::CellArray, air_phase, I::Vararg{Integer, N}) where N = 1 - @index phase_ratio[air_phase, I...]
@inline compute_air_ratio(phase_ratio::CellArray, air_phase, I::Vararg{Integer, N}) where N  = @index phase_ratio[air_phase, I...]

@parallel_indices (I...) function update_rock_ratio_cv!(ϕ, ratio_center, ratio_vertex, air_phase)
    if all(I .≤ size(ratio_center))
        ϕ.center[I...] = Float64(Float16(compute_rock_ratio(ratio_center, air_phase, I...)))
    end
    ϕ.vertex[I...] = Float64(Float16(compute_rock_ratio(ratio_vertex, air_phase, I...)))
    return nothing
end

@parallel_indices (I...) function update_rock_ratio_vel!(ϕ::RockRatio{T, N}) where {T, N}
    # 2D
    @inline av_x(A::AbstractArray{T, 2}) where T = _av_xa(A, I...)
    @inline av_y(A::AbstractArray{T, 2}) where T = _av_ya(A, I...)
    # 3D
    @inline av_x(A::AbstractArray{T, 3}) where T = _av_yz(A, I...)
    @inline av_y(A::AbstractArray{T, 3}) where T = _av_xz(A, I...)
    @inline av_z(A::AbstractArray{T, 3}) where T = _av_xy(A, I...)

    all(I .≤ size(ϕ.Vx)) && (ϕ.Vx[I...] = av_y(ϕ.vertex))
    all(I .≤ size(ϕ.Vy)) && (ϕ.Vy[I...] = av_x(ϕ.vertex))
    if N === 3 # control flow here, so that the branch can be removed by the compiler in the 2D case 
        all(I .≤ size(ϕ.Vy)) && (ϕ.Vy[I...] = av_x(ϕ.vertex))
    end
    return nothing
end

function update_rock_ratio!(ϕ::RockRatio, phase_ratios, air_phase)
    nvi = size_v(ϕ)

    @parallel (@idx nvi)  update_rock_ratio_cv!(ϕ, phase_ratios.center, phase_ratios.vertex, air_phase)
    @parallel (@idx nvi) update_rock_ratio_vel!(ϕ)
    
    return nothing
end

# true means that is a nullspace and needs to be removed
# @inline Base.@propagate_inbounds isvalid(A::T, I::Vararg{Integer, N}) where {N, T<:AbstractArray} = iszero(A[I...])

# nullspace: check whether any point of the stencil is 0, if so, eliminate equation
function isvalid(A::T, I::Vararg{Integer, N}) where {N, T<:AbstractArray}
    v = true
    Base.@nexprs 2 j -> begin
        Base.@nexprs 2 i -> begin
            @inline
            ii = clamp(I[1] + 2 * i - 3, 1, size(A, 1))
            jj = clamp(I[2] + 2 * j - 3, 1, size(A, 2))
            v *= if N == 3
                A[ii, jj, I[3]] > 0
            else
                A[ii, jj] > 0
            end
        end
    end
    if N === 3
        Base.@nexprs 2 k -> begin    
            kk = clamp(I[3] + 2 * k - 3, 1, size(A, 3))
            v *= A[ii, jj, kk] > 0
        end
    end
    return v * (A[I...] > 0)
end

function isvalid_c(ϕ::RockRatio, i, j)
    vx  = (ϕ.Vx[i, j] > 0) * (ϕ.Vx[i + 1, j] > 0)
    vy  = (ϕ.Vy[i, j] > 0) * (ϕ.Vy[i, j + 1] > 0)
    v = vx * vy
    # return v * (ϕ.center[i, j] > 0)
    return true
end

function isvalid_v(ϕ::RockRatio, i, j)
    nx, ny = size(ϕ.Vx)
    j_bot  = max(j - 1, 1)
    j0     = min(j, ny)
    vx     = (ϕ.Vx[i, j0] > 0) * (ϕ.Vx[i, j_bot] > 0)
    
    nx, ny  = size(ϕ.Vy)
    i_left  = max(i - 1, 1)
    i0      = min(i, nx)
    vy      = (ϕ.Vy[i0, j] > 0) * (ϕ.Vy[i_left, j] > 0)
    v       = vx * vy
    # return v * (ϕ.vertex[i, j] > 0)
    return true
end

function isvalid_vx(ϕ::RockRatio, i, j)
    c  = (ϕ.center[i, j] > 0) * (ϕ.center[i-1, j] > 0)
    v  = (ϕ.vertex[i, j] > 0) * (ϕ.vertex[i, j+1] > 0)
    cv = c * v
    return cv * (ϕ.Vx[i, j] > 0)
    # c  = (ϕ.center[i, j] > 0) || (ϕ.center[i-1, j] > 0)
    # v  = (ϕ.vertex[i, j] > 0) || (ϕ.vertex[i, j+1] > 0)
    # cv = c || v
    # return cv || (ϕ.Vx[i, j] > 0)
end

function isvalid_vy(ϕ::RockRatio, i, j)
    c  = (ϕ.center[i, j] > 0) * (ϕ.center[i, j - 1] > 0)
    v  = (ϕ.vertex[i, j] > 0) * (ϕ.vertex[i + 1, j] > 0)
    cv = c * v
    return cv * (ϕ.Vy[i, j] > 0)
    # c  = (ϕ.center[i, j] > 0) || (ϕ.center[i, j - 1] > 0)
    # v  = (ϕ.vertex[i, j] > 0) || (ϕ.vertex[i + 1, j] > 0)
    # cv = c || v
    # return cv || (ϕ.Vy[i, j] > 0)
end

# function isvalid(A::T, I::Vararg{Integer, N}) where {N, T<:AbstractArray}
#     v = true
#     Base.@nexprs 2 j -> begin
#         Base.@nexprs 2 i -> begin
#             @inline
#             ii = clamp(I[1] + 2 * i - 3, 1, size(A, 1))
#             jj = clamp(I[2] + 2 * j - 3, 1, size(A, 2))
#             v *= if N == 3
#                 A[ii, jj, I[3]] > 0
#             else
#                 A[ii, jj] > 0
#             end
#         end
#     end
#     if N === 3
#         Base.@nexprs 2 k -> begin    
#             kk = clamp(I[3] + 2 * k - 3, 1, size(A, 3))
#             v *= A[ii, jj, kk] > 0
#         end
#     end
#     return v * (A[I...] > 0)
# end

@parallel_indices (I...) function compute_∇V!(
    ∇V::AbstractArray{T, N}, V::NTuple{N}, ϕ::RockRatio, _di::NTuple{N}
) where {T, N}
    @inline d_xi(A) = _d_xi(A, _di[1], I...)
    @inline d_yi(A) = _d_yi(A, _di[2], I...)
    @inline d_zi(A) = _d_zi(A, _di[3], I...)

    f = d_xi, d_yi, d_zi

    if isvalid_c(ϕ, I...)
        @inbounds ∇V[I...] = sum(f[i](V[i]) for i in 1:N)
    else 
        @inbounds ∇V[I...] = zero(T)
    end
    return nothing
end

@parallel_indices (I...) function compute_P!(
    P, P0, RP, ∇V, η, rheology::NTuple{N,MaterialParams}, phase_ratio, ϕ::RockRatio, dt, r, θ_dτ
) where {N}
    if isvalid_c(ϕ, I...)
        K = JustRelax2D.fn_ratio(JustRelax2D.get_bulk_modulus, rheology, @cell(phase_ratio[I...]))
        RP[I...], P[I...] = JustRelax2D._compute_P!(P[I...], P0[I...], ∇V[I...], η[I...], K, dt, r, θ_dτ)
    else
        RP[I...] = P[I...] = zero(eltype(P))
    end
    return nothing
end
            
# 2D kernel
@parallel_indices (I...) function update_stresses_center_vertex_ps!(
    ε::NTuple{3, T},      # normal components @ centers; shear components @ vertices
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
    ϕ
) where T
    τxyv = τshear_v[1]
    τxyv_old = τshear_ov[1]
    ni = size(Pr)
    Ic = JustRelax2D.clamped_indices(ni, I...)

    if isvalid_v(ϕ, I...)
        # interpolate to ith vertex
        Pv_ij       = JustRelax2D.av_clamped(Pr, Ic...)
        εxxv_ij     = JustRelax2D.av_clamped(ε[1], Ic...)
        εyyv_ij     = JustRelax2D.av_clamped(ε[2], Ic...)
        τxxv_ij     = JustRelax2D.av_clamped(τ[1], Ic...)
        τyyv_ij     = JustRelax2D.av_clamped(τ[2], Ic...)
        τxxv_old_ij = JustRelax2D.av_clamped(τ_o[1], Ic...)
        τyyv_old_ij = JustRelax2D.av_clamped(τ_o[2], Ic...)
        EIIv_ij     = JustRelax2D.av_clamped(EII, Ic...)

        ## vertex
        phase = @inbounds phase_vertex[I...]
        is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = JustRelax2D.plastic_params_phase(rheology, EIIv_ij, phase)
        _Gvdt = inv(JustRelax2D.fn_ratio(JustRelax2D.get_shear_modulus, rheology, phase) * dt)
        Kv = JustRelax2D.fn_ratio(JustRelax2D.get_bulk_modulus, rheology, phase)
        volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        ηv_ij = JustRelax2D.av_clamped(η, Ic...)
        dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

        # stress increments @ vertex
        dτxxv =
            (-(τxxv_ij - τxxv_old_ij) * ηv_ij * _Gvdt - τxxv_ij + 2.0 * ηv_ij * εxxv_ij) * dτ_rv
        dτyyv =
            (-(τyyv_ij - τyyv_old_ij) * ηv_ij * _Gvdt - τyyv_ij + 2.0 * ηv_ij * εyyv_ij) * dτ_rv
        dτxyv =
            (
                -(τxyv[I...] - τxyv_old[I...]) * ηv_ij * _Gvdt - τxyv[I...] +
                2.0 * ηv_ij * ε[3][I...]
            ) * dτ_rv
        τIIv_ij = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)

        # yield function @ center
        Fv = τIIv_ij - Cv - Pv_ij * sinϕv
        if is_pl && !iszero(τIIv_ij)
            # stress correction @ vertex
            λv[I...] =
                (1.0 - relλ) * λv[I...] +
                relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
            dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij
            τxyv[I...] += dτxyv - 2.0 * ηv_ij * 0.5 * λv[I...] * dQdτxy * dτ_rv
        else
            # stress correction @ vertex
            τxyv[I...] += dτxyv
        end
    else
        τxyv[I...] = zero(eltype(T))
    end

    ## center
    if all(I .≤ ni)
        if isvalid_c(ϕ, I...)
            # Material properties
            phase = @inbounds phase_center[I...]
            _Gdt = inv(JustRelax2D.fn_ratio(JustRelax2D.get_shear_modulus, rheology, phase) * dt)
            is_pl, C, sinϕ, cosϕ, sinψ, η_reg = JustRelax2D.plastic_params_phase(rheology, EII[I...], phase)
            K = JustRelax2D.fn_ratio(JustRelax2D.get_bulk_modulus, rheology, phase)
            volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
            ηij = η[I...]
            dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

            # cache strain rates for center calculations
            τij, τij_o, εij = JustRelax2D.cache_tensors(τ, τ_o, ε, I...)

            # visco-elastic strain rates @ center
            εij_ve = @. εij + 0.5 * τij_o * _Gdt
            εII_ve = GeoParams.second_invariant(εij_ve)
            # stress increments @ center
            dτij = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij) * dτ_r
            τII_ij = GeoParams.second_invariant(dτij .+ τij)
            # yield function @ center
            F = τII_ij - C - Pr[I...] * sinϕ

            if is_pl && !iszero(τII_ij)
                # stress correction @ center
                λ[I...] =
                    (1.0 - relλ) * λ[I...] +
                    relλ .* (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
                dQdτij = @. 0.5 * (τij + dτij) / τII_ij
                εij_pl = λ[I...] .* dQdτij
                dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
                τij = dτij .+ τij
                setindex!.(τ, τij, I...)
                setindex!.(ε_pl, εij_pl, I...)
                τII[I...] = GeoParams.second_invariant(τij)
                Pr_c[I...] = Pr[I...] + K * dt * λ[I...] * sinψ
                η_vep[I...] = 0.5 * τII_ij / εII_ve
            else
                # stress correction @ center
                setindex!.(τ, dτij .+ τij, I...)
                η_vep[I...] = ηij
                τII[I...] = τII_ij
            end

            Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
        else
            Pr_c[I...] = zero(eltype(T))
            # τij, = JustRelax2D.cache_tensors(τ, τ_o, ε, I...)
            dτij = zero(eltype(T)), zero(eltype(T)), zero(eltype(T))
            # setindex!.(τ, dτij .+ τij, I...)
            setindex!.(τ, dτij, I...)
        end
    end

    return nothing
end


# # with free surface stabilization
@parallel_indices (i, j) function compute_V!(
    Vx::AbstractArray{T,2}, Vy, Rx, Ry, P, τxx, τyy, τxy, ηdτ, ρgx, ρgy, ητ, ϕ::RockRatio, _dx, _dy
) where {T}
    d_xi(A, ϕ) = _d_xi(A, ϕ, _dx, i, j)
    d_xa(A, ϕ) = _d_xa(A, ϕ, _dx, i, j)
    d_yi(A, ϕ) = _d_yi(A, ϕ, _dy, i, j)
    d_ya(A, ϕ) = _d_ya(A, ϕ, _dy, i, j)
    av_xa(A) = _av_xa(A, i, j)
    av_ya(A) = _av_ya(A, i, j)
    harm_xa(A) = _av_xa(A, i, j)
    harm_ya(A) = _av_ya(A, i, j)

    if all((i, j) .< size(Vx) .- 1)
        if isvalid_vx(ϕ, i + 1, j)
            Rx[i, j]= R_Vx    = (-d_xa(P, ϕ.center) + d_xa(τxx, ϕ.center) + d_yi(τxy, ϕ.vertex) - av_xa(ρgx)) 
            Vx[i + 1, j + 1] += R_Vx * ηdτ / av_xa(ητ)
        else
            Rx[i, j]         = zero(T)
            Vx[i + 1, j + 1] = zero(T)
        end
    end

    if all((i, j) .< size(Vy) .- 1)        
        if isvalid_vy(ϕ, i, j + 1)
            Ry[i, j] = R_Vy   = -d_ya(P, ϕ.center) + d_ya(τyy, ϕ.center) + d_xi(τxy, ϕ.vertex) - av_ya(ρgy)
            Vy[i + 1, j + 1] += R_Vy * ηdτ / av_ya(ητ)
        else
            Ry[i, j]         = zero(T)
            Vy[i + 1, j + 1] = zero(T)
        end
    end

    return nothing
end

@parallel_indices (i, j) function compute_V!(
    Vx::AbstractArray{T,2}, Vy, Rx, Ry, P, τxx, τyy, τxy, ηdτ, ρgx, ρgy, ητ, ϕ_Vx, ϕ_Vy, _dx, _dy
) where {T}
    d_xi(A, ϕ) = _d_xi(A, ϕ, _dx, i, j)
    d_xa(A, ϕ) = _d_xa(A, ϕ, _dx, i, j)
    d_yi(A, ϕ) = _d_yi(A, ϕ, _dy, i, j)
    d_ya(A, ϕ) = _d_ya(A, ϕ, _dy, i, j)
    av_xa(A) = _av_xa(A, i, j)
    av_ya(A) = _av_ya(A, i, j)
    harm_xa(A) = _av_xa(A, i, j)
    harm_ya(A) = _av_ya(A, i, j)

    if all((i, j) .< size(Vx) .- 1)
        if iszero(ϕ_Vx[i + 1, j])
            Rx[i, j]         = zero(T)
            Vx[i + 1, j + 1] = zero(T)
        else
            Rx[i, j]= R_Vx    = (-d_xa(P, ϕ.center) + d_xa(τxx, ϕ.center) + d_yi(τxy, ϕ.vertex) - av_xa(ρgx)) 
            Vx[i + 1, j + 1] += R_Vx * ηdτ / av_xa(ητ)
        end
    end

    if all((i, j) .< size(Vy) .- 1)        
        if iszero(ϕ_Vy[i, j + 1])
            Ry[i, j]         = zero(T)
            Vy[i + 1, j + 1] = zero(T)
        else
            Ry[i, j] = R_Vy   = -d_ya(P, ϕ.center) + d_ya(τyy, ϕ.center) + d_xi(τxy, ϕ.vertex) - av_ya(ρgy)
            Vy[i + 1, j + 1] += R_Vy * ηdτ / av_ya(ητ)
        end
    end

    return nothing
end

# @parallel_indices (i, j) function compute_V!(
#     Vx::AbstractArray{T,2}, Vy, Vx_on_Vy, P, τxx, τyy, τxy, ηdτ, ρgx, ρgy, ητ, ϕ_Vx, ϕ_Vy, _dx, _dy, dt
# ) where {T}
#     d_xi(A, ϕ) = _d_xi(A, ϕ, _dx, i, j)
#     d_yi(A, ϕ) = _d_yi(A, ϕ, _dy, i, j)
#     d_xa(A, ϕ) = _d_xa(A, ϕ, _dx, i, j)
#     d_ya(A, ϕ) = _d_ya(A, ϕ, _dy, i, j)
#     av_xa(A) = _av_xa(A, i, j)
#     av_ya(A) = _av_ya(A, i, j)
#     harm_xa(A) = _av_xa(A, i, j)
#     harm_ya(A) = _av_ya(A, i, j)

#     nx, ny = size(ρgy)

#     if all((i, j) .< size(Vx) .- 1)
#         dVx = if iszero(ϕ_Vx[i + 1, j])
#             zero(T)
#         else
#             (-d_xa(P, ϕ.center) + d_xa(τxx, ϕ.center) + d_yi(τxy, ϕ.vertex) - av_xa(ρgx)) * ηdτ / av_xa(ητ)
#         end
#         Vx[i + 1, j + 1] += dVx
#     end

#     if all((i, j) .< size(Vy) .- 1)
#         dVy = if iszero(ϕ_Vy[i, j + 1])
#             zero(T)
#         else
#             ρg_correction = if iszero(dt)
#                 zero(dt)
#             else
#                 θ = 1
#                 # Interpolated Vx into Vy node (includes density gradient)
#                 Vxᵢⱼ = Vx_on_Vy[i + 1, j + 1]
#                 # Vertical velocity
#                 Vyᵢⱼ = Vy[i + 1, j + 1]
#                 # Get necessary buoyancy forces
#                 j_N = min(j + 1, ny)
#                 ρg_S = ρgy[i, j]
#                 ρg_N = ρgy[i, j_N]
#                 # Spatial derivatives
#                 ∂ρg∂y = (ρg_N - ρg_S) * _dy            
#                 # correction term
#                 (Vxᵢⱼ + Vyᵢⱼ * ∂ρg∂y) * θ * dt
#             end
#             (-d_ya(P, ϕ.center) + d_ya(τyy, ϕ.center) + d_xi(τxy, ϕ.vertex) - av_ya(ρgy) + ρg_correction) * ηdτ / av_ya(ητ)
#         end
#         Vy[i + 1, j + 1] += dVy
#     end

#     return nothing
# end

@parallel_indices (i, j) function compute_Res!(
    Rx::AbstractArray{T,2}, Ry, Vx, Vy, Vx_on_Vy, P, τxx, τyy, τxy, ρgx, ρgy, ϕ_Vx, ϕ_Vy, _dx, _dy, dt
) where {T}
    d_xi(A, ϕ) = _d_xi(A, ϕ, _dx, i, j)
    d_yi(A, ϕ) = _d_yi(A, ϕ, _dy, i, j)
    d_xa(A, ϕ) = _d_xa(A, ϕ, _dx, i, j)
    d_ya(A, ϕ) = _d_ya(A, ϕ, _dy, i, j)
    @inline av_xa(A) = _av_xa(A, i, j)
    @inline av_ya(A) = _av_ya(A, i, j)

    nx, ny = size(ρgy)
    @inbounds begin
        if all((i, j) .≤ size(Rx))
            Rx[i, j] = if iszero(ϕ_Vx[i + 1, j])
                zero(T)
            else
                -d_xa(P, ϕ.center) + d_xa(τxx, ϕ.center) + d_yi(τxy, ϕ.vertex) - av_xa(ρgx)
            end
        end
        
        if all((i, j) .≤ size(Ry))
            Ry[i, j] = if iszero(ϕ_Vy[i, j + 1])
                zero(T)
            else
                θ = 1.0
                # Interpolated Vx into Vy node (includes density gradient)
                Vxᵢⱼ = Vx_on_Vy[i + 1, j + 1]
                # Vertical velocity
                Vyᵢⱼ = Vy[i + 1, j + 1]
                # Get necessary buoyancy forces
                j_N = min(j + 1, ny)
                ρg_S = ρgy[i, j]
                ρg_N = ρgy[i, j_N]
                # Spatial derivatives
                ∂ρg∂y = (ρg_N - ρg_S) * _dy
                # correction term
                ρg_correction = (Vxᵢⱼ + Vyᵢⱼ * ∂ρg∂y) * θ * dt
                -d_ya(P, ϕ.center) + d_ya(τyy, ϕ.center) + d_xi(τxy, ϕ.vertex) - av_ya(ρgy) + ρg_correction
            end
        end
    end

    return nothing
end

# #### TESTING GROUNDS
# N = 4
# ni = n, n = (N,N)
# ϕ = RockRatio(n, n)
# @test RockRatio(n, n) isa RockRatio

# @test RockRatio(ni)   isa RockRatio
# @test size_c(ϕ)  === ni
# @test size_v(ϕ)  === ni.+1
# @test size_vx(ϕ) === (N+1, N)
# @test size_vy(ϕ) === (N, N+1)

# A = [
#     0 1 1
#     1 0 1
# ]

# @test isvalid(A, 1, 1) || isvalid(A, 2, 2)  === false
# @test isvalid(A, 1, 2) && isvalid(A, 2, 1)  === true
# @test isvalid_x(A, 1, 1) === false
# @test isvalid_x(A, 1, 3) === true
# @test isvalid_y(A, 1, 1) === false
# @test isvalid_y(A, 1, 2) === true
