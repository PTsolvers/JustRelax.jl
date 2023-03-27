## Helperfunctions for MTK vs JustRelax

using Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions, Parameters

# to be added to GP# Viscosity with partial melting -----------------------------------------
"""
MeltViscous(η_s = 1e22 * Pa*s,η_f = 1e16 * Pa*s,ϕ = 0.0 * NoUnits,S = 1.0 * NoUnits,mfac = -2.8 * NoUnits)
Defines a effective viscosity of partially molten rock as: 
```math  
\\eta  = \\min(\\eta_f (1-S(1-\\phi))^m_{fac})
```
"""
@with_kw_noshow struct MeltViscous{T,U1,U2} <: AbstractCreepLaw{T}
η_s::GeoUnit{T,U1} = 1e22 * Pa*s # rock's viscosity
η_f::GeoUnit{T,U1} = 1e16 * Pa*s # magma's viscosity 
S::GeoUnit{T,U2} = 1.0 * NoUnits # factors for hexagons
mfac::GeoUnit{T,U2} = -2.8 * NoUnits # factors for hexagons
end
MeltViscous(a...) = MeltViscous(convert.(GeoUnit, a)...)


# unpacks fields of the struct x into a tuple
@generated function unpack(x::T) where {T}
    return quote
        Base.@_inline_meta
        tuple(_unpack(x, fieldnames($T))...)
    end
end
_unpack(a, fields) = (getfield(a, fi) for fi in fields)

macro unpack(x)
    return quote
        unpack($(esc(x)))
    end
end


@parallel_indices (i, j, k) function update_buoyancy!(fz, T, ρ0gα)
    
    fz[i, j, k] = ρ0gα * 0.125 * (
        T[i, j, k  ] + T[i+1, j, k  ] + T[i, j+1, k  ] + T[i+1, j+1, k  ] +
        T[i, j, k+1] + T[i+1, j, k+1] + T[i, j+1, k+1] + T[i+1, j+1, k+1]
    )
    
    return nothing
end

@parallel_indices (i, j) function initViscosity!(η, Phases, MatParam)
    @inbounds η[i, j] = MatParam[Phases[i, j]].CompositeRheology[1][1].η.val
    return nothing
end

@parallel function computeViscosity!(η, ϕ, S, mfac, η_f, η_s)
    # We assume that η is f(ϕ), following Deubelbeiss, Kaus, Connolly (2010) EPSL 
    # factors for hexagons
    @all(η) = min(
        η_f * (1.0 - S * (1.0 - @all(ϕ)))^mfac,
        η_s, # upper cutoff
    )
    return nothing
end

@parallel function update_viscosity(η, ϕ, S, mfac, η_f, η_s)                   #update Viscosity
    # We assume that η is f(ϕ), following Deubelbeiss, Kaus, Connolly (2010) EPSL 
    # factors for hexagons
    @all(η) = min(
        η_f * (1.0 - S * (1.0 - @all(ϕ)))^mfac,
        η_s, # upper cutoff
    )
    return nothing
end


@parallel_indices (i, j) function compute_melt_fraction!(ϕ, rheology, phase_c, args)
    ϕ[i, j] = compute_meltfraction(rheology, phase_c[i, j], ntuple_idx(args, i, j))
    return nothing
end


@parallel_indices (i, j) function compute_ρg!(ρg, rheology, phase_c, args)
    ρg[i, j] =
        compute_density(rheology, phase_c[i, j], ntuple_idx(args, i, j)) *
        compute_gravity(rheology, phase_c[i, j])
    return nothing
end

@parallel_indices (i, j, k) function _elliptical_perturbation!(T, δT, xc, yc, zc, r, x, y, z)
    if (((x[i]-xc ))^2 + ((y[j] - yc))^2 + ((z[k] - zc))^2) ≤ r^2
        T[i,j,k] *= δT/100 + 1
    end
    return nothing
end

function elliptical_perturbation!(T, δT, xc, yc, zc, r, xvi)

    @parallel_indices (i, j, k) function _elliptical_perturbation!(T, δT, xc, yc, zc, r, x, y, z)
        @inbounds if (((x[i]-xc ))^2 + ((y[j] - yc))^2 + ((z[k] - zc))^2) ≤ r^2
            T[i,j,k] *= δT/100 + 1
        end
        return nothing
    end

    @parallel _elliptical_perturbation!(T, δT, xc, yc, zc, r, xvi...)

end

# macro idx(args...)
#   return quote
#       _idx(tuple($(esc.(args)...))...)
#   end
# end


# @inline Base.@pure _idx(args::Vararg{Int, N}) where N = ntuple(i->1:args[i], Val(N)) 
# @inline Base.@pure _idx(args::NTuple{N, Int}) where N = ntuple(i->1:args[i], Val(N)) 

@parallel function compute_maxRatio!(Musτ2::AbstractArray, Musτ::AbstractArray)
    @inn(Musτ2) = @maxloc(Musτ) / @minloc(Musτ)
    return nothing
end

@parallel function compute_qT!(
    qTx::AbstractArray,
    qTy::AbstractArray,
    T::AbstractArray,
    κ::Number,
    dx::Number,
    dy::Number,
)
    @all(qTx) = -κ * @d_xi(T) / dx
    @all(qTy) = -κ * @d_yi(T) / dy
    return nothing
end

@parallel_indices (ix, iy) function advect_T!(
    dT_dt::AbstractArray,
    qTx::AbstractArray,
    qTy::AbstractArray,
    T::AbstractArray,
    Vx::AbstractArray,
    Vy::AbstractArray,
    dx::Number,
    dy::Number,
)
    if (ix <= size(dT_dt, 1) && iy <= size(dT_dt, 2))
        dT_dt[ix, iy] =
            -((qTx[ix + 1, iy] - qTx[ix, iy]) / dx + (qTy[ix, iy + 1] - qTy[ix, iy]) / dy) -
            (Vx[ix + 1, iy + 1] > 0) *
            Vx[ix + 1, iy + 1] *
            (T[ix + 1, iy + 1] - T[ix, iy + 1]) / dx -
            (Vx[ix + 2, iy + 1] < 0) *
            Vx[ix + 2, iy + 1] *
            (T[ix + 2, iy + 1] - T[ix + 1, iy + 1]) / dx -
            (Vy[ix + 1, iy + 1] > 0) *
            Vy[ix + 1, iy + 1] *
            (T[ix + 1, iy + 1] - T[ix + 1, iy]) / dy -
            (Vy[ix + 1, iy + 2] < 0) *
            Vy[ix + 1, iy + 2] *
            (T[ix + 1, iy + 2] - T[ix + 1, iy + 1]) / dy
    end
    return nothing
end

@parallel function update_T!(
    T::AbstractArray, T_old::AbstractArray, dT_dt::AbstractArray, Δt::Number
)
    @inn(T) = @inn(T_old) + @all(dT_dt) * Δt
    return nothing
end

@parallel_indices (ix, iy) function no_fluxY_T!(T::AbstractArray)
    if (ix == size(T, 1) && iy <= size(T, 2))
        T[ix, iy] = T[ix - 1, iy]
    end
    if (ix == 1 && iy <= size(T, 2))
        T[ix, iy] = T[ix + 1, iy]
    end
    return nothing
end

# # MTK coupled old Stokes solver of BK
# # Stokes solver (taken from PseudoTransientStokes.jl)
# function ComputeStokesTimestep_continuation!(A, Num, Grid, MatParam, Phases, Max_Eta_factor)

#     # In order to make the code more robust for arbitrary viscosity contrasts, 
#     # we start the calculation with a smooth viscosity structure 
#     η_original = zeros(size(A.Mus))
#     η_smooth = zeros(size(A.Mus))
#     η = zeros(size(A.Mus))
#     η_original .= A.Mus

#     @parallel compute_maxRatio!(A.MuRatio, η_original)
#     ViscosityContrast = maximum(A.MuRatio)

#     while ViscosityContrast > Max_Eta_factor
#         @parallel smooth!(A.Mus2, A.Mus, 1.0)
#         @parallel (1:Grid.N[1]) bc_y!(A.Mus2)
#         @parallel (1:Grid.N[2]) bc_x!(A.Mus2)
#         A.Mus .= A.Mus2
#         A.Mus2 .= A.Mus

#         @parallel compute_maxRatio!(A.MuRatio, A.Mus)  # local viscosity ratio
#         ViscosityContrast = maximum(A.MuRatio)

#         #ViscosityContrast = maximum(A.Mus)/minimum(A.Mus)
#     end
#     η_smooth .= A.Mus

#     smooth_factor = 1.0

#     η .= smooth_factor * η_smooth + (1.0 - smooth_factor) * η_original
#     @parallel compute_maxloc!(A.Musτ, η)
#     @parallel (1:Grid.N[1]) bc_y!(A.Musτ)
#     @parallel (1:Grid.N[2]) bc_x!(A.Musτ)

#     @show maximum(η_smooth) / minimum(η_smooth),
#     maximum(η_original) / minimum(η_original), maximum(η) / minimum(η),
#     maximum(A.MuRatio)

#     dx, dy = Grid.Δ[1], Grid.Δ[2]
#     @parallel compute_iter_params!(
#         A.dτ_Rho, A.Gdτ, A.Musτ, Num.Vpdτ, Num.Re, Num.r, Num.max_lxy
#     )

#     # Update density 
#     @parallel (1:Grid.N[1], 1:Grid.N[2]) compute_ρg!(A.ρg, MatParam, Phases, (T=A.T, P=A.P))

#     # For viscous rheologies, reset stresses
#     A.τxx .= 0.0
#     A.τyy .= 0.0
#     A.σxy .= 0.0

#     err = 2 * Num.ε
#     iter = 0
#     err_evo1 = []
#     err_evo2 = []
#     norm_Rx, norm_Ry, norm_∇V = 0.0, 0.0, 0.0
#     while err > Num.ε && iter <= Num.iterMax || smooth_factor > 1e-5
#         @parallel compute_P!(A.∇V, A.Pt, A.Vx, A.Vy, A.Gdτ, Num.r, dx, dy)

#         #@parallel compute_ε!(A.εxx, A.εyy, A.εxy, A.∇V, A.Vx, A.Vy, dx, dy)
#         @parallel compute_τ!(A.τxx, A.τyy, A.σxy, A.Vx, A.Vy, η, A.Gdτ, dx, dy)
#         @parallel compute_dV!(
#             A.Rx, A.Ry, A.dVx, A.dVy, A.Pt, A.τxx, A.τyy, A.σxy, A.dτ_Rho, A.ρg, dx, dy
#         )

#         @parallel compute_V!(A.Vx, A.Vy, A.dVx, A.dVy)
#         @parallel (1:Grid.N[1]) bc_y!(A.Vx)
#         @parallel (1:Grid.N[2]) bc_x!(A.Vy)

#         iter += 1
#         if iter % Num.nout_iter == 0
#             Vmin, Vmax = minimum(A.Vx), maximum(A.Vx)
#             Pmin, Pmax = minimum(A.Pt), maximum(A.Pt)

#             ΔP = Pmax - Pmin
#             if abs(ΔP) > 0.0
#                 norm_Rx = norm(A.Rx) / ΔP * Num.lx / sqrt(length(A.Rx))
#                 norm_Ry = norm(A.Ry) / ΔP * Num.lx / sqrt(length(A.Ry))
#             else
#                 norm_Rx = norm(A.Rx) / length(A.Rx)
#                 norm_Ry = norm(A.Ry) / length(A.Ry)
#             end

#             ΔV = Vmax - Vmin
#             if abs(ΔV) > 0.0
#                 norm_∇V = norm(A.∇V) / ΔV * Num.lx / sqrt(length(A.∇V))
#             else
#                 norm_∇V = norm(A.∇V) / length(A.∇V)
#             end

#             # norm_Rx = norm(Rx)/length(Rx); norm_Ry = norm(Ry)/length(Ry); norm_∇V = norm(∇V)/length(∇V)
#             err = maximum([norm_Rx, norm_Ry, norm_∇V])
#             @printf(
#                 "  Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] smooth_factor = %e \n",
#                 iter,
#                 err,
#                 norm_Rx,
#                 norm_Ry,
#                 norm_∇V,
#                 smooth_factor
#             )

#             # Adjust the viscosity contrast if the error is sufficiently reduced
#             if err < 1e-2 && smooth_factor > 1e-5
#                 smooth_factor = smooth_factor * 0.1

#                 #η .= smooth_factor*η_smooth + (1.0 - smooth_factor)*η_original;
#                 η .=
#                     exp.(
#                         smooth_factor * log.(η_smooth) +
#                         (1.0 - smooth_factor) * log.(η_original)
#                     )

#                 @show maximum(η) / minimum(η)
#                 A.Mus .= η                     # store the actually used viscosity field
#             end

#             if isnan(err)
#                 break       # stop in case of NaN
#             end
#         end
#     end

#     return iter, norm_Rx, norm_Ry, norm_∇V
# end

# #-----------------------------------------------------------------------------------

