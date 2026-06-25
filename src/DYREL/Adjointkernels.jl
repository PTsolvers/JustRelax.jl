using Enzyme
using ParallelStencil
using JustRelax
using JustRelax.JustRelax2D
import JustRelax.JustRelax2D:
    compute_∇V_strain_rate!,
    compute_PH_residual_V!,
    compute_DR_residual_V!,
    compute_residual_P!,
    compute_stress_DRYEL!,
    update_V_damping_DR_V!,
    compute_dV!,
    update_α_β!,
    update_dτV_α_β!,
    Gershgorin_Stokes2D_SchurComplement!

# Enzyme wrappers for JustRelax.jl DYREL 2D solver kernels.

function launch_compute_strain_rate_DYREL2D!(
        ∇V, εxx, εyy, εxy, Vx, Vy, _di_vertex, _di_vx, _di_vy,
    )
    ni = size(∇V)
    @parallel (@idx ni .+ 1) compute_∇V_strain_rate!(
        ∇V, εxx, εyy, εxy, Vx, Vy, _di_vertex, _di_vx, _di_vy,
    )
    return nothing
end

function diff_compute_strain_rate_DYREL2D!(
        ∇V, εxx, εyy, εxy, Vx, Vy, _di_vertex, _di_vx, _di_vy,
        d∇V, dεxx, dεyy, dεxy, dVx, dVy,
    )
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        Enzyme.Const(launch_compute_strain_rate_DYREL2D!),
        Enzyme.Const,
        Enzyme.DuplicatedNoNeed(∇V, d∇V),
        Enzyme.DuplicatedNoNeed(εxx, dεxx),
        Enzyme.DuplicatedNoNeed(εyy, dεyy),
        Enzyme.DuplicatedNoNeed(εxy, dεxy),
        Enzyme.DuplicatedNoNeed(Vx, dVx),
        Enzyme.DuplicatedNoNeed(Vy, dVy),
        Enzyme.Const(_di_vertex),
        Enzyme.Const(_di_vx),
        Enzyme.Const(_di_vy),
    )
    return nothing
end

function launch_compute_PH_residual_V_DYREL2D!(
        Rx, Ry, P, ΔPψ, τxx, τyy, τxy, ρgx, ρgy, _di_center, _di_vertex,
    )
    ni = size(P)
    @parallel (@idx ni) compute_PH_residual_V!(
        Rx, Ry, P, ΔPψ, τxx, τyy, τxy, ρgx, ρgy, _di_center, _di_vertex,
    )
    return nothing
end

function diff_compute_PH_residual_V_DYREL2D!(
        Rx, Ry, P, ΔPψ, τxx, τyy, τxy, ρgx, ρgy, _di_center, _di_vertex,
        dRx, dRy, dP, dΔPψ, dτxx, dτyy, dτxy, dρgx, dρgy,
    )
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        Enzyme.Const(launch_compute_PH_residual_V_DYREL2D!),
        Enzyme.Const,
        Enzyme.DuplicatedNoNeed(Rx, dRx),
        Enzyme.DuplicatedNoNeed(Ry, dRy),
        Enzyme.DuplicatedNoNeed(P, dP),
        Enzyme.DuplicatedNoNeed(ΔPψ, dΔPψ),
        Enzyme.DuplicatedNoNeed(τxx, dτxx),
        Enzyme.DuplicatedNoNeed(τyy, dτyy),
        Enzyme.DuplicatedNoNeed(τxy, dτxy),
        Enzyme.DuplicatedNoNeed(ρgx, dρgx),
        Enzyme.DuplicatedNoNeed(ρgy, dρgy),
        Enzyme.Const(_di_center),
        Enzyme.Const(_di_vertex),
    )
    return nothing
end

function launch_compute_DR_residual_V_DYREL2D!(
        Rx, Ry, P, P_num, ΔPψ, τxx, τyy, τxy, ρgx, ρgy, Dx, Dy,
        _di_center, _di_vertex,
    )
    ni = size(P)
    @parallel (@idx ni) compute_DR_residual_V!(
        Rx, Ry, P, P_num, ΔPψ, τxx, τyy, τxy, ρgx, ρgy, Dx, Dy,
        _di_center, _di_vertex,
    )
    return nothing
end

function diff_compute_DR_residual_V_DYREL2D!(
        Rx, Ry, P, P_num, ΔPψ, τxx, τyy, τxy, ρgx, ρgy, Dx, Dy,
        _di_center, _di_vertex,
        dRx, dRy, dP, dP_num, dΔPψ, dτxx, dτyy, dτxy, dρgx, dρgy,
    )
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        Enzyme.Const(launch_compute_DR_residual_V_DYREL2D!),
        Enzyme.Const,
        Enzyme.DuplicatedNoNeed(Rx, dRx),
        Enzyme.DuplicatedNoNeed(Ry, dRy),
        Enzyme.DuplicatedNoNeed(P, dP),
        Enzyme.DuplicatedNoNeed(P_num, dP_num),
        Enzyme.DuplicatedNoNeed(ΔPψ, dΔPψ),
        Enzyme.DuplicatedNoNeed(τxx, dτxx),
        Enzyme.DuplicatedNoNeed(τyy, dτyy),
        Enzyme.DuplicatedNoNeed(τxy, dτxy),
        Enzyme.DuplicatedNoNeed(ρgx, dρgx),
        Enzyme.DuplicatedNoNeed(ρgy, dρgy),
        Enzyme.Const(Dx),
        Enzyme.Const(Dy),
        Enzyme.Const(_di_center),
        Enzyme.Const(_di_vertex),
    )
    return nothing
end

function launch_compute_residual_P_DYREL2D!(
        RP, P, P0, ∇V, Q, ηb, rheology, phase_ratios, dt, args,
    )
    compute_residual_P!(
        RP, P, P0, ∇V, Q, ηb, rheology, phase_ratios, dt, args,
    )
    return nothing
end

function diff_compute_residual_P_DYREL2D!(
        RP, P, P0, ∇V, Q, ηb, rheology, phase_ratios, dt, args,
        dRP, dP, dP0, d∇V, dQ, dηb,
    )
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        Enzyme.Const(launch_compute_residual_P_DYREL2D!),
        Enzyme.Const,
        Enzyme.DuplicatedNoNeed(RP, dRP),
        Enzyme.DuplicatedNoNeed(P, dP),
        Enzyme.DuplicatedNoNeed(P0, dP0),
        Enzyme.DuplicatedNoNeed(∇V, d∇V),
        Enzyme.DuplicatedNoNeed(Q, dQ),
        Enzyme.DuplicatedNoNeed(ηb, dηb),
        Enzyme.Const(rheology),
        Enzyme.Const(phase_ratios),
        Enzyme.Const(dt),
        Enzyme.Const(args),
    )
    return nothing
end

function launch_update_pressure_DYREL2D!(P, RP, γ_eff)
    @. P += γ_eff * RP
    return nothing
end

function diff_update_pressure_DYREL2D!(P, RP, γ_eff, dP, dRP, dγ_eff)
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        Enzyme.Const(launch_update_pressure_DYREL2D!),
        Enzyme.Const,
        Enzyme.Duplicated(P, dP),
        Enzyme.DuplicatedNoNeed(RP, dRP),
        Enzyme.DuplicatedNoNeed(γ_eff, dγ_eff),
    )
    return nothing
end

function launch_compute_stress_DRYEL_DYREL2D!(
        stokes, rheology, phase_ratios, λ_relaxation, dt,
    )
    compute_stress_DRYEL!(
        stokes, rheology, phase_ratios, λ_relaxation, dt,
    )
    return nothing
end

function diff_compute_stress_DRYEL_DYREL2D!(
        stokes, rheology, phase_ratios, λ_relaxation, dt, dstokes,
    )
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        Enzyme.Const(launch_compute_stress_DRYEL_DYREL2D!),
        Enzyme.Const,
        Enzyme.Duplicated(stokes, dstokes),
        Enzyme.Const(rheology),
        Enzyme.Const(phase_ratios),
        Enzyme.Const(λ_relaxation),
        Enzyme.Const(dt),
    )
    return nothing
end

function launch_update_V_damping_DR_V_DYREL2D!(
        Vx, Vy, dVxdτ, dVydτ, Rx, Ry, αVx, αVy, βVx, βVy, dτVx, dτVy,
    )
    ni = size(Rx) .+ (1, 0)
    @parallel (@idx ni) update_V_damping_DR_V!(
        (Vx, Vy),
        (dVxdτ, dVydτ),
        (Rx, Ry),
        (αVx, αVy),
        (βVx, βVy),
        (dτVx, dτVy),
    )
    return nothing
end

function diff_update_V_damping_DR_V_DYREL2D!(
        Vx, Vy, dVxdτ, dVydτ, Rx, Ry, αVx, αVy, βVx, βVy, dτVx, dτVy,
        dVx, dVy, ddVxdτ, ddVydτ, dRx, dRy,
    )
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        Enzyme.Const(launch_update_V_damping_DR_V_DYREL2D!),
        Enzyme.Const,
        Enzyme.Duplicated(Vx, dVx),
        Enzyme.Duplicated(Vy, dVy),
        Enzyme.Duplicated(dVxdτ, ddVxdτ),
        Enzyme.Duplicated(dVydτ, ddVydτ),
        Enzyme.DuplicatedNoNeed(Rx, dRx),
        Enzyme.DuplicatedNoNeed(Ry, dRy),
        Enzyme.Const(αVx),
        Enzyme.Const(αVy),
        Enzyme.Const(βVx),
        Enzyme.Const(βVy),
        Enzyme.Const(dτVx),
        Enzyme.Const(dτVy),
    )
    return nothing
end

function launch_compute_dV_DYREL2D!(dVx, dVy, dVxdτ, dVydτ, βVx, βVy, dτVx, dτVy)
    ni = size(dVx) .+ (1, 0)
    @parallel (@idx ni) compute_dV!(
        (dVx, dVy),
        (dVxdτ, dVydτ),
        (βVx, βVy),
        (dτVx, dτVy),
    )
    return nothing
end

function diff_compute_dV_DYREL2D!(
        dVx, dVy, dVxdτ, dVydτ, βVx, βVy, dτVx, dτVy,
        ddVx, ddVy, ddVxdτ, ddVydτ,
    )
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        Enzyme.Const(launch_compute_dV_DYREL2D!),
        Enzyme.Const,
        Enzyme.DuplicatedNoNeed(dVx, ddVx),
        Enzyme.DuplicatedNoNeed(dVy, ddVy),
        Enzyme.DuplicatedNoNeed(dVxdτ, ddVxdτ),
        Enzyme.DuplicatedNoNeed(dVydτ, ddVydτ),
        Enzyme.Const(βVx),
        Enzyme.Const(βVy),
        Enzyme.Const(dτVx),
        Enzyme.Const(dτVy),
    )
    return nothing
end

function launch_update_α_β_DYREL2D!(βVx, βVy, αVx, αVy, dτVx, dτVy, cVx, cVy)
    update_α_β!(βVx, βVy, αVx, αVy, dτVx, dτVy, cVx, cVy)
    return nothing
end

function diff_update_α_β_DYREL2D!(
        βVx, βVy, αVx, αVy, dτVx, dτVy, cVx, cVy,
        dβVx, dβVy, dαVx, dαVy, ddτVx, ddτVy, dcVx, dcVy,
    )
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        Enzyme.Const(launch_update_α_β_DYREL2D!),
        Enzyme.Const,
        Enzyme.DuplicatedNoNeed(βVx, dβVx),
        Enzyme.DuplicatedNoNeed(βVy, dβVy),
        Enzyme.DuplicatedNoNeed(αVx, dαVx),
        Enzyme.DuplicatedNoNeed(αVy, dαVy),
        Enzyme.DuplicatedNoNeed(dτVx, ddτVx),
        Enzyme.DuplicatedNoNeed(dτVy, ddτVy),
        Enzyme.DuplicatedNoNeed(cVx, dcVx),
        Enzyme.DuplicatedNoNeed(cVy, dcVy),
    )
    return nothing
end

function launch_update_dτV_α_β_DYREL2D!(
        dτVx, dτVy, βVx, βVy, αVx, αVy, cVx, cVy, λmaxVx, λmaxVy, CFL,
    )
    update_dτV_α_β!(
        dτVx, dτVy, βVx, βVy, αVx, αVy, cVx, cVy, λmaxVx, λmaxVy, CFL,
    )
    return nothing
end

function diff_update_dτV_α_β_DYREL2D!(
        dτVx, dτVy, βVx, βVy, αVx, αVy, cVx, cVy, λmaxVx, λmaxVy, CFL,
        ddτVx, ddτVy, dβVx, dβVy, dαVx, dαVy, dcVx, dcVy, dλmaxVx, dλmaxVy,
    )
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        Enzyme.Const(launch_update_dτV_α_β_DYREL2D!),
        Enzyme.Const,
        Enzyme.DuplicatedNoNeed(dτVx, ddτVx),
        Enzyme.DuplicatedNoNeed(dτVy, ddτVy),
        Enzyme.DuplicatedNoNeed(βVx, dβVx),
        Enzyme.DuplicatedNoNeed(βVy, dβVy),
        Enzyme.DuplicatedNoNeed(αVx, dαVx),
        Enzyme.DuplicatedNoNeed(αVy, dαVy),
        Enzyme.DuplicatedNoNeed(cVx, dcVx),
        Enzyme.DuplicatedNoNeed(cVy, dcVy),
        Enzyme.DuplicatedNoNeed(λmaxVx, dλmaxVx),
        Enzyme.DuplicatedNoNeed(λmaxVy, dλmaxVy),
        Enzyme.Const(CFL),
    )
    return nothing
end

function launch_Gershgorin_Stokes2D_SchurComplement_DYREL2D!(
        Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, phase_ratios, rheology, di, dt,
    )
    Gershgorin_Stokes2D_SchurComplement!(
        Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, phase_ratios, rheology, di, dt,
    )
    return nothing
end

function diff_Gershgorin_Stokes2D_SchurComplement_DYREL2D!(
        Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, phase_ratios, rheology, di, dt,
        dDx, dDy, dλmaxVx, dλmaxVy, dη, dηv, dγ_eff,
    )
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        Enzyme.Const(launch_Gershgorin_Stokes2D_SchurComplement_DYREL2D!),
        Enzyme.Const,
        Enzyme.DuplicatedNoNeed(Dx, dDx),
        Enzyme.DuplicatedNoNeed(Dy, dDy),
        Enzyme.DuplicatedNoNeed(λmaxVx, dλmaxVx),
        Enzyme.DuplicatedNoNeed(λmaxVy, dλmaxVy),
        Enzyme.DuplicatedNoNeed(η, dη),
        Enzyme.DuplicatedNoNeed(ηv, dηv),
        Enzyme.DuplicatedNoNeed(γ_eff, dγ_eff),
        Enzyme.Const(phase_ratios),
        Enzyme.Const(rheology),
        Enzyme.Const(di),
        Enzyme.Const(dt),
    )
    return nothing
end
