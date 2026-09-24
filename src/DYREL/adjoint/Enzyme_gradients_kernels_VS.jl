## MATERIAL-PARAMETER SENSITIVITY KERNELS FOR THE VARIATIONAL (RockRatio ϕ) DYREL SOLVER
#
# Counterparts of the kernels in Enzyme_gradients_kernels.jl that mirror the masked forward
# kernels of `_solve_VariationalDYREL!`. 2D only, like the variational DYREL solver.

"""
    stress_sensitivity_kernel!(
        centers, vertices, parameters, material, p, phase_center, phase_vertex,
        τ_o, τ_ov, ε, EII_pl, P, λ, λv, η, τ_seed, τv_seed, θ_seed, λ_relaxation, dt,
        periodic, ϕ::RockRatio,
    )

Material-parameter sensitivities of the masked local stress update of phase `p`, evaluated at
the same points and with the same inputs as the variational `compute_stress_DRYEL!`:

  - points the forward kernel masks out (`!isvalid_v` / `!isvalid_c`) are skipped;
  - the vertex viscosity is `harm_clamped(η)` of the surrounding centers, as there is no
    separate vertex viscosity in the variational stress kernel.

Only material parameters are stored here. The viscosity sensitivity is not: at a vertex it is
the derivative with respect to a harmonic mean of four centers, and scattering it back would
race between threads. The caller takes it from the reverse variational stress kernel instead.
"""
@parallel_indices (I...) function stress_sensitivity_kernel!(
        centers, vertices, parameters,
        material, p, phase_center, phase_vertex,
        τ_o, τ_ov, ε, EII_pl, P, λ, λv, η,
        τ_seed, τv_seed, θ_seed, λ_relaxation, dt, periodic, ϕ::JustRelax.RockRatio,
    )
    Base.@propagate_inbounds @inline av(A) = sum(JustRelax2D._gather(A, I...)) / 4
    ni = size(phase_center)
    @inbounds begin
        Ic = clamped_indices(ni, periodic, I...)
        if isvalid_v(ϕ, I...)
            ratio = phase_vertex[I...][p]
            derivative, _ = enzyme_stress_gradients(
                material,
                (av_clamped(ε[1], Ic...), av_clamped(ε[2], Ic...), ε[3][I...]),
                (τ_ov[1][I...], τ_ov[2][I...], τ_ov[3][I...]),
                harm_clamped(η, Ic...), av_clamped(P, Ic...), λv[I...], λ_relaxation, dt,
                av_clamped(EII_pl, Ic...),
                (τv_seed[1][I...], τv_seed[2][I...], τv_seed[3][I...]),
                0.0, ratio,
            )
            store_parameter_gradients!(vertices, parameters, material, derivative, p, I)
        end

        if all(I .≤ ni) && isvalid_c(ϕ, I...)
            ratio = phase_center[I...][p]
            derivative, _ = enzyme_stress_gradients(
                material,
                (ε[1][I...], ε[2][I...], av(ε[3])),
                (τ_o[1][I...], τ_o[2][I...], τ_o[3][I...]),
                η[I...], P[I...], λ[I...], λ_relaxation, dt, EII_pl[I...],
                (τ_seed[1][I...], τ_seed[2][I...], τ_seed[3][I...]),
                θ_seed[I...], ratio,
            )
            store_parameter_gradients!(centers, parameters, material, derivative, p, I)
        end
    end
    return nothing
end
