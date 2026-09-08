"""
    compute_sensitivities!(
        stokes, stokes_ad, ρg, phase_ratios, rheology, _di, ni, λ_relaxation, dt, igg
    )

Accumulate the sensitivities of the objective functional with respect to the material
parameters, given a converged adjoint state in `stokes_ad.λV` and `stokes_ad.λP`.

Fills

  - `stokes_ad.ρ` -- sensitivity with respect to density,
  - `stokes_ad.viscosity.η` and `stokes_ad.viscosity.ηv` -- sensitivity with respect to
    viscosity at the centers and vertices.

The adjoint working arrays (`P`, `θ`, `R.RP`, the stress tensor and the viscosity fields)
are zeroed on entry, so this has to be called *after* the adjoint iterations have converged
and not in between them.
"""
function compute_sensitivities!(
        stokes,
        stokes_ad,
        ρg,
        phase_ratios,
        rheology,
        _di,
        ni,
        λ_relaxation,
        dt,
        igg,
    )
    igg.me == 0 && @printf("\n######## Calculate Sensitivities ########\n")

    stokes_ad.P .= 0.0
    stokes_ad.θ .= 0.0
    stokes_ad.R.RP .= 0.0
    stokes_ad.τ.xx .= 0.0
    stokes_ad.τ.yy .= 0.0
    stokes_ad.τ.xy_c .= 0.0
    stokes_ad.τ.xx_v .= 0.0
    stokes_ad.τ.yy_v .= 0.0
    stokes_ad.τ.xy .= 0.0
    stokes_ad.τ.II .= 0.0
    stokes_ad.viscosity.η .= 0.0
    stokes_ad.viscosity.ηv .= 0.0
    stokes_ad.ρ .= 0.0

    dρgx = @zeros(ni...)
    @views stokes_ad.R.Rx .= -stokes_ad.λV.Vx[2:(end - 1), 2:(end - 1)]
    @views stokes_ad.R.Ry .= -stokes_ad.λV.Vy[2:(end - 1), 2:(end - 1)]
    enzyme_compute_PH_residual_V_sensitivity!(
        stokes, stokes_ad, ρg, (dρgx, stokes_ad.ρ), _di, ni
    )
    enzyme_compute_stress_DRYEL_sensitivity!(
        stokes, stokes_ad, rheology, phase_ratios, λ_relaxation, dt
    )

    gravity = compute_gravity(first(rheology))
    gx, gy = gravity isa Number ? (zero(gravity), gravity) : (gravity[1], gravity[3])
    # The residual depends on density through the buoyancy forces ρgx = ρ*gx and
    # ρgy = ρ*gy. The chain rule therefore gives dJ/dρ = gx*dJ/dρgx + gy*dJ/dρgy.
    @. stokes_ad.ρ = gx * dρgx + gy * stokes_ad.ρ

    return stokes_ad
end
