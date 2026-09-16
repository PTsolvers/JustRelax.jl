function combine_center_vertex_gradient!(center, vertex)
    nx, ny = size(center)
    size(vertex) == (nx + 1, ny + 1) ||
        throw(DimensionMismatch("vertex gradient must be one point larger than the center gradient"))

    # Pull back center2vertex!: each vertex value is the average of four centers.
    # Boundary vertices copy the nearest interior vertex, hence the clamped indices.
    for jv in axes(vertex, 2), iv in axes(vertex, 1)
        i = clamp(iv, 2, nx)
        j = clamp(jv, 2, ny)
        contribution = vertex[iv, jv] / 4
        center[i - 1, j - 1] += contribution
        center[i, j - 1] += contribution
        center[i - 1, j] += contribution
        center[i, j] += contribution
    end
    return center
end

"""
    compute_sensitivities!(
        stokes, stokes_ad, ρg, phase_ratios, rheology, _di, ni, λ_relaxation, dt, igg,
        controls = (;), gradients = nothing,
    )

Accumulate the sensitivities of the objective functional with respect to the material
parameters, given a converged adjoint state in `stokes_ad.λV` and `stokes_ad.λP`.

Fills

  - `stokes_ad.ρ` -- sensitivity with respect to density,
  - `stokes_ad.viscosity.η` and `stokes_ad.viscosity.ηv` -- sensitivity with respect to
    viscosity at the centers and vertices.
  - `gradients.G.center`, when supplied -- the derivative of the complete multiphase
    expression with respect to a center-based shear-modulus field, holding previous-step
    state fixed. It contains
    both the direct center contribution and the transpose-interpolated vertex contribution.
    Phases without elasticity (`G = Inf`) have zero sensitivity. `gradients.G.vertex`
    retains the uncombined vertex contribution for diagnostics.

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
        controls = (;),
        gradients = nothing,
    )
    if !isnothing(gradients)
        keys(gradients) == keys(controls) || throw(ArgumentError("controls and gradients must have matching parameters"))
        all(==(:G), keys(gradients)) || throw(ArgumentError("material gradients currently support only :G"))
        for name in keys(gradients), location in (:center, :vertex)
            A = getproperty(gradients[name], location)
            multiplier = getproperty(controls[name], location)
            dims = location === :center ? ni : ni .+ 1
            size(A) == size(multiplier) == dims ||
                throw(DimensionMismatch("$name $location fields must match the grid"))
            fill!(A, 0.0)
        end
    end
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
        stokes, stokes_ad, rheology, phase_ratios, λ_relaxation, dt, controls, gradients
    )

    if !isnothing(gradients) && haskey(gradients, :G)
        moduli = filter(isfinite, map(p -> get_shear_modulus(rheology, p), eachindex(rheology)))
        if !isempty(moduli)
            G = first(moduli)
            all(==(G), moduli) || throw(
                ArgumentError(
                    "a single G field requires the elastic phases to use the same base shear modulus"
                )
            )
            # G_local = multiplier * G, so dJ/dG_local = (dJ/dmultiplier) / G.
            center = gradients.G.center
            vertex = gradients.G.vertex
            center ./= G
            vertex ./= G
            combine_center_vertex_gradient!(center, vertex)
        end
    end

    gravity = compute_gravity(first(rheology))
    gx, gy = gravity isa Number ? (zero(gravity), gravity) : (gravity[1], gravity[3])
    # The residual depends on density through the buoyancy forces ρgx = ρ*gx and
    # ρgy = ρ*gy. The chain rule therefore gives dJ/dρ = gx*dJ/dρgx + gy*dJ/dρgy.
    @. stokes_ad.ρ = gx * dρgx + gy * stokes_ad.ρ

    return stokes_ad
end
