"""
    compute_sensitivities!(
        stokes, stokes_ad, ρg, phase_ratios, rheology, _di, ni, λ_relaxation, dt, igg,
        controls = (;), gradients = nothing, args = (;),
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
  - `gradients.<density_parameter>.center[p, ..]` -- the local derivative for phase `p`,
    including density/buoyancy and thermal pressure-residual contributions.

The adjoint working arrays (`P`, `θ`, the stress tensor and the viscosity fields) are zeroed
on entry, so this has to be called *after* the adjoint iterations have converged and not in
between them.
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
        args = (;),
    )
    if !isnothing(gradients)
        control_names = filter(name -> !is_density_parameter(name), keys(gradients))
        keys(controls) == control_names ||
            throw(ArgumentError("controls must match the non-density gradient parameters"))
        all(name -> name === :G || is_density_parameter(name), keys(gradients)) ||
            throw(ArgumentError("material gradients support :G and scalar density parameters"))
        for name in keys(gradients)
            if is_density_parameter(name)
                dims = (length(rheology), ni...)
                size(gradients[name].center) == dims ||
                    throw(DimensionMismatch("$name center field must have size $dims"))
                fill!(gradients[name].center, 0.0)
            else
                for location in (:center, :vertex)
                    A = getproperty(gradients[name], location)
                    multiplier = getproperty(controls[name], location)
                    dims = location === :center ? ni : ni .+ 1
                    size(A) == size(multiplier) == dims ||
                        throw(DimensionMismatch("$name $location fields must match the grid"))
                    fill!(A, 0.0)
                end
            end
        end
    end
    igg.me == 0 && @printf("\n######## Calculate Sensitivities ########\n")

    stokes_ad.P .= 0.0
    stokes_ad.θ .= 0.0
    stokes_ad.τ.xx .= 0.0
    stokes_ad.τ.yy .= 0.0
    stokes_ad.τ.xy_c .= 0.0
    stokes_ad.τ.xx_v .= 0.0
    stokes_ad.τ.yy_v .= 0.0
    stokes_ad.τ.xy .= 0.0
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
        stokes, stokes_ad, rheology, phase_ratios, λ_relaxation, dt, controls,
        isnothing(gradients) ? nothing : NamedTuple{keys(controls)}(map(name -> gradients[name], keys(controls))),
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

    if !isnothing(gradients)
        density_names = filter(is_density_parameter, keys(gradients))
        density_gradients = NamedTuple{density_names}(map(name -> gradients[name], density_names))
        compute_density_parameter_sensitivities!(stokes_ad, phase_ratios, rheology, args, dt, density_gradients)
    end

    return stokes_ad
end

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

function compute_density_parameter_sensitivities!(adjoint, phases, rheology, args, dt, gradients)
    isempty(gradients) && return nothing
    for name in keys(gradients)
        any(material -> hasproperty(first(material.Density), name), rheology) ||
            throw(ArgumentError("no phase density model has parameter $name"))
    end
    ΔT = get(args, :ΔT, nothing)
    melt_fraction = get(args, :melt_fraction, nothing)
    for (p, material) in enumerate(rheology)
        model = first(material.Density)
        for name in keys(gradients)
            if hasproperty(model, name)
                parameter = getproperty(model, name)
                parameter isa GeoParams.GeoUnit && parameter.val isa AbstractFloat ||
                    throw(ArgumentError("$name must be a scalar floating-point GeoUnit"))
            end
        end
        @parallel (@idx size(adjoint.ρ)) density_parameter_sensitivity_kernel!(
            gradients, model, p, phases.center, args, adjoint.ρ, adjoint.λP, ΔT, melt_fraction, dt
        )
    end
    return nothing
end

@parallel_indices (I...) function density_parameter_sensitivity_kernel!(
        gradients, model, p, phase_ratios, args, dρ, λP, ΔT, melt_fraction, dt
    )
    ratio = (@cell phase_ratios[I...])[p]
    for name in keys(gradients)
        gradients[name].center[p, I...] = 0.0
    end
    if !iszero(ratio)
        local_args = getindex_NamedTuple(args, I...)
        thermal_args = isnothing(melt_fraction) ? (;) : (; ϕ = melt_fraction[I...])
        α_seed = isnothing(ΔT) ? 0.0 : -λP[I...] * ΔT[(I .+ 1)...] / dt
        derivative = enzyme_density_parameter_gradient(
            model, local_args, ratio * dρ[I...], ratio * α_seed, thermal_args
        )
        for name in keys(gradients)
            if hasproperty(model, name)
                gradients[name].center[p, I...] = getproperty(derivative, name).val
            end
        end
    end
    return nothing
end
