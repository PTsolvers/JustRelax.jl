"""
    compute_sensitivities!(
        stokes, stokes_ad, ρg, phase_ratios, rheology, _di, ni, λ_relaxation, dt, igg,
        gradients = (;), args = (;); viscosity_cutoff = (-Inf, Inf),
    )

Accumulate the sensitivities of the objective functional with respect to the material
parameters, given a converged adjoint state in `stokes_ad.λV` and `stokes_ad.λP`.

Fills

  - `stokes_ad.ρ` -- sensitivity with respect to density,
  - `stokes_ad.viscosity.η` and `stokes_ad.viscosity.ηv` -- sensitivity with respect to
    viscosity at the centers and vertices.
  - each `gradients` entry -- the accumulated phase-wise derivative with respect to every
    use of the requested parameter name. Center arrays include the transpose-interpolated
    vertex contribution; vertex arrays retain that contribution separately for diagnostics.

`gradients` comes from [`material_controls`](@ref) and is keyed directly by parameter name.
Each name is resolved in every supported material-function path before the sensitivity
kernels launch.

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
        gradients = (;),
        args = (;),
        ;
        viscosity_cutoff = (-Inf, Inf),
    )

    center_dims = (length(rheology), ni...)
    vertex_dims = (length(rheology), (ni .+ 1)...)
    for name in keys(gradients)
        # Check that the parameter is used and its phase-wise buffers fit the grid.
        any(material -> material_parameter_is_used(material, name), rheology) ||
            throw(ArgumentError("no supported material function uses $name"))
        size(gradients[name].center) == center_dims ||
            throw(DimensionMismatch("$name center field must have size $center_dims"))
        size(gradients[name].vertex) == vertex_dims ||
            throw(DimensionMismatch("$name vertex field must have size $vertex_dims"))
        # Sensitivity paths add into these buffers with `+=`.
        fill!(gradients[name].center, 0.0)
        fill!(gradients[name].vertex, 0.0)
    end

    igg.me == 0 && @printf("\n######## Calculate Sensitivities ########\n")
    # zero out adjoint arrays
    stokes_ad.P            .= 0.0
    stokes_ad.θ            .= 0.0
    stokes_ad.τ.xx         .= 0.0
    stokes_ad.τ.yy         .= 0.0
    stokes_ad.τ.xy_c       .= 0.0
    stokes_ad.τ.xx_v       .= 0.0
    stokes_ad.τ.yy_v       .= 0.0
    stokes_ad.τ.xy         .= 0.0
    stokes_ad.viscosity.η  .= 0.0
    stokes_ad.viscosity.ηv .= 0.0
    stokes_ad.ρ            .= 0.0
    stokes_ad.dρgx         .= 0.0

    @views stokes_ad.R.Rx .= -stokes_ad.λV.Vx[2:(end - 1), 2:(end - 1)]
    @views stokes_ad.R.Ry .= -stokes_ad.λV.Vy[2:(end - 1), 2:(end - 1)]

    # differntiates momentum equation w.r.t. stress, pressure and plastic pressuure correction
    enzyme_compute_PH_residual_V_sensitivity!(stokes, stokes_ad, ρg, _di, ni)

    # Pull back the local stress update to its material parameters and viscosity fields.
    compute_stress_sensitivities!(
        stokes, stokes_ad, phase_ratios, rheology, λ_relaxation, dt, gradients
    )

    compute_linear_viscosity_parameter_sensitivities!(
        stokes, stokes_ad, phase_ratios, rheology, args, viscosity_cutoff, gradients
    )

    gravity = compute_gravity(first(rheology))
    gx, gy = gravity isa Number ? (zero(gravity), gravity) : (gravity[1], gravity[3])
    # The residual depends on density through the buoyancy forces ρgx = ρ*gx and
    # ρgy = ρ*gy. The chain rule therefore gives dJ/dρ = gx*dJ/dρgx + gy*dJ/dρgy.
    @. stokes_ad.ρ = gx * stokes_ad.dρgx + gy * stokes_ad.ρ

    compute_density_parameter_sensitivities!(stokes_ad, phase_ratios, rheology, args, dt, gradients)
    for gradient in values(gradients), p in eachindex(rheology)
        combine_center_vertex_gradient!(gradient.center, gradient.vertex, p)
    end

    return stokes_ad
end

function _density_parameter_paths(material, name::Symbol)
    paths = Tuple[]
    for (model_index, model) in pairs(material.Density)
        hasproperty(model, name) && push!(paths, (:Density, model_index, name))
    end
    return paths
end

function _composite_parameter_paths(material, name::Symbol, model_type)
    paths = Tuple[]
    for (rheology_index, composite) in pairs(material.CompositeRheology)
        for (element_index, element) in pairs(composite.elements)
            element isa model_type && hasproperty(element, name) && push!(
                paths,
                (:CompositeRheology, rheology_index, :elements, element_index, name),
            )
        end
    end
    return paths
end

@inline _linear_viscosity_parameter_paths(material, name::Symbol) =
    _composite_parameter_paths(material, name, GeoParams.LinearViscous)

@inline _plastic_parameter_paths(material, name::Symbol) =
    _composite_parameter_paths(material, name, GeoParams.AbstractPlasticity)

function _elastic_parameter_paths(material, name::Symbol)
    paths = Tuple[]
    for (rheology_index, composite) in pairs(material.CompositeRheology)
        for (element_index, element) in pairs(composite.elements)
            element isa GeoParams.AbstractElasticity && hasproperty(element, name) || continue
            value = _material_parameter_value(getproperty(element, name))
            isfinite(value) && !iszero(value) && push!(
                paths,
                (:CompositeRheology, rheology_index, :elements, element_index, name),
            )
        end
    end
    return paths
end

function _stress_parameter_paths(material, name::Symbol)
    return (_elastic_parameter_paths(material, name)..., _plastic_parameter_paths(material, name)...)
end

function _resolve_parameter_paths(material, names, path_function)
    return NamedTuple{names}(map(names) do name
        Tuple(Val(path) for path in path_function(material, name))
    end)
end

@inline _has_parameter_paths(parameters) = any(paths -> !isempty(paths), parameters)

function material_parameter_is_used(material, parameter)
    return !isempty(_density_parameter_paths(material, parameter)) ||
        !isempty(_linear_viscosity_parameter_paths(material, parameter)) ||
        !isempty(_stress_parameter_paths(material, parameter))
end

function compute_linear_viscosity_parameter_sensitivities!(
        stokes, adjoint, phases, rheology, args, viscosity_cutoff, gradients
    )
    isempty(gradients) && return nothing
    names = keys(gradients)
    centers = map(entry -> entry.center, gradients)
    vertices = map(entry -> entry.vertex, gradients)
    for (p, material) in enumerate(rheology)
        parameters = _resolve_parameter_paths(
            material, names, _linear_viscosity_parameter_paths
        )
        _has_parameter_paths(parameters) || continue

        @parallel (@idx size(phases.vertex)) linear_viscosity_parameter_sensitivity_kernel!(
            centers, vertices, parameters, material, rheology, Val(p),
            phases.center, phases.vertex,
            @strain_center(stokes), @tensor_vertex(stokes.ε), args,
            adjoint.viscosity.η, adjoint.viscosity.ηv, viscosity_cutoff,
        )
    end
    return nothing
end

function compute_stress_sensitivities!(
        stokes, adjoint, phases, rheology, λ_relaxation, dt, gradients
    )
    names = keys(gradients)
    centers = map(entry -> entry.center, gradients)
    vertices = map(entry -> entry.vertex, gradients)
    for (p, material) in enumerate(rheology)
        parameters = _resolve_parameter_paths(material, names, _stress_parameter_paths)

        @parallel (@idx size(phases.vertex)) stress_sensitivity_kernel!(
            centers, vertices, parameters,
            material, p, phases.center, phases.vertex,
            (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c),
            (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy),
            (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy),
            stokes.EII_pl, stokes.P, stokes.λ, stokes.λv,
            stokes.viscosity.η, stokes.viscosity.ηv,
            adjoint.viscosity.η, adjoint.viscosity.ηv,
            (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c),
            (adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy),
            adjoint.θ, λ_relaxation, dt,
        )
    end
    return nothing
end

"""
    combine_center_vertex_gradient!(center, vertex, p)

Fold phase `p`'s vertex gradient into its center gradient, the exact adjoint of
[`center2vertex!`](@ref): each interior vertex is the average of its four surrounding
centers and each boundary vertex copies the nearest interior one, which the clamped index
reproduces.

Written as a gather over centers rather than a scatter from vertices, so that no two
threads accumulate into the same cell.
"""
function combine_center_vertex_gradient!(center, vertex, p)
    ni = size(center)[2:end]
    size(vertex)[2:end] == ni .+ 1 || throw(
        DimensionMismatch(
            "the vertex gradient must be one point larger than the center gradient"
        ),
    )
    @parallel (@idx ni) combine_center_vertex_gradient_kernel!(center, vertex, p)
    return center
end

function compute_density_parameter_sensitivities!(adjoint, phases, rheology, args, dt, gradients)
    isempty(gradients) && return nothing
    names = keys(gradients)
    centers = map(entry -> entry.center, gradients)
    ΔT = get(args, :ΔT, nothing)
    melt_fraction = get(args, :melt_fraction, nothing)
    for (p, material) in enumerate(rheology)
        density_parameters = _resolve_parameter_paths(material, names, _density_parameter_paths)
        for (name, paths) in pairs(density_parameters)
            for path in paths
                value = _material_parameter(material, path)
                value isa GeoParams.GeoUnit && value.val isa AbstractFloat ||
                    throw(ArgumentError("$name must be a scalar floating-point GeoUnit"))
            end
        end
        _has_parameter_paths(density_parameters) || continue

        @parallel (@idx size(adjoint.ρ)) density_parameter_sensitivity_kernel!(
            centers, density_parameters, material, p, phases.center, args,
            adjoint.ρ, adjoint.λP, ΔT, melt_fraction, dt,
        )
    end
    return nothing
end
