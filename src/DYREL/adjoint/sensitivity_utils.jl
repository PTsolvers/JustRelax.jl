"""
    prepare_sensitivities!(stokes_ad, rheology, ni, gradients, igg)

Check and zero the `gradients` buffers, zero the adjoint working arrays and seed the momentum
residual adjoints with the converged `-λV`, ahead of the sensitivity passes.
"""
function prepare_sensitivities!(stokes_ad, rheology, ni, gradients, igg)
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
    stokes_ad.dρgx .= 0.0

    @views stokes_ad.R.Rx .= -stokes_ad.λV.Vx[2:(size(stokes_ad.R.Rx, 1) + 1), 2:(size(stokes_ad.R.Rx, 2) + 1)]
    @views stokes_ad.R.Ry .= -stokes_ad.λV.Vy[2:(size(stokes_ad.R.Ry, 1) + 1), 2:(size(stokes_ad.R.Ry, 2) + 1)]
    return nothing
end

"""
    finish_density_sensitivities!(
        stokes_ad, phase_ratios, rheology, args, dt, gradients, periodic; air_phase=0, ϕ=nothing,
    )

Turn the buoyancy adjoints into the density sensitivity `stokes_ad.ρ`, pull it back to the
density parameters and fold every vertex gradient into the centers. `air_phase` and `ϕ` must
match the forward buoyancy update and continuity residual.
"""
function finish_density_sensitivities!(
        stokes_ad, phase_ratios, rheology, args, dt, gradients, periodic;
        air_phase::Integer = 0, ϕ = nothing,
    )
    gravity = compute_gravity(first(rheology))
    gx, gy = gravity isa Number ? (zero(gravity), gravity) : (gravity[1], gravity[3])
    # The residual depends on density through the buoyancy forces ρgx = ρ*gx and
    # ρgy = ρ*gy. The chain rule therefore gives dJ/dρ = gx*dJ/dρgx + gy*dJ/dρgy.
    @. stokes_ad.ρ = gx * stokes_ad.dρgx + gy * stokes_ad.ρ

    compute_density_parameter_sensitivities!(
        stokes_ad, phase_ratios, rheology, args, dt, gradients; air_phase, ϕ
    )
    for gradient in values(gradients), p in eachindex(rheology)
        combine_center_vertex_gradient!(gradient.center, gradient.vertex, p, periodic)
    end

    return stokes_ad
end

const _derived_material_parameter_names = (:sinϕ, :cosϕ, :sinΨ, :cosΨ)

@inline _is_material_parameter(value::GeoParams.GeoUnit) = value.val isa AbstractFloat
@inline _is_material_parameter(value) = value isa AbstractFloat

@inline _is_parameter_container(value) = !(
    value isa GeoParams.GeoUnit || value isa Number || value isa AbstractArray ||
        value isa AbstractString || value isa Symbol || value isa Function ||
        value isa Type || isnothing(value)
)

function _collect_material_parameter_paths!(paths, object, name::Symbol, prefix::Tuple)
    name in _derived_material_parameter_names && return paths
    if object isa NamedTuple
        components = pairs(object)
    elseif object isa Tuple
        components = pairs(object)
    else
        components = ((field, getfield(object, field)) for field in fieldnames(typeof(object)))
    end

    for (component, value) in components
        path = (prefix..., component)
        component === name && _is_material_parameter(value) && push!(paths, path)
        _is_parameter_container(value) &&
            _collect_material_parameter_paths!(paths, value, name, path)
    end
    return paths
end

function _material_parameter_paths(object, name::Symbol, prefix::Tuple)
    return _collect_material_parameter_paths!(Tuple[], object, name, prefix)
end

@inline _density_parameter_paths(material, name::Symbol) =
    _material_parameter_paths(material.Density, name, (:Density,))

@inline _composite_parameter_paths(material, name::Symbol) =
    _material_parameter_paths(material.CompositeRheology, name, (:CompositeRheology,))

function _composite_parameter_paths(material, name::Symbol, predicate)
    paths = Tuple[]
    for (rheology_index, composite) in pairs(material.CompositeRheology)
        for (element_index, element) in pairs(composite.elements)
            predicate(element) || continue
            prefix = (:CompositeRheology, rheology_index, :elements, element_index)
            append!(paths, _material_parameter_paths(element, name, prefix))
        end
    end
    return paths
end

@inline _viscosity_parameter_paths(material, name::Symbol) =
    _composite_parameter_paths(material, name, element -> element isa GeoParams.AbstractCreepLaw)

function _stress_parameter_paths(material, name::Symbol)
    paths = _composite_parameter_paths(
        material, name,
        element -> element isa GeoParams.AbstractElasticity || element isa GeoParams.AbstractPlasticity,
    )
    filter!(paths) do path
        element = material.CompositeRheology[path[2]].elements[path[4]]
        element isa GeoParams.AbstractElasticity || return true
        value = _material_parameter_value(_material_parameter(material, Val(path)))
        return isfinite(value) && !iszero(value)
    end
    return paths
end

function _resolve_parameter_paths(material, names, path_function)
    return NamedTuple{names}(
        map(names) do name
            Tuple(Val(path) for path in path_function(material, name))
        end
    )
end

@inline _has_parameter_paths(parameters) = any(paths -> !isempty(paths), parameters)

function material_parameter_is_used(material, parameter)
    return !isempty(_density_parameter_paths(material, parameter)) ||
        !isempty(_composite_parameter_paths(material, parameter))
end

function compute_linear_viscosity_parameter_sensitivities!(
        stokes, adjoint, phases, rheology, args, viscosity_cutoff, gradients;
        air_phase::Integer = 0,
    )
    isempty(gradients) && return nothing
    names = keys(gradients)
    centers = map(entry -> entry.center, gradients)
    vertices = map(entry -> entry.vertex, gradients)
    for (p, material) in enumerate(rheology)
        parameters = _resolve_parameter_paths(
            material, names, _viscosity_parameter_paths
        )
        _has_parameter_paths(parameters) || continue

        @parallel (@idx size(phases.vertex)) linear_viscosity_parameter_sensitivity_kernel!(
            centers, vertices, parameters, material, rheology, Val(p),
            phases.center, phases.vertex,
            @strain_center(stokes), @tensor_vertex(stokes.ε), args,
            adjoint.viscosity.η, adjoint.viscosity.ηv, viscosity_cutoff, air_phase,
        )
    end
    return nothing
end

"""
    combine_center_vertex_gradient!(center, vertex, p, periodic = (false, false))

Fold phase `p`'s vertex gradient into its center gradient, the exact adjoint of
[`center2vertex!`](@ref): each interior vertex is the average of its four surrounding
centers. Nonperiodic boundary vertices copy the nearest interior value; periodic seam
vertices use the corresponding wrapped centers.

Written as a gather over centers rather than a scatter from vertices, so that no two
threads accumulate into the same cell.
"""
function combine_center_vertex_gradient!(
        center, vertex, p, periodic = (false, false)
    )
    ni = size(center)[2:end]
    size(vertex)[2:end] == ni .+ 1 || throw(
        DimensionMismatch(
            "the vertex gradient must be one point larger than the center gradient"
        ),
    )
    @parallel (@idx ni) combine_center_vertex_gradient_kernel!(center, vertex, p, periodic)
    return center
end

function compute_density_parameter_sensitivities!(
        adjoint, phases, rheology, args, dt, gradients; air_phase::Integer = 0, ϕ = nothing,
    )
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
                _is_material_parameter(value) ||
                    throw(ArgumentError("$name must be a scalar floating-point material parameter"))
            end
        end
        _has_parameter_paths(density_parameters) || continue

        @parallel (@idx size(adjoint.ρ)) density_parameter_sensitivity_kernel!(
            centers, density_parameters, material, p, phases.center, args,
            adjoint.ρ, adjoint.λP, ΔT, melt_fraction, dt, air_phase, ϕ,
        )
    end
    return nothing
end
