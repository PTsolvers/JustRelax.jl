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
        stokes_ad, ρg, phase_ratios, rheology, args, dt, gradients, periodic;
        air_phase=0, ϕ=nothing,
    )

Differentiate the forward buoyancy function with respect to its material parameters, add the
thermal-expansion contribution from continuity, form the diagnostic density adjoint
`stokes_ad.ρ`, and fold every vertex gradient into the centers. `air_phase` and `ϕ` must match
the forward buoyancy update and continuity residual.
"""
function finish_density_sensitivities!(
        stokes_ad, ρg, phase_ratios, rheology, args, dt, gradients, periodic;
        air_phase::Integer = 0, ϕ = nothing,
    )
    compute_buoyancy_parameter_gradients!(
        ρg, stokes_ad, phase_ratios, rheology, args, dt, gradients; air_phase, ϕ
    )

    gravity = compute_gravity(first(rheology))
    gx, gy = gravity isa Number ? (zero(gravity), gravity) : (gravity[1], gravity[3])
    # The residual depends on density through the buoyancy forces ρgx = ρ*gx and
    # ρgy = ρ*gy. The chain rule therefore gives dJ/dρ = gx*dJ/dρgx + gy*dJ/dρgy.
    @. stokes_ad.ρ = gx * stokes_ad.dρgx + gy * stokes_ad.ρ

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

@inline _gravity_parameter_paths(material, name::Symbol) =
    _material_parameter_paths(material.Gravity, name, (:Gravity,))

@inline _buoyancy_parameter_paths(material, name::Symbol) =
    (_density_parameter_paths(material, name)..., _gravity_parameter_paths(material, name)...)

@inline _composite_parameter_paths(material, name::Symbol) =
    _material_parameter_paths(material.CompositeRheology, name, (:CompositeRheology,))

function _parameter_has_parent(material, path, predicate)
    object = material
    for component in path[1:(end - 1)]
        object = component isa Symbol ? getproperty(object, component) : object[component]
        predicate(object) && return true
    end
    return false
end

function _composite_parameter_paths(material, name::Symbol, predicate)
    paths = _composite_parameter_paths(material, name)
    filter!(path -> _parameter_has_parent(material, path, predicate), paths)
    return paths
end

function _finite_parameter_paths!(paths, material)
    filter!(paths) do path
        value = _material_parameter_value(_material_parameter(material, Val(path)))
        return isfinite(value)
    end
    return paths
end

# Search all ancestors so laws nested inside Parallel or another composite route to the
# pointwise kernel that actually evaluates them.
function _viscosity_parameter_paths(material, name::Symbol)
    paths = _composite_parameter_paths(
        material, name, element -> element isa GeoParams.AbstractCreepLaw,
    )
    return _finite_parameter_paths!(paths, material)
end

function _stress_parameter_paths(material, name::Symbol)
    paths = _composite_parameter_paths(
        material, name,
        element -> element isa GeoParams.AbstractElasticity || element isa GeoParams.AbstractPlasticity,
    )
    _finite_parameter_paths!(paths, material)
    filter!(paths) do path
        return !_parameter_has_parent(
            material, path, element -> element isa GeoParams.AbstractElasticity,
        ) || !iszero(_material_parameter_value(_material_parameter(material, Val(path))))
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
        !isempty(_gravity_parameter_paths(material, parameter)) ||
        !isempty(_composite_parameter_paths(material, parameter))
end

function compute_viscosity_parameter_gradients!(
        stokes, adjoint, phases, rheology, args, viscosity_cutoff, gradients;
        air_phase::Integer = 0,
        viscosity_seed = (adjoint.viscosity.η, adjoint.viscosity.ηv),
    )
    isempty(gradients) && return nothing
    names = keys(gradients)
    parameters = map(
        material -> _resolve_parameter_paths(material, names, _viscosity_parameter_paths),
        rheology,
    )
    any(_has_parameter_paths, parameters) || return nothing
    η_bar, ηv_bar = viscosity_seed

    differentiate_pointwise_material_kernel!(
        compute_viscosity_kernel_point!, size(phases.center),
        map(entry -> entry.center, gradients), parameters,
        (
            stokes.viscosity.η => copy(η_bar),
            1.0 => nothing,
            phases.center => nothing,
            stokes.τ.xx => nothing,
            stokes.τ.yy => nothing,
            stokes.τ.xy_c => nothing,
            args => nothing,
            rheology => Enzyme.Active,
            air_phase => nothing,
            viscosity_cutoff => nothing,
            compute_viscosity_τII => nothing,
            local_viscosity_args => nothing,
        ),
    )
    differentiate_pointwise_material_kernel!(
        compute_viscosity_kernel_point!, size(phases.vertex),
        map(entry -> entry.vertex, gradients), parameters,
        (
            stokes.viscosity.ηv => copy(ηv_bar),
            1.0 => nothing,
            phases.vertex => nothing,
            stokes.τ.xx_v => nothing,
            stokes.τ.yy_v => nothing,
            stokes.τ.xy => nothing,
            args => nothing,
            rheology => Enzyme.Active,
            air_phase => nothing,
            viscosity_cutoff => nothing,
            compute_viscosity_τII => nothing,
            local_viscosity_args_vertex => nothing,
        ),
    )
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

function compute_buoyancy_parameter_gradients!(
        ρg, adjoint, phases, rheology, args, dt, gradients;
        air_phase::Integer = 0, ϕ = nothing,
    )
    isempty(gradients) && return nothing
    names = keys(gradients)
    centers = map(entry -> entry.center, gradients)
    parameters = map(
        material -> _resolve_parameter_paths(material, names, _buoyancy_parameter_paths),
        rheology,
    )
    any(_has_parameter_paths, parameters) || return nothing
    args_without_P, P = split_pressure_arg(args)
    differentiate_pointwise_material_kernel!(
        compute_ρg_kernel_point!, size(adjoint.ρ), centers, parameters,
        (
            ρg => (copy(adjoint.dρgx), copy(adjoint.ρ)),
            phases.center => nothing,
            rheology => Enzyme.Active,
            args_without_P => nothing,
            P => nothing,
            air_phase => nothing,
            size(adjoint.ρ) => nothing,
        ),
    )

    ΔT = get(args, :ΔT, nothing)
    isnothing(ΔT) && return nothing
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

        @parallel (@idx size(adjoint.ρ)) thermal_expansion_parameter_gradient_kernel!(
            centers, density_parameters, material, p, phases.center,
            adjoint.λP, ΔT, melt_fraction, dt, ϕ,
        )
    end
    return nothing
end
