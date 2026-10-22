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
        stokes, stokes_ad, ρg, phase_ratios, rheology, args, dt, gradients, periodic, _di;
        air_phase=0, ϕ=nothing,
    )

Differentiate the forward buoyancy and continuity-residual functions with respect to their
material parameters, form the diagnostic density adjoint `stokes_ad.ρ`, and fold every vertex
gradient into the centers. `air_phase` and `ϕ` must match the forward buoyancy update and
continuity residual.
"""
function finish_density_sensitivities!(
        stokes, stokes_ad, ρg, phase_ratios, rheology, args, dt, gradients, periodic,
        _di;
        air_phase::Integer = 0, ϕ = nothing,
    )
    compute_buoyancy_parameter_gradients!(
        ρg, stokes_ad, phase_ratios, rheology, args, gradients; air_phase
    )
    compute_continuity_parameter_gradients!(
        stokes, stokes_ad, phase_ratios, rheology, args, dt, gradients, _di, ϕ
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
        include_vertex::Bool = true,
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
    if include_vertex
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

function compute_buoyancy_parameter_gradients!(
        ρg, adjoint, phases, rheology, args, gradients; air_phase::Integer = 0,
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
    return nothing
end

# The bulk modulus reaches the continuity residual through `ηb = Kb * dt`. Only elastic
# elements carry it; `Kb = 0` disables elasticity, so it has no derivative.
_bulk_modulus_parameter_paths(material, name::Symbol) =
    name === :Kb ? _stress_parameter_paths(material, name) : Tuple[]

@inline _continuity_parameter_paths(material, name::Symbol) =
    (_density_parameter_paths(material, name)..., _bulk_modulus_parameter_paths(material, name)...)

"""
    compute_continuity_parameter_gradients!(
        stokes, adjoint, phases, rheology, args, dt, gradients, _di, ϕ=nothing,
    )

Accumulate the material-parameter contributions of the continuity residual, the bulk modulus
through the compressibility term and, when `args` contains `ΔT`, thermal expansion, by
differentiating the matching pointwise forward pressure-residual function. The bulk viscosity
is recomputed from `rheology` instead of read from the DYREL state, so that `Kb` is
differentiated. Passing `ϕ` selects the variational overload, including its rock-fraction
weighting and masked cells.
"""
function compute_continuity_parameter_gradients!(
        stokes, adjoint, phases, rheology, args, dt, gradients, _di, ϕ = nothing,
    )
    isempty(gradients) && return nothing
    names = keys(gradients)
    centers = map(entry -> entry.center, gradients)
    ΔT = get(args, :ΔT, nothing)
    path_function = isnothing(ΔT) ? _bulk_modulus_parameter_paths : _continuity_parameter_paths
    parameters = map(
        material -> _resolve_parameter_paths(material, names, path_function), rheology,
    )
    any(_has_parameter_paths, parameters) || return nothing
    melt_fraction = get(args, :melt_fraction, nothing)
    # Material sensitivities use the negative Lagrangian multiplier as the seed of the forward
    # pressure residual. Keep a separate array because reverse mode consumes output seeds.
    RP_seed = -copy(adjoint.λP)
    point_args = (
        stokes.ε.xx => nothing,
        stokes.ε.yy => nothing,
        stokes.ε.xy => nothing,
        stokes.V.Vx => nothing,
        stokes.V.Vy => nothing,
        copy(stokes.R.RP) => RP_seed,
        stokes.P => nothing,
        stokes.P0 => nothing,
        stokes.Q => nothing,
        nothing => nothing,
    )
    if !isnothing(ϕ)
        point_args = (point_args..., ϕ => nothing)
    end
    differentiate_pointwise_material_kernel!(
        compute_∇V_strain_rate_RP_point!, size(stokes.P), centers, parameters,
        (
            point_args...,
            _di.vertex => nothing,
            _di.velocity[1] => nothing,
            _di.velocity[2] => nothing,
            rheology => Enzyme.Active,
            phases.center => nothing,
            ΔT => nothing,
            melt_fraction => nothing,
            dt => nothing,
            false => nothing,
        ),
    )
    return nothing
end
