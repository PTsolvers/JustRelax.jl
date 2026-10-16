using Enzyme

@generated function _material_parameter(derivative, ::Val{path}) where {path}
    path isa Tuple || error("a material-parameter path must be a tuple")
    value = :derivative
    for component in path
        if component isa Symbol
            value = :(getproperty($value, $(QuoteNode(component))))
        elseif component isa Integer
            value = :($value[$component])
        else
            error("material-parameter paths support only field names and tuple indices")
        end
    end
    return value
end

@inline _material_parameter_value(parameter::GeoParams.GeoUnit) = parameter.val
@inline _material_parameter_value(parameter) = parameter

"""
    material_parameter_gradient(material, derivative, Val(path))

Extract a scalar parameter derivative from a nested material-derivative object returned by
Enzyme. Most paths are direct. GeoParams caches the trigonometric forms of `ϕ` and `Ψ`, so
those two paths additionally include the chain rule from the cached fields back to the
user-facing angles in degrees.
"""
@generated function material_parameter_gradient(material, derivative, ::Val{path}) where {path}
    path isa Tuple || error("a material-parameter path must be a tuple")
    isempty(path) && error("a material-parameter path must not be empty")
    name = last(path)
    name in (:ϕ, :Ψ) || return :(
        _material_parameter_value(
            _material_parameter(derivative, Val($(QuoteNode(path))))
        )
    )

    value(object, field) = begin
        field_path = (path[1:(end - 1)]..., field)
        :(
            _material_parameter_value(
                _material_parameter($object, Val($(QuoteNode(field_path))))
            )
        )
    end
    angle = value(:material, name)
    angle_gradient = value(:derivative, name)
    sin_gradient = value(:derivative, Symbol(:sin, name))
    cos_gradient = value(:derivative, Symbol(:cos, name))
    return quote
        $angle_gradient + oftype($angle, π / 180) * (
            cosd($angle) * $sin_gradient - sind($angle) * $cos_gradient
        )
    end
end

"""
    differentiate_pointwise_material_kernel!(
        update!, n, gradient_fields, parameter_paths_by_phase, args,
    )

Reverse-differentiate one local forward update `update!(primals..., i, j)` point by point.
`args` lists its arguments as pairs, as for [`enzyme_reverse_pointwise!`](@ref), with the
rheology given as `rheology => Enzyme.Active`.

Output seeds in the shadows are consumed by the reverse pass, while derivatives of duplicated
inputs accumulate in their shadows. The derivative with respect to `rheology` comes back per
point and holds every material parameter of every phase. The requested paths in
`parameter_paths_by_phase` are added to `gradient_fields` at `[p, i, j]`. Because the forward
function itself is differentiated, every input reconstruction, mask, and phase weighting
matches the forward solve by construction.
"""
function differentiate_pointwise_material_kernel!(
        update!::F, n, gradient_fields, parameter_paths_by_phase, args::Tuple,
    ) where {F}
    primals, shadows = map(first, args), map(last, args)
    rheology_argument = findfirst(==(Enzyme.Active), shadows)
    isnothing(rheology_argument) &&
        throw(ArgumentError("the rheology must be passed as `rheology => Enzyme.Active`"))
    op = PointwiseMaterialGradient{F, rheology_argument}(update!)
    context = (; gradient_fields, parameter_paths_by_phase)
    reverse_colored!(op, n, primals, shadows, context)
    return nothing
end

# Reverse one point update with the rheology argument active and store its material derivatives.
struct PointwiseMaterialGradient{F, rheology_argument}
    update!::F
end

@inline function apply_point!(
        op::PointwiseMaterialGradient{F, rheology_argument}, primals, shadows, context, i, j,
    ) where {F, rheology_argument}
    (; gradient_fields, parameter_paths_by_phase) = context
    rheology = primals[rheology_argument]

    # Array fields paired with adjoint arrays become `DuplicatedNoNeed` arguments. Enzyme
    # therefore accumulates their reverse derivatives directly into the supplied adjoint
    # arrays, for example `adjoint.ε`, `adjoint.P`, and `adjoint.viscosity.η`.
    # The rheology is different: it is an `Active` argument without a shadow object, so its
    # complete nested derivative is returned by `autodiff_deferred` instead.
    annotations = map(annotate, primals, shadows)
    rheology_derivative = Enzyme.autodiff_deferred(
        Enzyme.Reverse, Enzyme.Const(op.update!), Enzyme.Const, annotations...,
        Enzyme.Const(i), Enzyme.Const(j),
    )[1][rheology_argument]

    # Select only the material-parameter paths requested in `gradients` from the returned
    # rheology derivative and accumulate them in the phase- and point-resolved fields.
    store_rheology_gradients!(
        gradient_fields, parameter_paths_by_phase, rheology, rheology_derivative, (i, j)
    )
    return nothing
end

# Extract the requested compile-time material paths and accumulate their derivatives in the
# corresponding phase- and point-resolved gradient fields.
@generated function store_parameter_gradients!(
        fields::NamedTuple{names}, parameters::NamedTuple{names, parameter_types},
        material, derivative, p, I::NTuple,
    ) where {names, parameter_types}
    writes = Expr[]
    for k in eachindex(names)
        paths_type = parameter_types.parameters[k]
        for path_index in 1:fieldcount(paths_type)
            push!(
                writes, quote
                    fields[$k][p, I...] += material_parameter_gradient(
                        material, derivative, parameters[$k][$path_index]
                    )
                end
            )
        end
    end
    return quote
        @inbounds begin
            $(writes...)
        end
        return nothing
    end
end

# `store_parameter_gradients!` for every phase of a whole-rheology derivative; `parameters[p]`
# holds the resolved paths of phase `p`.
@generated function store_rheology_gradients!(
        fields, parameters::NTuple{N, Any}, rheology, derivative, I
    ) where {N}
    return quote
        Base.@nexprs $N p -> store_parameter_gradients!(
            fields, parameters[p], rheology[p], derivative[p], p, I
        )
        return nothing
    end
end

@parallel_indices (i, j) function combine_center_vertex_gradient_kernel!(
        center, vertex, p, periodic
    )
    nx, ny = size(center, 2), size(center, 3)
    total = zero(eltype(center))
    @inbounds begin
        iv_left = periodic[1] ? (i == nx ? 1 : 0) : (i == 2 ? 1 : 0)
        iv_right = periodic[1] ? (i == 1 ? nx + 1 : 0) : (i == nx - 1 ? nx + 1 : 0)
        jv_bot = periodic[2] ? (j == ny ? 1 : 0) : (j == 2 ? 1 : 0)
        jv_top = periodic[2] ? (j == 1 ? ny + 1 : 0) : (j == ny - 1 ? ny + 1 : 0)
        for iv in (i, i + 1, iv_left, iv_right), jv in (j, j + 1, jv_bot, jv_top)
            (iszero(iv) || iszero(jv)) && continue
            iv_eff = periodic[1] ? iv : clamp(iv, 2, nx)
            jv_eff = periodic[2] ? jv : clamp(jv, 2, ny)
            i0 = periodic[1] ? mod1(iv_eff - 1, nx) : iv_eff - 1
            ic = periodic[1] ? mod1(iv_eff, nx) : iv_eff
            j0 = periodic[2] ? mod1(jv_eff - 1, ny) : jv_eff - 1
            jc = periodic[2] ? mod1(jv_eff, ny) : jv_eff
            weight = ((i0 == i) + (ic == i)) * ((j0 == j) + (jc == j))
            total += weight * vertex[p, iv, jv]
        end
        center[p, i, j] += total / 4
    end
    return nothing
end
