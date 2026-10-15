using Enzyme

"""
    enzyme_material_gradient(objective, material, args...)

Differentiate a scalar local `objective(material, args...)` with respect to the complete
GeoParams material object. All arguments after `material` are held constant. The returned
object has the same nested structure as `material` and contains the local reverse-mode
derivatives.

Keeping this operation local is what allows callers to retain one contribution per phase and
grid point instead of immediately reducing all uses of a phase parameter to one scalar.
"""
@generated function enzyme_material_gradient(objective, material, args::Vararg{Any, N}) where {N}
    constant_args = ntuple(i -> :(Enzyme.Const(args[$i])), N)
    return quote
        Enzyme.autodiff_deferred(
            Enzyme.Reverse,
            Enzyme.Const(objective),
            Enzyme.Active,
            Enzyme.Active(material),
            $(constant_args...),
        )[1][1]
    end
end

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

Extract a scalar parameter derivative from the nested object returned by
`enzyme_material_gradient`. Most paths are direct. GeoParams caches the trigonometric
forms of `ϕ` and `Ψ`, so those two paths additionally include the chain rule from the
cached fields back to the user-facing angles in degrees.
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

# Pick the thermal-expansion accessor the density model actually provides: the
# melt-dependent models take the melt fraction, while the plain ones (PT_Density and
# friends) only define the single-argument method.
@inline material_thermal_expansion(material, ::NamedTuple{()}) =
    get_thermal_expansion(material)
@inline material_thermal_expansion(material, thermal_args::NamedTuple) =
    get_thermal_expansion(material, thermal_args)

# Differentiate the original GeoParams evaluations with local reverse seeds.
# Phase weighting is applied to the seeds by the caller, exactly once.
@inline thermal_expansion_parameter_objective(material, thermal_args, seed) =
    seed * material_thermal_expansion(material, thermal_args)

"""
    enzyme_pointwise_material_gradients!(update!, n, fields, parameters, args)

Reverse-differentiate one local forward update `update!(primals..., i, j)` point by point.
`args` lists its arguments as pairs, as for [`enzyme_reverse_pointwise!`](@ref), with the
rheology given as `rheology => Enzyme.Active`.

Output seeds in the shadows are consumed by the reverse pass, while derivatives of duplicated
inputs accumulate in their shadows. The derivative with respect to `rheology` comes back per
point and holds every material parameter of every phase; the requested ones (`parameters`, one
resolved path set per phase) are added to `fields` at `[p, i, j]`. Because the forward function
itself is differentiated, every input reconstruction, mask, and phase weighting matches the
forward solve by construction.
"""
function enzyme_pointwise_material_gradients!(
        update!::F, n, fields, parameters, args::Tuple,
    ) where {F}
    primals, shadows = map(first, args), map(last, args)
    k = findfirst(==(Enzyme.Active), shadows)
    isnothing(k) &&
        throw(ArgumentError("the rheology must be passed as `rheology => Enzyme.Active`"))
    op = PointwiseMaterialGradient{F, k}(update!)
    context = (; fields, parameters)
    reverse_colored!(op, n, primals, shadows, context)
    return nothing
end

# Reverse one point update with rheology argument `k` active and store its material derivatives.
struct PointwiseMaterialGradient{F, k}
    update!::F
end

@inline function apply_point!(op::PointwiseMaterialGradient{F, k}, primals, shadows, context, i, j) where {F, k}
    (; fields, parameters) = context
    rheology = primals[k]
    annotations = map(annotate, primals, shadows)
    derivative = Enzyme.autodiff_deferred(
        Enzyme.Reverse, Enzyme.Const(op.update!), Enzyme.Const, annotations...,
        Enzyme.Const(i), Enzyme.Const(j),
    )[1][k]
    store_rheology_gradients!(fields, parameters, rheology, derivative, (i, j))
    return nothing
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

# rock fraction weighting the continuity residual; `1` without a rock ratio
Base.@propagate_inbounds @inline _continuity_weight(::Nothing, I...) = 1.0
Base.@propagate_inbounds @inline _continuity_weight(ϕ::JustRelax.RockRatio, I...) = ϕ.center[I...]

# Thermal expansion enters through the continuity residual rather than `compute_ρg!`. The
# variational solver additionally weights that residual by the rock fraction `ϕ`.
@parallel_indices (I...) function thermal_expansion_parameter_gradient_kernel!(
        centers, parameters, material, p, phase_ratios, λP, ΔT, melt_fraction, dt, ϕ,
    )
    raw_ratio = @cell phase_ratios[I...]
    ratio_α = raw_ratio[p] * _continuity_weight(ϕ, I...)
    if !iszero(ratio_α)
        thermal_args = isnothing(melt_fraction) ? (;) : (; ϕ = melt_fraction[I...])
        α_seed = -ratio_α * λP[I...] * ΔT[(I .+ 1)...] / dt
        derivative = enzyme_material_gradient(
            thermal_expansion_parameter_objective, material, thermal_args, α_seed
        )
        store_parameter_gradients!(centers, parameters, material, derivative, p, I)
    end
    return nothing
end

# These generated helpers unroll the gradient-buffer walk inside the kernels and keep each
# resolved material path available as a compile-time `Val` at the extraction site.
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
