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
    name in (:ϕ, :Ψ) || return :(_material_parameter_value(
        _material_parameter(derivative, Val($(QuoteNode(path))))
    ))

    value(object, field) = begin
        field_path = (path[1:(end - 1)]..., field)
        :(_material_parameter_value(
            _material_parameter($object, Val($(QuoteNode(field_path))))
        ))
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
@inline function density_parameter_objective(material, args, ρ_seed, α_seed, thermal_args)
    result = ρ_seed * compute_density(material, args)
    if !iszero(α_seed)
        result += α_seed * material_thermal_expansion(material, thermal_args)
    end
    return result
end

@inline function enzyme_density_parameter_gradient(material, args, ρ_seed, α_seed, thermal_args)
    return enzyme_material_gradient(
        density_parameter_objective, material, args, ρ_seed, α_seed, thermal_args
    )
end

# Contract one phase's constitutive outputs with the converged adjoint seeds. This calls
# the same local stress update as the forward kernel; only the surrounding scalar
# contraction is specific to sensitivity evaluation.
@inline function stress_parameter_objective(
        material,
        εij,
        τij_o,
        η,
        P,
        λ,
        λ_relaxation,
        dt,
        EII,
        τ_seed,
        θ_seed,
        ratio,
    )
    G = get_shear_modulus(material)
    Kb = get_bulk_modulus(material)
    solution = _compute_local_stress(
        εij, τij_o, η, P, G, Kb, λ, λ_relaxation, material, dt, EII
    )
    return ratio * (
        τ_seed[1] * solution[1] +
            τ_seed[2] * solution[2] +
            τ_seed[3] * solution[3] +
            θ_seed * solution[9]
    )
end

@generated function enzyme_stress_gradients(
        material, εij, τij_o, η, args::Vararg{Any, N}
    ) where {N}
    constant_args = ntuple(i -> :(Enzyme.Const(args[$i])), N)
    return quote
        derivatives = Enzyme.autodiff_deferred(
            Enzyme.Reverse,
            Enzyme.Const(stress_parameter_objective),
            Enzyme.Active,
            Enzyme.Active(material),
            Enzyme.Const(εij),
            Enzyme.Const(τij_o),
            Enzyme.Active(η),
            $(constant_args...),
        )[1]
        derivative = derivatives[1]
        dη = derivatives[4]
        # Enzyme returns `nothing` for an Active scalar that is inactive on the
        # executed material branch. Its derivative contribution is then zero.
        return derivative, isnothing(dη) ? zero(η) : dη
    end
end

@inline function viscosity_parameter_objective(
        material, rheology, ::Val{p}, ratio, AII, args, seed, cutoff
    ) where {p}
    local_rheology = Base.setindex(rheology, material, p)
    η = compute_phase_viscosity(
        local_rheology, ratio, AII, compute_viscosity_εII, args
    )
    return seed * clamp(η, cutoff...)
end

@inline function enzyme_viscosity_parameter_gradient(material, args...)
    return enzyme_material_gradient(viscosity_parameter_objective, material, args...)
end

@parallel_indices (I...) function linear_viscosity_parameter_sensitivity_kernel!(
        centers, vertices, parameters, material, rheology, phase::Val{p},
        phase_center, phase_vertex, ε_center, ε_vertex, args,
        η_seed, ηv_seed, cutoff,
    ) where {p}
    Base.@propagate_inbounds @inline AII(A) = begin
        AII_0 = allzero(A...) * eps()
        second_invariant(AII_0 + A[1], -AII_0 + A[2], A[3])
    end
    ni = size(phase_center)
    @inbounds begin
        derivative = enzyme_viscosity_parameter_gradient(
            material, rheology, phase, phase_vertex[I...],
            AII((ε_vertex[1][I...], ε_vertex[2][I...], ε_vertex[3][I...])),
            local_viscosity_args_vertex(args, I...), ηv_seed[I...], cutoff,
        )
        store_parameter_gradients!(vertices, parameters, material, derivative, p, I)

        if all(I .≤ ni)
            derivative = enzyme_viscosity_parameter_gradient(
                material, rheology, phase, phase_center[I...],
                AII((ε_center[1][I...], ε_center[2][I...], ε_center[3][I...])),
                local_viscosity_args(args, I...), η_seed[I...], cutoff,
            )
            store_parameter_gradients!(centers, parameters, material, derivative, p, I)
        end
    end
    return nothing
end

@parallel_indices (I...) function stress_sensitivity_kernel!(
        centers, vertices, parameters,
        material, p, phase_center, phase_vertex,
        τ_o, τ_ov, ε, EII_pl, P, λ, λv, η, ηv,
        η_gradient, ηv_gradient,
        τ_seed, τv_seed, θ_seed, λ_relaxation, dt, periodic,
    )
    Base.@propagate_inbounds @inline av(A) = sum(JustRelax2D._gather(A, I...)) / 4
    ni = size(phase_center)
    @inbounds begin
        Ic = clamped_indices(ni, periodic, I...)
        ratio = phase_vertex[I...][p]
        derivative, dη = enzyme_stress_gradients(
            material,
            (av_clamped(ε[1], Ic...), av_clamped(ε[2], Ic...), ε[3][I...]),
            (τ_ov[1][I...], τ_ov[2][I...], τ_ov[3][I...]),
            ηv[I...], av_clamped(P, Ic...), λv[I...], λ_relaxation, dt,
            av_clamped(EII_pl, Ic...),
            (τv_seed[1][I...], τv_seed[2][I...], τv_seed[3][I...]),
            0.0, ratio,
        )
        ηv_gradient[I...] += dη
        store_parameter_gradients!(vertices, parameters, material, derivative, p, I)

        if all(I .≤ ni)
            ratio = phase_center[I...][p]
            derivative, dη = enzyme_stress_gradients(
                material,
                (ε[1][I...], ε[2][I...], av(ε[3])),
                (τ_o[1][I...], τ_o[2][I...], τ_o[3][I...]),
                η[I...], P[I...], λ[I...], λ_relaxation, dt, EII_pl[I...],
                (τ_seed[1][I...], τ_seed[2][I...], τ_seed[3][I...]),
                θ_seed[I...], ratio,
            )
            η_gradient[I...] += dη
            store_parameter_gradients!(centers, parameters, material, derivative, p, I)
        end
    end
    return nothing
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

@parallel_indices (I...) function density_parameter_sensitivity_kernel!(
        centers, parameters, material, p, phase_ratios, args, dρ, λP, ΔT, melt_fraction, dt
    )
    ratio = (@cell phase_ratios[I...])[p]
    if !iszero(ratio)
        local_args = getindex_NamedTuple(args, I...)
        thermal_args = isnothing(melt_fraction) ? (;) : (; ϕ = melt_fraction[I...])
        α_seed = isnothing(ΔT) ? 0.0 : -λP[I...] * ΔT[(I .+ 1)...] / dt
        derivative = enzyme_density_parameter_gradient(
            material, local_args, ratio * dρ[I...], ratio * α_seed, thermal_args
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
            push!(writes, quote
                fields[$k][p, I...] += material_parameter_gradient(
                    material, derivative, parameters[$k][$path_index]
                )
            end)
        end
    end
    return quote
        @inbounds begin
            $(writes...)
        end
        return nothing
    end
end
