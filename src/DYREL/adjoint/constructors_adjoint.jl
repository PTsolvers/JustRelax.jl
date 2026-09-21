"""
    material_controls(CPUBackend, ni, parameters; nphases=nothing)

Allocate only the requested material fields. `:G` retains its center/vertex multipliers
and gradient buffers. Density parameters (`:ρ0`, `:α`, `:β`, `:T0`, `:P0`, `:ρ`)
allocate gradient outputs only, of size `(nphases, ni...)`; no input multipliers are needed.

Pass the returned `controls, gradients` to `solve_DYREL!`. Density gradients are
derivatives with respect to the numeric GeoParams parameter values, including buoyancy
and thermal pressure-residual contributions, with previous-step state held fixed.
A parameter absent from a phase's density model has zero gradient for that phase.
The direct path supports scalar GeoUnit fields of Enzyme-compatible density models;
nested parameter selection is not yet supported.
"""
is_density_parameter(name) = name in (:ρ0, :α, :β, :T0, :P0, :ρ)

function material_controls(
        ::Type{CPUBackend}, ni::NTuple{N, Integer}, parameters::NTuple{M, Symbol};
        nphases = nothing,
    ) where {N, M}
    if any(is_density_parameter, parameters)
        nphases isa Integer && nphases > 0 ||
            throw(ArgumentError("a positive nphases is required for density gradients"))
    end
    control_names = filter(name -> !is_density_parameter(name), parameters)
    multipliers = map(control_names) do _
        (; center = @ones(ni...), vertex = @ones(ni .+ 1...))
    end
    gradients = map(parameters) do parameter
        if is_density_parameter(parameter)
            (; center = @zeros(nphases, ni...))
        else
            (; center = @zeros(ni...), vertex = @zeros(ni .+ 1...))
        end
    end
    return NamedTuple{control_names}(multipliers), NamedTuple{parameters}(gradients)
end

function AdjointStokesArrays(::Type{CPUBackend}, ni::Vararg{Integer, N}) where {N}
    return AdjointStokesArrays(tuple(ni...))
end

function AdjointStokesArrays(::Type{CPUBackend}, ni::NTuple{N, Integer}) where {N}
    return AdjointStokesArrays(ni)
end

AdjointStokesArrays(ni::Vararg{Integer, N}) where {N} = AdjointStokesArrays(tuple(ni...))

function AdjointStokesArrays(ni::NTuple{N, Integer}) where {N}
    P = @zeros(ni...)
    θ = @zeros(ni...)
    λP = @zeros(ni...)
    P0 = @zeros(ni...)
    V = Velocity(ni...)
    λV = Velocity(ni...)
    ∇V = @zeros(ni...)
    τ = SymmetricTensor(ni...)
    ε = SymmetricTensor(ni...)
    ε_pl = SymmetricTensor(ni...)
    EII_pl = @zeros(ni...)
    viscosity = Viscosity(ni)
    τ_o = SymmetricTensor(ni...)
    R = Residual(ni...)
    U = Displacement(ni...)
    ω = Vorticity(ni...)
    η = @zeros(ni...)
    ρ = @zeros(ni...)

    return JustRelax.AdjointStokesArrays(
        P,
        θ,
        λP,
        P0,
        V,
        λV,
        ∇V,
        τ,
        ε,
        ε_pl,
        EII_pl,
        viscosity,
        τ_o,
        R,
        U,
        ω,
        η,
        ρ,
    )
end

function AdjointStokesArrays(::Number, ::Number)
    throw(ArgumentError("AdjointStokesArrays dimensions must be given as integers"))
end

function AdjointStokesArrays(::Number, ::Number, ::Number)
    throw(ArgumentError("AdjointStokesArrays dimensions must be given as integers"))
end
