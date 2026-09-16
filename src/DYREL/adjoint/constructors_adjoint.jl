"""
    material_controls(CPUBackend, ni, parameters)

Allocate center and vertex multiplier fields and matching zero-valued gradient fields only
for the selected material-parameter symbols. Multipliers start at one, so selecting a
parameter does not change the forward problem. Each field differentiates the complete
multiphase constitutive expression; there is no separate control per material phase.

Pass both returned named tuples as `controls` and `gradients` to
`solve_DYREL!(...; adjoint = true, controls, gradients, ...)`. Currently `:G` is
supported for material sensitivities in the 2D adjoint solver. After the solve,
`gradients.G.center` contains the full derivative with respect to a center-based local
shear modulus, including the vertex contribution. `gradients.G.vertex` retains the raw
vertex contribution for diagnostics. Calls without `gradients` leave controls inactive
in the sensitivity pass.
"""
function material_controls(
        ::Type{CPUBackend}, ni::NTuple{N, Integer}, parameters::NTuple{M, Symbol}
    ) where {N, M}
    multipliers = map(parameters) do _
        (; center = @ones(ni...), vertex = @ones(ni .+ 1...))
    end
    gradients = map(parameters) do _
        (; center = @zeros(ni...), vertex = @zeros(ni .+ 1...))
    end
    return NamedTuple{parameters}(multipliers), NamedTuple{parameters}(gradients)
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
