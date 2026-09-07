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
    PA = @zeros(ni...)
    P0 = @zeros(ni...)
    V = Velocity(ni...)
    VA = Velocity(ni...)
    ∇V = @zeros(ni...)
    τ = SymmetricTensor(ni...)
    dτ = SymmetricTensor(ni...)
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
        PA,
        P0,
        V,
        VA,
        ∇V,
        τ,
        dτ,
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
