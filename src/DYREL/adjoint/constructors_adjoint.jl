"""
    material_controls(CPUBackend, ni, parameters; nphases = nothing)

Allocate gradient fields for the requested material parameters.

`parameters` is a tuple of material parameter names, as in

    (:ρ0, :G)

Each requested name represents one upstream parameter. If that name occurs in several
material functions, all contributions are accumulated into the same gradient.

Returns a named tuple keyed by the selected parameter names. Every entry has the same
pointwise, phase-wise layout:

    gradients[name].center[p, I...]  # size (nphases, ni...)
    gradients[name].vertex[p, I...]  # size (nphases, (ni .+ 1)...)

The center and vertex contributions remain separate during sensitivity evaluation. After
all paths have been accumulated, the transpose of the center-to-vertex interpolation is
added to `center`; `vertex` retains the uncombined contribution for diagnostics.
"""
function material_controls(
        ::Type{CPUBackend}, ni::NTuple{N, Integer}, names::NTuple{M, Symbol};
        nphases = nothing,
    ) where {N, M}
    if !isempty(names) && !(nphases isa Integer && nphases > 0)
        throw(ArgumentError("a positive nphases is required for material gradients"))
    end
    allunique(names) || throw(ArgumentError("gradient parameter names must be unique"))

    # All material parameters use this layout, independent of which local material
    # function produces their sensitivity.
    gradient_fields = map(names) do _
        (;
            center = @zeros(nphases, ni...),
            vertex = @zeros(nphases, (ni .+ 1)...),
        )
    end
    return NamedTuple{names}(gradient_fields)
end

function AdjointStokesArrays(::Type{CPUBackend}, ni::Vararg{Integer, N}) where {N}
    return AdjointStokesArrays(tuple(ni...))
end

function AdjointStokesArrays(::Type{CPUBackend}, ni::NTuple{N, Integer}) where {N}
    return AdjointStokesArrays(ni)
end

function AdjointStokesArrays(
        ::Type{CPUBackend}, ni::NTuple{N, Integer}, bcs::AbstractFlowBoundaryConditions
    ) where {N}
    return AdjointStokesArrays(ni, periodic_dims(bcs))
end

function AdjointStokesArrays(
        ::Type{CPUBackend}, ni::NTuple{N, Integer}, periodic::NTuple{N, Bool}
    ) where {N}
    return AdjointStokesArrays(ni, periodic)
end

AdjointStokesArrays(ni::Vararg{Integer, N}) where {N} = AdjointStokesArrays(tuple(ni...))
AdjointStokesArrays(ni::NTuple{N, Integer}) where {N} =
    AdjointStokesArrays(ni, ntuple(_ -> false, Val(N)))

function AdjointStokesArrays(
        ni::NTuple{N, Integer}, periodic::NTuple{N, Bool}
    ) where {N}
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
    R = Residual(ni, periodic)
    U = Displacement(ni...)
    ω = Vorticity(ni...)
    η = @zeros(ni...)
    ρ = @zeros(ni...)
    dρgx = @zeros(ni...)

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
        dρgx,
    )
end

function AdjointStokesArrays(::Number, ::Number)
    throw(ArgumentError("AdjointStokesArrays dimensions must be given as integers"))
end

function AdjointStokesArrays(::Number, ::Number, ::Number)
    throw(ArgumentError("AdjointStokesArrays dimensions must be given as integers"))
end
