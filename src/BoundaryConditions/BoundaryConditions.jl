abstract type AbstractBoundaryCondition end
abstract type Dirichlet <: AbstractBoundaryCondition end
abstract type FreeSlip <: AbstractBoundaryCondition end
abstract type Horizontal <: FreeSlip end
abstract type Vertical <: FreeSlip end
abstract type Neumann <: AbstractBoundaryCondition end

# struct BoundaryCondition{T<:FreeSlip, nDim}
#     idx::NTuple{nDim, AbstractVector{Int64}}
# end

# struct BoundaryCondition{Dirichlet, nDim}
#     idx::NTuple{nDim, AbstractVector{Int64}}
#     val::AbstractVector{Float64}
# end

# include("Dirichlet.jl")

Base.getindex(v::AbstractArray,  I::NTuple{N, Vector{Int64}}) where N = v[CartesianIndex.(I...)]

function Ωboundaries(A::AbstractArray{T, 2}) where T
    # unpack
    nyy, nxx = size(A)

    Ω_idx = Dict{Symbol, NTuple{2, Union{Vector{Int64}, Int64}}}()
    # boundaries
    Ω_idx[:W] = (
        ones(Int64, nyy-2),
        collect(2:nyy-1)
    )
    Ω_idx[:E] = (
        ones(Int64,nyy-2)*nyy,
        collect(2:nyy-1)
    )
    Ω_idx[:N] = (
        collect(2:nxx-1),
        ones(Int, nxx-2)
    )
    Ω_idx[:S] = (
        collect(2:nxx-1),
        ones(Int, nxx-2)*nxx
    )
    # corners
    Ω_idx[:SW] = (
        1,
        1
    )
    Ω_idx[:SE] = (
        1,
        nxx
    )
    Ω_idx[:NW] = (
        nyy,
        1
    )
    Ω_idx[:NE] = (
        nyy,
        nxx
    )

    Ω_idx[:halo] = (
        reduce(vcat, direction[1] for (_, direction) in Ω_idx),
        reduce(vcat, direction[2] for (_, direction) in Ω_idx)
    )

    return Ω_idx
end

# Ωx = Ωboundaries(stokes.V.Vx) # Dictionary containing the indices (i, j) for the boundaries of the domain
# ΩD_vx = dirichlet(Ωx[:halo], stokes.V.Vx[Ωx[:halo]])
# Ωy = Ωboundaries(stokes.V.Vy) # Dictionary containing the indices (i, j) for the boundaries of the domain
# ΩD_vy = dirichlet(Ωy[:halo], stokes.V.Vy[Ωy[:halo]])

function pureshear_bc!(stokes::StokesArrays, geometry::Geometry, εbg)
    # unpack
    (; Vx, Vy) = stokes.V
    dx, dy = geometry.di 
    lx, ly = geometry.li 
    # Velocity boundary conditions
    stokes.V.Vx .= PTArray( -εbg.*[((ix-1)*dx -0.5*lx) for ix=1:size(Vx,1), iy=1:size(Vx,2)] )
    stokes.V.Vy .= PTArray(  εbg.*[((iy-1)*dy -0.5*ly) for ix=1:size(Vy,1), iy=1:size(Vy,2)] )
end
