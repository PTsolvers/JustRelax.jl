function dirichlet(boundary::NTuple{N, Symbol}, val::NTuple{N, Real}, geometry::Geometry) where N
    Ω = Ωboundaries(geometry)
    idxs = [Ω[boundary_i] for boundary_i in boundary]
    bc_idx = (
        reduce(vcat, idx[1] for idx in idxs),
        reduce(vcat, idx[2] for idx in idxs)
    )
    bc_val = reduce(vcat, fill(eltype(PTArray)(val[i]), length(bc_idx[i])) for i in 1:N)
    return BoundaryCondition{Dirichlet, N}(
        bc_idx,
        bc_val
    )
end

function dirichlet(boundary::NTuple{N, Symbol}, val::NTuple{N, AbstractVector}, geometry::Geometry) where N
    Ω = Ωboundaries(geometry)
    idxs = [Ω[boundary_i] for boundary_i in boundary]
    bc_idx = (
        reduce(vcat, idx[1] for idx in idxs),
        reduce(vcat, idx[2] for idx in idxs)
    )
    return BoundaryCondition{Dirichlet, N}(
        bc_idx,
        vcat(val...)
    )
end

function dirichlet(boundary::Symbol, val::Union{Real, AbstractVector}, geometry::Geometry{N}) where N
    Ω = Ωboundaries(geometry)
    return dirichlet(boundary, val, Ω, N)
end

function dirichlet(boundary::Symbol, val::Real, Ω::Dict, N)
    BoundaryCondition{Dirichlet, N}(
        Ω[boundary],
        fill(val, length(Ω[boundary]) )
    )
end

function dirichlet(boundary::Symbol, val::AbstractVector, Ω::Dict, N)
    BoundaryCondition{Dirichlet, N}(
        Ω[boundary],
        val
    )
end

dirichlet(Ω::NTuple{nDim, AbstractVector{Int64}}, val::AbstractVector{Float64}) where nDim = 
    BoundaryCondition{Dirichlet, nDim}(
            Ω,
            val
        )