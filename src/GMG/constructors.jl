function GMGOptions(
        ::Val{N}; coarsening = ntuple(_ -> 2, Val(N)),
        coarsest_cells = 16, max_levels::Integer = 32
    ) where {N}
    N in (2, 3) || throw(ArgumentError("GMG supports Cartesian 2D/3D grids"))
    coarsening isa Tuple && length(coarsening) == N || throw(ArgumentError("coarsening must be a tuple with $N directions"))
    all(r -> r isa Integer && r in (1, 2), coarsening) ||
        throw(ArgumentError("coarsening factors must be 1 or 2"))
    target = coarsest_cells isa Integer ? ntuple(_ -> Int(coarsest_cells), Val(N)) : coarsest_cells
    target isa Tuple && length(target) == N || throw(ArgumentError("coarsest_cells must be an integer or a tuple with $N directions"))
    all(n -> n isa Integer && n >= 2, target) ||
        throw(ArgumentError("coarsest_cells must be at least 2 in each direction"))
    max_levels >= 1 || throw(ArgumentError("max_levels must include at least the fine level"))
    return GMGOptions{N}(Int.(coarsening), Int.(target), Int(max_levels))
end

function GMGPartition(
        global_ni::NTuple{N, Int}, owned::NTuple{N, UnitRange{Int}};
        periodic::NTuple{N, Bool} = ntuple(_ -> false, Val(N)),
        halo::NTuple{N, Int} = ntuple(_ -> 1, Val(N)), comm = nothing
    ) where {N}
    valid = N in (2, 3) && all(n -> n >= 2, global_ni) && all(h -> h >= 0, halo)
    valid &= all(d -> length(owned[d]) >= 2 && 1 <= first(owned[d]) <= last(owned[d]) <= global_ni[d], 1:N)
    gmg_require(valid, "invalid GMG domain, owned cell ranges, or halo widths", comm)
    storage = ntuple(Val(N)) do d
        lo, hi = first(owned[d]) - halo[d], last(owned[d]) + halo[d]
        return periodic[d] ? (lo:hi) : (max(1, lo):min(global_ni[d], hi))
    end
    owned_local = ntuple(Val(N)) do d
        lo = first(owned[d]) - first(storage[d]) + 1
        return lo:(lo + length(owned[d]) - 1)
    end
    return GMGPartition(global_ni, owned, storage, owned_local, periodic, halo, comm)
end

function gmg_domain(xvi::NTuple{N, Vector{T}}, comm) where {N, T <: AbstractFloat}
    valid = N in (2, 3) && T in (Float32, Float64) && all(x -> length(x) >= 3 && all(isfinite, x) && all(isfinite, diff(x)) && all(>(zero(T)), diff(x)) && isfinite(last(x) - first(x)), xvi)
    gmg_require(valid, "GMG vertices must use Float32/Float64, be finite and strictly increasing, and define at least two cells", comm)
    # Setup owns its coordinates; later changes to caller vectors cannot alter nesting.
    vertices = map(copy, xvi)
    ni = map(x -> length(x) - 1, vertices)
    li = map(x -> last(x) - first(x), vertices)
    origin = map(first, vertices)
    return GMGDomain(ni, vertices, li, origin)
end

function GMGMaterialContext(rheology::GeoParams.MaterialParams, args::NamedTuple)
    return GMGMaterialContext(rheology, args, nothing)
end

function GMGMaterialContext(rheology::Tuple, args::NamedTuple, phase_ratios::JustPIC.PhaseRatios)
    !isempty(rheology) && all(r -> r isa GeoParams.MaterialParams, rheology) ||
        throw(ArgumentError("multiphase GMG rheology must be a nonempty tuple of GeoParams.MaterialParams"))
    length(rheology) == JustPIC.numphases(phase_ratios) ||
        throw(ArgumentError("GMG material count must match the phase-ratio container"))
    return GMGMaterialContext{typeof(rheology), typeof(args), typeof(phase_ratios)}(rheology, args, phase_ratios)
end

function gmg_fields(template::AbstractArray{T, N}, ni::NTuple{N, Int}) where {T, N}
    p = fill!(similar(template, T, ni), zero(T))
    v = ntuple(Val(N)) do component
        shape = ntuple(d -> ni[d] + (d == component ? 1 : 2), Val(N))
        return fill!(similar(template, T, shape), zero(T))
    end
    return GMGCoupledFields(p, v)
end

"""
    GMGHierarchy(backend, global_vertices, partition; options)

Internal Stage 1 setup from explicit global host coordinates and owned pressure
cells. Coordinates and numerical fields use `PTArray(backend)` and the coordinate
scalar type. It never reads or changes IGG globals. All ranks remain active and
coarsening stops before alignment or minimum-local-cell constraints fail.
This constructor does not implement transfer kernels, coarse halos, or a solver.
"""
function GMGHierarchy(
        backend_type::Type{<:AbstractBackend}, xvi::NTuple{N, Vector{T}},
        partition::GMGPartition{N}; options::GMGOptions{N} = GMGOptions(Val(N))
    ) where {N, T <: AbstractFloat}
    domain = gmg_domain(xvi, partition.comm)
    gmg_require(domain.ni == partition.global_ni, "GMG vertices and partition cell counts disagree", partition.comm)
    gmg_collective_setup(domain, partition, options)
    TA = PTArray(backend_type)
    template = TA(zeros(T, ntuple(_ -> 0, Val(N))))
    return gmg_hierarchy(template, TA, domain, partition, options)
end
