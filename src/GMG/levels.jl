@inline function gmg_vertex(x::Vector{T}, j::Int, periodic::Bool) where {T}
    if periodic
        n = length(x) - 1
        winding, index = fldmod(j - 1, n)
        return x[index + 1] + winding * (last(x) - first(x))
    end
    return x[j]
end

@inline function gmg_width(x::Vector{T}, j::Int, periodic::Bool) where {T}
    return gmg_vertex(x, j + 1, periodic) - gmg_vertex(x, j, periodic)
end

# Construct Geometry directly from explicit coordinates, preserving T rather
# than passing through the legacy uniform constructor or Float64 metadata path.
function gmg_geometry(::Type{TA}, xvi::NTuple{N, Vector{T}}) where {TA, N, T}
    ni = map(x -> length(x) - 1, xvi)
    xci = map(x -> x[1:(end - 1)] .+ diff(x) ./ 2, xvi)
    li = map(x -> last(x) - first(x), xvi)
    origin = map(first, xvi)
    di_vertex = diff.(xvi)
    di_center = diff.(xci)
    xi_vel = map(x -> TA.(x), velocity_grids(xci, xvi, di_center))
    di = (; center = TA.(di_center), vertex = TA.(di_vertex), velocity = map(x -> diff.(x), xi_vel))
    _di = (;
        center = map(x -> inv.(x), di.center),
        vertex = map(x -> inv.(x), di.vertex),
        velocity = map(x -> map(y -> inv.(y), x), di.velocity),
    )
    vertices, centers = TA.(xvi), TA.(xci)
    return Geometry{N, typeof(vertices[1]), typeof(di), T}(ni, li, origin, maximum(li), di, _di, centers, vertices, xi_vel)
end

function gmg_measures(::Type{TA}, domain::GMGDomain{N, T}, partition::GMGPartition{N}) where {TA, N, T}
    cell = ntuple(Val(N)) do d
        x, periodic = domain.xvi[d], partition.periodic[d]
        return TA([gmg_width(x, j, periodic) for j in partition.storage[d]])
    end
    face = ntuple(Val(N)) do d
        x, periodic = domain.xvi[d], partition.periodic[d]
        vertices = first(partition.storage[d]):(last(partition.storage[d]) + 1)
        widths = map(vertices) do j
            left = !periodic && j == 1 ? zero(T) : gmg_width(x, j - 1, periodic)
            right = !periodic && j == domain.ni[d] + 1 ? zero(T) : gmg_width(x, j, periodic)
            return (left + right) / 2
        end
        return TA(widths)
    end
    return GMGMeasures(cell, face)
end

function gmg_level(template, ::Type{TA}, domain::GMGDomain{N}, partition::GMGPartition{N}) where {TA, N}
    xvi = ntuple(Val(N)) do d
        vertices = first(partition.storage[d]):(last(partition.storage[d]) + 1)
        return [gmg_vertex(domain.xvi[d], j, partition.periodic[d]) for j in vertices]
    end
    grid = gmg_geometry(TA, xvi)
    measures = gmg_measures(TA, domain, partition)
    return GMGLevel(
        grid, domain, partition, measures,
        gmg_fields(template, grid.ni), gmg_fields(template, grid.ni), gmg_fields(template, grid.ni),
    )
end

function gmg_coarsening(domain::GMGDomain{N}, partition::GMGPartition{N}, options::GMGOptions{N}) where {N}
    return ntuple(Val(N)) do d
        n, owned = domain.ni[d], partition.owned[d]
        legal = options.coarsening[d] == 2 && iseven(n) && n ÷ 2 >= options.coarsest_cells[d]
        legal &= iseven(first(owned) - 1) && iseven(length(owned)) && length(owned) ÷ 2 >= 2
        return gmg_all(legal, partition.comm) ? 2 : 1
    end
end

function gmg_coarsen(domain::GMGDomain{N}, partition::GMGPartition{N}, ratio::NTuple{N, Int}) where {N}
    xvi = ntuple(d -> domain.xvi[d][firstindex(domain.xvi[d]):ratio[d]:lastindex(domain.xvi[d])], Val(N))
    coarse_domain = gmg_domain(xvi, partition.comm)
    owned = ntuple(d -> ((first(partition.owned[d]) - 1) ÷ ratio[d] + 1):(last(partition.owned[d]) ÷ ratio[d]), Val(N))
    coarse_partition = GMGPartition(
        coarse_domain.ni, owned; periodic = partition.periodic, halo = partition.halo, comm = partition.comm,
    )
    return coarse_domain, coarse_partition
end

function gmg_hierarchy(template, ::Type{TA}, domain::GMGDomain{N}, partition::GMGPartition{N}, options) where {TA, N}
    fine = gmg_level(template, TA, domain, partition)
    levels = [fine]
    transitions = GMGTransition{N}[]
    while length(levels) < options.max_levels
        ratio = gmg_coarsening(domain, partition, options)
        all(==(1), ratio) && break
        domain, partition = gmg_coarsen(domain, partition, ratio)
        push!(transitions, GMGTransition(ratio))
        push!(levels, gmg_level(template, TA, domain, partition))
    end
    return GMGHierarchy(levels, transitions, options)
end

@inline gmg_parent(transition::GMGTransition{N}, index::NTuple{N, Int}) where {N} =
    ntuple(d -> fld(index[d] - 1, transition.ratio[d]) + 1, Val(N))

"""Local unique owned face indices; transverse ghosts and periodic duplicates are excluded."""
function gmg_owned_velocity(partition::GMGPartition{N}, ::Val{C}) where {N, C}
    1 <= C <= N || throw(ArgumentError("velocity component must be between 1 and $N"))
    return ntuple(Val(N)) do d
        cells = partition.owned_local[d]
        d == C || return (first(cells) + 1):(last(cells) + 1)
        terminal = !partition.periodic[d] && last(partition.owned[d]) == partition.global_ni[d]
        return first(cells):(last(cells) + Int(terminal))
    end
end
