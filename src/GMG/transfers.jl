# Transfer weights are built once on the host from explicit global coordinates,
# then copied to the level backend. Kernels in `transfer_kernels.jl` apply them.
#
# Prolongation is a tensor product of one-dimensional rows. Restriction is the
# measure-weighted transpose R = M_c⁻¹ Pᵀ M_f, built from the same rows, so the
# adjoint identity holds by construction on retained, owned DOFs.

"""Pressure correction copied from the parent cell."""
struct GMGInjection end
"""Coordinate-linear interpolation between coarse cell centers (smooth prototype)."""
struct GMGLinearCells end
"""Coincident-face injection and coordinate-linear interpolation between coarse faces."""
struct GMGLinearFaces end

"""Velocity transfer whose transverse rule copies the parent value (flux-conserving reference)."""
struct GMGConservativeVelocity end
"""Prototype velocity transfer with transverse linear interpolation (not flux-conserving)."""
struct GMGSmoothVelocity end

@inline gmg_transverse_rule(::GMGConservativeVelocity) = GMGInjection()
@inline gmg_transverse_rule(::GMGSmoothVelocity) = GMGLinearCells()

"""One direction and storage location: padded index/weight rows for both operators."""
struct GMGTable{I, W}
    prolong_index::I
    prolong_weight::W
    restrict_index::I
    restrict_weight::W
end

"""
Transfer between two adjacent levels. Per direction, `pressure` acts on cell
storage, `tangential` on ghosted cell storage, and `normal` on face storage.
"""
struct GMGTransfer{N, Tb}
    pressure::NTuple{N, Tb}
    tangential::NTuple{N, Tb}
    normal::NTuple{N, Tb}
end

@inline gmg_center(x::Vector, g::Int, periodic::Bool) =
    (gmg_vertex(x, g, periodic) + gmg_vertex(x, g + 1, periodic)) / 2

@inline gmg_wrap(g::Int, n::Int, periodic::Bool) = periodic ? mod1(g, n) : g

# A nonperiodic boundary face carries a prescribed (zero) normal correction.
@inline gmg_retained_face(g::Int, n::Int, periodic::Bool) = periodic || 1 < g <= n

# One-dimensional prolongation rows in global indices. Each returns two
# (coarse index, weight) pairs; index 0 marks an unused entry.
function gmg_row(::GMGInjection, xf, xc, g, r, periodic, no_slip)
    T = eltype(xf)
    return ((fld(g - 1, r) + 1, one(T)), (0, zero(T)))
end

function gmg_row(::GMGLinearCells, xf::Vector{T}, xc, g, r, periodic, no_slip) where {T}
    r == 1 && return ((g, one(T)), (0, zero(T)))
    nc = length(xc) - 1
    parent = fld(g - 1, r) + 1
    x, xp = gmg_center(xf, g, periodic), gmg_center(xc, parent, periodic)
    neighbor = x < xp ? parent - 1 : parent + 1
    if periodic || 1 <= neighbor <= nc
        xq = gmg_center(xc, neighbor, periodic)
        w = (xq - x) / (xq - xp)
        return ((gmg_wrap(parent, nc, periodic), w), (gmg_wrap(neighbor, nc, periodic), one(T) - w))
    end
    # Homogeneous tangential correction: free slip extends the parent value to
    # the wall, no slip interpolates linearly to a zero wall value.
    lower = neighbor < 1
    wall = lower ? first(xc) : last(xc)
    α = (lower ? no_slip[1] : no_slip[2]) ? zero(T) : one(T)
    w = α + (one(T) - α) * (x - wall) / (xp - wall)
    return ((parent, w), (0, zero(T)))
end

function gmg_row(::GMGLinearFaces, xf::Vector{T}, xc, g, r, periodic, no_slip) where {T}
    nf, nc = length(xf) - 1, length(xc) - 1
    gmg_retained_face(g, nf, periodic) || return ((0, zero(T)), (0, zero(T)))
    keep(G, w) = gmg_retained_face(G, nc, periodic) ? (gmg_wrap(G, nc, periodic), w) : (0, zero(T))
    if r == 1 || isodd(g)
        return (keep(fld(g - 1, r) + 1, one(T)), (0, zero(T)))
    end
    G = g ÷ 2
    x, x0, x1 = gmg_vertex(xf, g, periodic), gmg_vertex(xc, G, periodic), gmg_vertex(xc, G + 1, periodic)
    w = (x1 - x) / (x1 - x0)
    return (keep(G, w), keep(G + 1, one(T) - w))
end

@inline function gmg_measure(::GMGLinearFaces, x::Vector{T}, g, periodic) where {T}
    left = !periodic && g == 1 ? zero(T) : gmg_width(x, g - 1, periodic)
    right = !periodic && g == length(x) ? zero(T) : gmg_width(x, g, periodic)
    return (left + right) / 2
end
@inline gmg_measure(rule, x::Vector, g, periodic) = gmg_width(x, g, periodic)

# Global DOF range per location; faces include the nonperiodic terminal vertex.
@inline gmg_global_range(::GMGLinearFaces, n, periodic) = 1:(n + !periodic)
@inline gmg_global_range(rule, n, periodic) = 1:n

@inline function gmg_owned_global(::GMGLinearFaces, partition, d)
    owned = partition.owned[d]
    terminal = !partition.periodic[d] && last(owned) == partition.global_ni[d]
    return first(owned):(last(owned) + terminal)
end
@inline gmg_owned_global(rule, partition, d) = partition.owned[d]

# Storage positions: cells and faces start at the first stored cell; ghosted
# velocity storage adds one transverse ghost in front.
@inline gmg_storage_length(::GMGLinearFaces, partition, d, ghosted) = length(partition.storage[d]) + 1
@inline gmg_storage_length(rule, partition, d, ghosted) = length(partition.storage[d]) + 2ghosted

@inline gmg_storage_global(partition, d, s, ghosted) = first(partition.storage[d]) + s - 1 - ghosted

# Prefer the owned representative of a periodic index, then any stored copy.
function gmg_storage_position(rule, partition, d, g, ghosted)
    n, periodic = partition.global_ni[d], partition.periodic[d]
    owned = gmg_owned_global(rule, partition, d)
    candidates = periodic ? (mod1(g, n), g, mod1(g, n) - n, mod1(g, n) + n) : (g,)
    for c in candidates
        c in owned && return c - first(partition.storage[d]) + 1 + ghosted
    end
    len = gmg_storage_length(rule, partition, d, ghosted)
    for c in candidates
        s = c - first(partition.storage[d]) + 1 + ghosted
        (1 + ghosted) <= s <= len - ghosted && return s
    end
    throw(ArgumentError("GMG transfer support for global index $g in direction $d is not stored on this rank"))
end

function gmg_table_host(rule, fine::GMGLevel, coarse::GMGLevel, d, r, ghosted, no_slip)
    xf, xc = fine.domain.xvi[d], coarse.domain.xvi[d]
    fpart, cpart = fine.partition, coarse.partition
    periodic = fpart.periodic[d]
    T = eltype(xf)
    nf = fine.domain.ni[d]
    row(g) = gmg_row(rule, xf, xc, g, r, periodic, no_slip)

    # Prolongation: one row per owned fine storage position.
    nfs = gmg_storage_length(rule, fpart, d, ghosted)
    pidx, pw = ones(Int32, nfs, 2), zeros(T, nfs, 2)
    fowned = gmg_owned_global(rule, fpart, d)
    for s in 1:nfs
        g = gmg_storage_global(fpart, d, s, ghosted)
        g in fowned || continue
        for (k, (G, w)) in enumerate(row(g))
            iszero(G) && continue
            pidx[s, k] = gmg_storage_position(rule, cpart, d, G, ghosted)
            pw[s, k] = w
        end
    end

    # Restriction: gather every fine row that touches an owned coarse DOF.
    ncs = gmg_storage_length(rule, cpart, d, ghosted)
    cowned = gmg_owned_global(rule, cpart, d)
    entries = [Tuple{Int32, T}[] for _ in 1:ncs]
    fine_range = gmg_global_range(rule, nf, periodic)
    for s in 1:ncs
        G = gmg_storage_global(cpart, d, s, ghosted)
        G in cowned || continue
        mc = gmg_measure(rule, xc, G, periodic)
        for g0 in (r * (G - 1) - 2r):(r * G + 2r)
            g = periodic ? mod1(g0, nf) : g0
            (g0 == g || periodic) && g in fine_range || continue
            any(e -> e[1] == g, entries[s]) && continue
            for (Gr, w) in row(g)
                Gr == G && !iszero(w) || continue
                push!(entries[s], (Int32(g), w * gmg_measure(rule, xf, g, periodic) / mc))
            end
        end
    end
    width = max(1, maximum(length, entries))
    ridx, rw = ones(Int32, ncs, width), zeros(T, ncs, width)
    for s in 1:ncs, (k, (g, w)) in enumerate(entries[s])
        ridx[s, k] = gmg_storage_position(rule, fpart, d, Int(g), ghosted)
        rw[s, k] = w
    end
    return pidx, pw, ridx, rw
end

function gmg_device_table(template, (pidx, pw, ridx, rw))
    device(x) = copyto!(similar(template, eltype(x), size(x)), x)
    return GMGTable(device(pidx), device(pw), device(ridx), device(rw))
end

"""
    GMGTransfer(fine, coarse, transition; velocity = GMGConservativeVelocity(), no_slip)

Setup-time transfer weights between adjacent levels. `no_slip[d]` gives the
homogeneous tangential correction condition on the lower/upper physical
boundary in direction `d`; `false` is free slip. Normal velocity corrections
on nonperiodic boundaries are eliminated (prescribed zero). Every fine and
coarse read must be stored on this rank; coarse halo transport is not part of
this unit, so distributed use requires a prior level exchange.
"""
function GMGTransfer(
        fine::GMGLevel, coarse::GMGLevel, transition::GMGTransition{N};
        velocity = GMGConservativeVelocity(),
        no_slip::NTuple{N, NTuple{2, Bool}} = ntuple(_ -> (false, false), Val(N))
    ) where {N}
    template = fine.correction.p
    table(rule, d, ghosted) = gmg_device_table(
        template, gmg_table_host(rule, fine, coarse, d, transition.ratio[d], ghosted, no_slip[d])
    )
    pressure = ntuple(d -> table(GMGInjection(), d, false), Val(N))
    tangential = ntuple(d -> table(gmg_transverse_rule(velocity), d, true), Val(N))
    normal = ntuple(d -> table(GMGLinearFaces(), d, false), Val(N))
    return GMGTransfer(pressure, tangential, normal)
end

"""Transfers for every adjacent level pair of `hierarchy`, finest first."""
function gmg_transfers(hierarchy::GMGHierarchy; kwargs...)
    levels = hierarchy.levels
    return [GMGTransfer(levels[l], levels[l + 1], hierarchy.transitions[l]; kwargs...) for l in eachindex(hierarchy.transitions)]
end

@inline gmg_component_tables(transfer::GMGTransfer{N}, ::Val{C}) where {N, C} =
    ntuple(d -> d == C ? transfer.normal[d] : transfer.tangential[d], Val(N))
