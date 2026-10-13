# Level halo exchange for GMG fields. Directions are exchanged in order and each
# slab spans the full storage of the other directions, so edge and corner halos
# are forwarded through the earlier stages. Buffers are allocated at setup;
# device data is staged through host buffers for MPI.

"""Halo plan for one storage location and direction: lower and upper sides."""
struct GMGHaloDirection{B, H}
    neighbor::NTuple{2, Int}
    send::NTuple{2, UnitRange{Int}}
    recv::NTuple{2, UnitRange{Int}}
    send_buffer::NTuple{2, B}
    recv_buffer::NTuple{2, B}
    send_host::NTuple{2, H}
    recv_host::NTuple{2, H}
end

"""
    GMGExchange(level)

Persistent halo plan and buffers for the coupled fields of one level. Neighbors
are found from the explicit owned ranges of every rank on the level
communicator; the decomposition must be Cartesian. A periodic direction with
one rank exchanges with itself.
"""
struct GMGExchange{N, D, C}
    pressure::NTuple{N, D}
    velocity::NTuple{N, NTuple{N, D}}
    comm::C
end

gmg_all_owned(partition::GMGPartition{N, Nothing}) where {N} = ([partition.owned], 0)

function gmg_all_owned(partition::GMGPartition{N, MPI.Comm}) where {N}
    comm = partition.comm
    bounds = MPI.Allgather(Int[first.(partition.owned)..., last.(partition.owned)...], comm)
    nranks = MPI.Comm_size(comm)
    table = reshape(bounds, 2N, nranks)
    owned = [ntuple(d -> table[d, r]:table[N + d, r], Val(N)) for r in 1:nranks]
    return owned, MPI.Comm_rank(comm)
end

# Rank (zero-based) owning the adjacent block along `d`, or -1 at a physical boundary.
function gmg_neighbor(all_owned, mine::NTuple{N}, d, side, n, periodic) where {N}
    target = side == 1 ? first(mine[d]) - 1 : last(mine[d]) + 1
    if !(1 <= target <= n)
        periodic || return -1
        target = mod1(target, n)
    end
    for (r, owned) in enumerate(all_owned)
        all(k -> k == d || owned[k] == mine[k], 1:N) && target in owned[d] && return r - 1
    end
    return -2
end

@inline gmg_valid_positions(rule, partition, d, ghosted) =
    (1 + ghosted):(gmg_storage_length(rule, partition, d, ghosted) - ghosted)

@inline function gmg_owned_positions(rule, partition, d, ghosted)
    owned = gmg_owned_global(rule, partition, d)
    offset = 1 + ghosted - first(partition.storage[d])
    return (first(owned) + offset):(last(owned) + offset)
end

# Halo positions on one side and their unwrapped global indices.
function gmg_halo_positions(rule, partition, d, side, ghosted)
    valid = gmg_valid_positions(rule, partition, d, ghosted)
    owned = gmg_owned_positions(rule, partition, d, ghosted)
    return side == 1 ? (first(valid):(first(owned) - 1)) : ((last(owned) + 1):last(valid))
end

function gmg_halo_direction(template, rule, partition::GMGPartition{N}, all_owned, d, ghosted, shape) where {N}
    n, periodic = partition.global_ni[d], partition.periodic[d]
    neighbors = ntuple(side -> gmg_neighbor(all_owned, partition.owned, d, side, n, periodic), Val(2))
    recv = ntuple(Val(2)) do side
        neighbors[side] < 0 && return 1:0
        return gmg_halo_positions(rule, partition, d, side, ghosted)
    end
    valid = all(>=(-1), neighbors)
    for side in 1:2
        neighbors[side] < 0 && (valid &= isempty(gmg_halo_positions(rule, partition, d, side, ghosted)); continue)
        other = GMGPartition(partition.global_ni, all_owned[neighbors[side] + 1]; periodic = partition.periodic, halo = partition.halo)
        owned = gmg_owned_global(rule, other, d)
        for s in recv[side]
            g = gmg_storage_global(partition, d, s, ghosted)
            valid &= gmg_wrap(g, n, periodic) in owned
        end
    end
    # What a neighbor receives from this rank, in its own (increasing) order.
    send = ntuple(Val(2)) do side
        neighbors[side] < 0 && return 1:0
        other = GMGPartition(partition.global_ni, all_owned[neighbors[side] + 1]; periodic = partition.periodic, halo = partition.halo)
        positions = map(gmg_halo_positions(rule, other, d, 3 - side, ghosted)) do s
            g = gmg_wrap(gmg_storage_global(other, d, s, ghosted), n, periodic)
            return gmg_storage_position(rule, partition, d, g, ghosted)
        end
        isempty(positions) && return 1:0
        return first(positions):last(positions)
    end
    gmg_require(valid, "GMG halo exchange needs a Cartesian decomposition whose halos are owned by direct neighbors", partition.comm)
    T = eltype(template)
    buffer(range) = fill!(similar(template, T, ntuple(k -> k == d ? length(range) : shape[k], Val(N))), zero(T))
    host(b) = b isa Array ? b : Array(b)
    send_buffer, recv_buffer = map(buffer, send), map(buffer, recv)
    return GMGHaloDirection(neighbors, send, recv, send_buffer, recv_buffer, map(host, send_buffer), map(host, recv_buffer))
end

GMGExchange(level::GMGLevel) = gmg_exchange(level, level.partition)

function gmg_exchange(level::GMGLevel, partition::GMGPartition{N}) where {N}
    all_owned, _ = gmg_all_owned(partition)
    template = level.correction.p
    pressure = ntuple(d -> gmg_halo_direction(template, GMGInjection(), partition, all_owned, d, false, size(level.correction.p)), Val(N))
    velocity = ntuple(Val(N)) do C
        ntuple(Val(N)) do d
            rule = d == C ? GMGLinearFaces() : GMGInjection()
            return gmg_halo_direction(template, rule, partition, all_owned, d, d != C, size(level.correction.v[C]))
        end
    end
    return GMGExchange(pressure, velocity, partition.comm)
end

@inline gmg_slab(A, d, range) = view(A, ntuple(k -> k == d ? range : axes(A, k), Val(ndims(A)))...)

function gmg_pack!(halo::GMGHaloDirection, A, d)
    for side in 1:2
        halo.neighbor[side] < 0 && continue
        halo.send_buffer[side] .= gmg_slab(A, d, halo.send[side])
        halo.send_host[side] === halo.send_buffer[side] || copyto!(halo.send_host[side], halo.send_buffer[side])
    end
    return nothing
end

function gmg_unpack!(halo::GMGHaloDirection, A, d)
    for side in 1:2
        halo.neighbor[side] < 0 && continue
        halo.recv_host[side] === halo.recv_buffer[side] || copyto!(halo.recv_buffer[side], halo.recv_host[side])
        gmg_slab(A, d, halo.recv[side]) .= halo.recv_buffer[side]
    end
    return nothing
end

# Serial: the only possible neighbor is this rank across a periodic seam.
# Data sent to the lower neighbor is what that neighbor's upper halo receives.
function gmg_exchange_direction!(halo::GMGHaloDirection, A, d, ::Nothing)
    gmg_pack!(halo, A, d)
    for side in 1:2
        halo.neighbor[side] < 0 || copyto!(halo.recv_host[side], halo.send_host[3 - side])
    end
    gmg_unpack!(halo, A, d)
    return nothing
end

function gmg_exchange_direction!(halo::GMGHaloDirection, A, d, comm::MPI.Comm)
    requests = MPI.MultiRequest(4)
    # Tag = side the sender targets: data arriving from below was sent upward.
    for side in 1:2
        halo.neighbor[side] < 0 && continue
        MPI.Irecv!(halo.recv_host[side], comm, requests[side]; source = halo.neighbor[side], tag = 3 - side)
    end
    gmg_pack!(halo, A, d)
    for side in 1:2
        halo.neighbor[side] < 0 && continue
        MPI.Isend(halo.send_host[side], comm, requests[2 + side]; dest = halo.neighbor[side], tag = side)
    end
    MPI.Waitall(requests)
    gmg_unpack!(halo, A, d)
    return nothing
end

"""
    exchange_level!(exchange, fields)

Fill the halos of `fields` (pressure and every velocity component) from the
owning ranks. Owned entries are never modified. Velocity ghost layers beyond
local storage are not physical-boundary values and stay the caller's
responsibility.
"""
function exchange_level!(exchange::GMGExchange{N}, fields::GMGCoupledFields{N}) where {N}
    for d in 1:N
        gmg_exchange_direction!(exchange.pressure[d], fields.p, d, exchange.comm)
        for C in 1:N
            gmg_exchange_direction!(exchange.velocity[C][d], fields.v[C], d, exchange.comm)
        end
    end
    return nothing
end
