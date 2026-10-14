"""
    DistributedMarkerChain(backend, xvi_global, igg; initial_elevation=0.0,
                           nxcell=100, min_xcell=75, max_xcell=125)

Create a distributed 2D free surface from global vertex-coordinate vectors.
`backend` is a JustPIC backend type, such as `JustPIC.CPU`. Every rank must call
this constructor with the same grid and marker-count settings. Rank zero supplies
the initial elevation, either a scalar or one height per global horizontal vertex.
The grid must use nonperiodic IGG boundaries and the default two-cell overlap.

`surface.chain` is the backend-local `JustPIC.MarkerChain` for phase correction
and rock-fraction calculations. `surface.cell_vertices` and `surface.h_vertices`
are replicated global CPU vectors, usable with `save_marker_chain`. Grid metadata,
local vertex indices and the communicator are retained internally.

Advance with `advect_markerchain!(surface, stokes, dt)` on every rank. Only local
marker arrays are allocated; the global surface stores coordinate and height vectors.
"""
struct DistributedMarkerChain{C, G, I, V}
    chain::C
    grid::G
    igg::I
    cell_vertices::V
    h_vertices::V
    local_vertices::UnitRange{Int}
end

function DistributedMarkerChain(
        backend, xvi_global::NTuple{2, AbstractVector}, igg::IGG;
        initial_elevation = 0.0, nxcell = 100, min_xcell = 75, max_xcell = 125,
    )
    local_grid = Geometry(Array, xvi_global, igg)
    global_grid = Geometry(Array, Array.(xvi_global)...)
    vertices = global_grid.xvi[1]
    first_local = searchsortedfirst(vertices, local_grid.origin[1])
    indices = first_local:(first_local + local_grid.ni[1])
    valid_counts = 0 < nxcell <= max_xcell && 0 < min_xcell <= max_xcell
    MPI.Allreduce(valid_counts, &, igg.comm_cart) || throw(ArgumentError("invalid marker counts on one or more ranks"))
    heights = _root_value(igg) do
        T = eltype(vertices)
        h = initial_elevation isa AbstractVector ? T.(Array(initial_elevation)) : fill(T(initial_elevation), length(vertices))
        length(h) == length(vertices) && all(isfinite, h) ||
            throw(ArgumentError("initial elevation must supply one finite height per global vertex"))
        h
    end
    chain = JustPIC.init_markerchain(
        backend, nxcell, min_xcell, max_xcell,
        local_grid.xvi[1], heights[indices]
    )
    return DistributedMarkerChain(chain, global_grid, igg, vertices, heights, indices)
end

"""
    advect_markerchain!(surface::DistributedMarkerChain, stokes, dt)

Collectively advance a distributed 2D surface through the local Stokes velocity.
All ranks must supply the same finite `dt`. Uses semi-Lagrangian RK2 backtracking,
the serial 45-degree slope smoothing and weighted mean-height conservation.
Rank zero's old heights are authoritative. Interpolation queries are sampled by
the owning ranks and combined in small MPI reductions; velocity fields stay local.

Sampling errors propagate collectively. Failed characteristics leave the global
heights and local chain unchanged. Selected corners are copied to the host for
GPU fields; CUDA-aware MPI is not required by this surface update.
"""
function JustPIC.advect_markerchain!(surface::DistributedMarkerChain, stokes::StokesArrays, dt)
    return JustPIC.advect_markerchain!(backend(stokes), surface, stokes, dt)
end

JustPIC.advect_markerchain!(::CPUBackendTrait, surface::DistributedMarkerChain, stokes, dt) =
    _advect_markerchain_mpi!(surface, stokes, dt)

# Each query/component has one owner, determined by its global interpolation cell.
# Only four corner values per owned query are copied from a GPU to the host.
function _sample_markerchain_velocity(V, grid_vxi, queries, active, ni, igg)
    T = eltype(first(V))
    n = length(queries)
    buffer = zeros(T, n + 1, 2) # final row carries a collective failure flag
    failure = nothing
    try
        for component in 1:2
            A, xi = V[component], grid_vxi[component]
            ids, indices = Int[], Int[]
            weights = NTuple{2, T}[]
            linear = LinearIndices(A)
            for q in eachindex(queries)
                active[q] || continue
                p = queries[q]
                if !all(isfinite, p)
                    buffer[q, component] = T(NaN)
                    continue
                end
                I = ntuple(d -> clamp(searchsortedlast(xi[d], p[d]), 1, length(xi[d]) - 1), 2)
                owner = ntuple(d -> min((I[d] - 1) ÷ (ni[d] - 2), igg.dims[d] - 1), 2)
                owner == Tuple(igg.coords[1:2]) || continue
                i, j = ntuple(d -> I[d] - owner[d] * (ni[d] - 2), 2)
                push!(ids, q)
                push!(weights, ntuple(d -> (p[d] - xi[d][I[d]]) * inv(xi[d][I[d] + 1] - xi[d][I[d]]), 2))
                push!(indices, linear[i, j], linear[i + 1, j], linear[i, j + 1], linear[i + 1, j + 1])
            end
            isempty(ids) && continue
            device_indices = similar(A, Int, length(indices))
            copyto!(device_indices, indices)
            corners = Array(A[device_indices])
            for (k, q) in enumerate(ids)
                values = ntuple(c -> corners[4 * (k - 1) + c], 4)
                buffer[q, component] = JustPIC.lerp(values, weights[k])
            end
        end
    catch err
        failure = sprint(showerror, err)
        buffer[end, 1] = 1
    end
    MPI.Allreduce!(buffer, +, igg.comm_cart)
    if buffer[end, 1] > 0
        failures = MPI.gather(failure, igg.comm_cart)
        message = igg.me == 0 ? join(filter(!isnothing, failures), "\n") : nothing
        error(MPI.bcast(message, 0, igg.comm_cart))
    end
    return buffer[1:n, :]
end

# Lockstep secant iterations let every rank batch its RK2 velocity queries.
# This is the same characteristic solve as JustPIC.backtrack_surface_height.
function _backtrack_markerchain(h_old, V, grid_global, dt, ni, igg)
    xv = grid_global.xvi[1]
    h, h_previous = copy(h_old), copy(h_old)
    residual_previous = zero(h_old)
    active = trues(length(h))
    method = RungeKutta2()
    for iteration in 1:20
        p0 = [(xv[i], h[i]) for i in eachindex(h)]
        v0 = _sample_markerchain_velocity(V, grid_global.xi_vel, p0, active, ni, igg)
        p1 = [JustPIC.first_stage(method, dt, (v0[i, 1], v0[i, 2]), p0[i]; backtracking = true) for i in eachindex(h)]
        v1 = _sample_markerchain_velocity(V, grid_global.xi_vel, p1, active, ni, igg)
        for i in eachindex(h)
            active[i] || continue
            xd, yd = JustPIC.second_stage(method, dt, (v0[i, 1], v0[i, 2]), (v1[i, 1], v1[i, 2]), p0[i]; backtracking = true)
            isfinite(xd) && isfinite(yd) || error("Marker-chain backtracking did not converge; reduce dt and check that the surface remains single-valued")
            residual = yd - JustPIC._interpolate_topography(xd, xv, h_old, i)
            if abs(residual) <= 2 * eps(eltype(h)) * max(eps(eltype(h)), abs(h[i]), abs(yd))
                active[i] = false
                continue
            end
            denominator = residual - residual_previous[i]
            correction = iteration == 1 || iszero(denominator) ? residual : residual * (h[i] - h_previous[i]) / denominator
            h_previous[i], residual_previous[i] = h[i], residual
            h[i] -= correction
        end
        any(active) || return h
    end
    error("Marker-chain backtracking did not converge; reduce dt and check that the surface remains single-valued")
end

function _advect_markerchain_mpi!(surface::DistributedMarkerChain, stokes, dt)
    (; chain, grid, igg, local_vertices) = surface
    ni = size(stokes.P)
    valid_shape = length(ni) == 2 && Tuple(igg.dims[1:2]) .* (ni .- 2) .+ 2 == grid.ni
    MPI.Allreduce(valid_shape, &, igg.comm_cart) ||
        throw(DimensionMismatch("Stokes arrays must match the distributed 2D surface grid on every rank"))
    # Synchronize the lightweight authoritative surface before issuing matching queries.
    state = [surface.h_vertices; dt]
    MPI.Bcast!(state, 0, igg.comm_cart)
    h_old = state[1:(end - 1)]
    valid = isfinite(dt) && dt == state[end] && all(isfinite, h_old)
    MPI.Allreduce(valid, &, igg.comm_cart) ||
        throw(ArgumentError("surface heights must be finite and every rank must supply the same finite dt"))
    h_new = _backtrack_markerchain(h_old, (stokes.V.Vx, stokes.V.Vy), grid, dt, size(stokes.P), igg)
    target_mean = JustPIC.mean_height(h_old, surface.cell_vertices)
    JustPIC.smooth_slopes!(h_new, surface.cell_vertices, deg2rad(45.0))
    h_new .-= JustPIC.mean_height(h_new, surface.cell_vertices) - target_mean
    copyto!(surface.h_vertices, h_new)
    copyto!(chain.h_vertices, view(surface.h_vertices, local_vertices))
    JustPIC.finish_markerchain_step!(chain, nothing)
    return nothing
end
