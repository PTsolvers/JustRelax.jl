"""
    pack_velocity(velocity::Tuple, precision, slices = nothing)

Pack the velocity components into the `(3, size...)` array expected for a VTK
vector attribute. Readers copy three values per tuple regardless of the declared
`NumberOfComponents`, so a two-component array leaves the third component
undefined and ParaView orients its glyphs from uninitialized memory. Components
beyond `N` are written as zeros.

`slices` optionally restricts each component to a sub-range (used to trim the
ghost layer of MPI-distributed arrays).
"""
function pack_velocity(velocity::Tuple, precision, slices = nothing)
    length(velocity) ≤ 3 ||
        throw(ArgumentError("velocity must have at most 3 components, got $(length(velocity))"))
    for v in velocity
        axes(v) == axes(first(velocity)) ||
            throw(DimensionMismatch("velocity components must share their axes: $(axes(v)) vs $(axes(first(velocity)))"))
    end

    sz = isnothing(slices) ? size(first(velocity)) : length.(slices)
    velocity_field = zeros(precision, 3, sz...)
    for (i, v) in enumerate(velocity)
        vi = precision.(Array(v))
        selectdim(velocity_field, 1, i) .= isnothing(slices) ? vi : view(vi, slices...)
    end
    return velocity_field
end

"""
    add_field!(vtk, name, array, npoints, ncells, precision)

Write `array` to `vtk` as point data or cell data, whichever its size matches.
A vertex grid holds one cell fewer than it has nodes per dimension, so the two
sizes are always distinguishable.
"""
function add_field!(vtk, name, array, npoints, ncells, precision)
    data = precision.(Array(array))
    location = if size(data) == npoints
        VTKPointData()
    elseif size(data) == ncells
        VTKCellData()
    else
        throw(
            DimensionMismatch(
                "$name has size $(size(data)), which matches neither the $npoints vertices nor the $ncells cells of the grid"
            )
        )
    end
    vtk[string(name), location] = data
    return nothing
end

struct VTKDataSeries{T, S, G}
    series::T
    path::S
    name::S
    grid::G

    function VTKDataSeries(full_name::String, xi)
        split_path = splitpath(full_name)
        name = last(split_path)
        path = if length(split_path) > 1
            joinpath(split_path[1:(end - 1)])
        else
            pwd()
        end
        series = paraview_collection(full_name; append = true)
        return new{typeof(series), String, typeof(xi)}(series, path, name, xi)
    end
end

function append!(data_series, data::NamedTuple, time_step, seconds)
    # unpack data names and arrays
    data_names = string.(keys(data))
    data_arrays = values(data)
    # create vtk file
    vtk_name = joinpath(data_series.path, "$time_step")
    vtk = vtk_grid(vtk_name, data_series.grid...)
    # add data to vtk file
    for (name_i, array_i) in zip(data_names, data_arrays)
        vtk[name_i] = Array(array_i)
    end
    # close vtk file
    vtk_save(vtk)
    # open pvd file
    pvd_name = joinpath(data_series.path, data_series.name)
    pvd = paraview_collection(pvd_name; append = true)
    # add vtk file to time series
    collection_add_timestep(pvd, vtk, seconds)
    # close pvd file
    vtk_save(pvd)

    return nothing
end

"""
    save_vtk(fname::String, xvi, xci, data_v::NamedTuple, data_c::NamedTuple, velocity; t=0, pvd=nothing)

Save vertex and cell data to a single VTK file. The file holds the grid spanned
by the vertices `xvi`; `data_v` and `velocity` are written as point data and
`data_c` as cell data of that same grid.

## Arguments
- `fname::String`: The filename for the VTK file (without extension)
- `xvi`: Vertex coordinates (tuple of coordinate arrays)
- `xci`: Cell center coordinates (tuple of coordinate arrays); must have one entry fewer per dimension than `xvi`
- `data_v::NamedTuple`: Data defined at vertices
- `data_c::NamedTuple`: Data defined at cell centers. Fields of `data_v` and `data_c` are written as point or cell data according to their size, so a cell-centered field passed in `data_v` still lands on the cells
- `velocity::Tuple`: Velocity components, each an array defined at the vertices
- `t::Number`: Time value (default: 0)
- `pvd::Union{Nothing, String}`: Optional ParaView collection filename. If provided, the VTK file will be added to a time series collection. WriteVTK.jl automatically handles creating new collections or appending to existing ones.

## Examples
```julia
# Basic usage
save_vtk("output", xvi, xci, data_v, data_c, velocity; t=1.0)

# With ParaView collection for time series
save_vtk("timestep_001", xvi, xci, data_v, data_c, velocity; t=1.0, pvd="simulation")
save_vtk("timestep_002", xvi, xci, data_v, data_c, velocity; t=2.0, pvd="simulation")
# This creates simulation.pvd containing the time series

# Time series example
times = 0:0.1:10
for (i, t) in enumerate(times)
    fname = "timestep_\$(lpad(i, 3, '0'))"
    save_vtk(fname, xvi, xci, data_v, data_c, velocity; t=t, pvd="full_simulation")
end
```
"""
function save_vtk(
        fname::String,
        xvi,
        xci,
        data_v::NamedTuple,
        data_c::NamedTuple,
        velocity::Tuple;
        precision = Float32,
        t::Number = 0,
        pvd::Union{Nothing, String} = nothing,
    )

    length.(xvi) == length.(xci) .+ 1 || throw(
        DimensionMismatch(
            "the vertex grid must have one node more per dimension than the center grid: $(length.(xvi)) vs $(length.(xci))"
        )
    )

    size(first(velocity)) == length.(xvi) || throw(
        DimensionMismatch(
            "velocity must be given on the vertices: $(size(first(velocity))) vs $(length.(xvi))"
        )
    )

    velocity_field = pack_velocity(velocity, precision)

    # A grid spanned by the vertices carries the cell-centered fields as cell
    # data, so vertex and center fields share one file and one geometry.
    vtk_grid(fname, xvi...) do vtk
        for (name_i, array_i) in Iterators.flatten((pairs(data_v), pairs(data_c)))
            add_field!(vtk, name_i, array_i, length.(xvi), length.(xci), precision)
        end
        vtk["Velocity", VTKPointData()] = velocity_field
        isnothing(t) || (vtk["TimeValue"] = t)

        # If pvd collection name is provided, add this file to the collection
        if !isnothing(pvd)
            paraview_collection(pvd; append = true) do pvd_collection
                collection_add_timestep(pvd_collection, vtk, t)
            end
        end
    end

    return nothing
end

"""
    save_vtk(fname::String, xci, data_c::NamedTuple, velocity; t=nothing, pvd=nothing)

Save VTK data with cell-centered data and velocity field.

## Arguments
- `fname::String`: The filename for the VTK file (without extension)
- `xci`: Cell center coordinates (tuple of coordinate arrays)
- `data_c::NamedTuple`: Data defined at cell centers
- `velocity::Tuple`: Velocity components, each an array defined on the grid nodes
- `t::Number`: Time value (default: nothing)
- `pvd::Union{Nothing, String}`: Optional ParaView collection filename. If provided, the VTK file will be added to a time series collection. WriteVTK.jl automatically handles creating new collections or appending to existing ones.

## Examples
```julia
# Basic usage
save_vtk("output", xci, data_c, velocity; t=1.0)

# With ParaView collection
save_vtk("timestep_001", xci, data_c, velocity; t=1.0, pvd="simulation")
```
"""
function save_vtk(
        fname::String,
        xci,
        data_c::NamedTuple,
        velocity::Tuple;
        precision = Float32,
        t::Union{Number, Nothing} = nothing,
        pvd::Union{Nothing, String} = nothing
    )

    # A velocity that does not match the grid is demoted to field data by
    # WriteVTK, i.e. silently written as something ParaView cannot plot.
    size(first(velocity)) == length.(xci) || throw(
        DimensionMismatch(
            "velocity must be given on the grid nodes: $(size(first(velocity))) vs $(length.(xci))"
        )
    )

    # unpack data names and arrays
    data_names_c = string.(keys(data_c))
    data_arrays_c = values(data_c)

    velocity_field = pack_velocity(velocity, precision)

    # Create the VTK file
    vtk_grid(fname, xci...) do vtk
        for (name_i, array_i) in zip(data_names_c, data_arrays_c)
            vtk[name_i] = precision.(Array(array_i))
        end
        vtk["Velocity"] = velocity_field
        isnothing(t) || (vtk["TimeValue"] = t)

        # If pvd collection name is provided, add this file to the collection
        if !isnothing(pvd)
            time_value = isnothing(t) ? 0.0 : t
            paraview_collection(pvd; append = true) do pvd_collection
                collection_add_timestep(pvd_collection, vtk, time_value)
            end
        end
    end

    return nothing
end

function save_vtk(fname::String, xi, data::NamedTuple; precision = Float32, pvd::Union{Nothing, String} = nothing, t::Number = 0.0)
    # unpack data names and arrays
    data_names = string.(keys(data))
    data_arrays = values(data)

    # Create the VTK file
    vtk_grid(fname, xi...) do vtk
        for (name_i, array_i) in zip(data_names, data_arrays)
            vtk[name_i] = precision.(Array(array_i))
        end

        # If pvd collection name is provided, add this file to the collection
        if !isnothing(pvd)
            paraview_collection(pvd; append = true) do pvd_collection
                collection_add_timestep(pvd_collection, vtk, t)
            end
        end
    end

    return nothing
end

function _save_pvtk(
        fname::String, di::NTuple{N}, data::NamedTuple, velocity, igg::IGG, t, precision, pvd
    ) where {N}

    nxyz_A = size(first(values(data)))
    # global per-process extents (VTK overlap = 1), computed analytically (no comm)
    extents = vec(ImplicitGlobalGrid.metagrid(ImplicitGlobalGrid.extents_g, nxyz_A, 1))
    # `metagrid` is ordered column-major by Cartesian coords ⇒ matching part index
    part = LinearIndices(Tuple(igg.dims))[(igg.coords .+ 1)...]
    # this rank's local Cartesian coordinate extents (VTK overlap = 1)
    coords = ImplicitGlobalGrid.extents_g(nxyz_A, 1; dxyz = di)

    pvtk_grid(fname, coords...; part = part, extents = extents) do pvtk
        for (name_i, array_i) in pairs(data)
            sl = ImplicitGlobalGrid.extents(array_i, 1)
            pvtk[string(name_i)] = view(precision.(Array(array_i)), sl...)
        end
        if !isnothing(velocity)
            sl = ImplicitGlobalGrid.extents(first(velocity), 1)
            pvtk["Velocity"] = pack_velocity(velocity, precision, sl)
        end
        isnothing(t) || (pvtk["TimeValue"] = t)
        # only the main rank (part 1) writes the header, so only it touches the pvd
        if !isnothing(pvd) && igg.me == 0
            paraview_collection(pvd; append = true) do pvd_collection
                collection_add_timestep(pvd_collection, pvtk, isnothing(t) ? 0.0 : t)
            end
        end
    end
    return nothing
end

"""
    save_pvtk(fname, di::NTuple{N}, data_v::NamedTuple, data_c::NamedTuple, velocity::Tuple, igg::IGG; t=nothing, precision=Float32, pvd=nothing)

Parallel (MPI) counterpart of the serial `save_vtk` for an
`ImplicitGlobalGrid`-distributed grid (requires ImplicitGlobalGrid ≥ 0.17).
Writes vertex fields `data_v` + `velocity` to `<fname>_vertex.pvti` and cell
fields `data_c` to `<fname>_center.pvti`, one `.vti` piece per rank. `di` is the
**global** grid spacing (e.g. `grid.di.center`). Ranks overlap by one ghost
layer, so `update_halo!` before writing. If `pvd` is given, the datasets are
appended to `<pvd>_vertex.pvd` / `<pvd>_center.pvd` at time `t`.
"""
function save_pvtk(
        fname::String,
        di::NTuple{N, T},
        data_v::NamedTuple,
        data_c::NamedTuple,
        velocity::Tuple,
        igg::IGG;
        t::Union{Nothing, Number} = nothing,
        precision = Float32,
        pvd::Union{Nothing, String} = nothing,
    ) where {N, T}
    pvd_v = isnothing(pvd) ? nothing : pvd * "_vertex"
    pvd_c = isnothing(pvd) ? nothing : pvd * "_center"
    _save_pvtk(fname * "_vertex", di, data_v, velocity, igg, t, precision, pvd_v)
    _save_pvtk(fname * "_center", di, data_c, nothing, igg, t, precision, pvd_c)
    return nothing
end

function save_pvtk(
        fname::String,
        di::NTuple{N, T},
        data::NamedTuple,
        velocity::Tuple,
        igg::IGG;
        t::Union{Nothing, Number} = nothing,
        precision = Float32,
        pvd::Union{Nothing, String} = nothing,
    ) where {N, T}
    _save_pvtk(fname, di, data, velocity, igg, t, precision, pvd)
    return nothing
end

function save_pvtk(
        fname::String,
        di::NTuple{N, T},
        data::NamedTuple,
        igg::IGG;
        t::Union{Nothing, Number} = nothing,
        precision = Float32,
        pvd::Union{Nothing, String} = nothing,
    ) where {N, T}
    _save_pvtk(fname, di, data, nothing, igg, t, precision, pvd)
    return nothing
end

"""
    save_marker_chain(fname::String, chain::MarkerChain; conversion=1.0e3, pvd=nothing, t=0.0)

Save a vector of points as a line in a VTK file.

## Arguments
- `fname::String`: The name of the VTK file to save. The extension `.vtk` will be appended to the name.
- `chain::MarkerChain`: Marker chain object from JustPIC.jl.
- `conversion`: Conversion factor for coordinates (default: 1.0e3)
- `pvd::Union{Nothing, String}`: Optional ParaView collection filename for time series
- `t::Number`: Time value (default: 0.0)
"""
save_marker_chain(fname::String, chain; conversion = 1.0e3, pvd::Union{Nothing, String} = nothing, t::Number = 0.0) = save_marker_chain(fname, chain.cell_vertices ./ conversion, chain.h_vertices ./ conversion; pvd = pvd, t = t)

function save_marker_chain(
        fname::String, cell_vertices::Union{LinRange{Float64}, Vector{Float64}}, h_vertices::Vector{Float64};
        pvd::Union{Nothing, String} = nothing, t::Number = 0.0
    )
    cell_vertices_vec = collect(cell_vertices)  # Convert LinRange to Vector
    n_points = length(cell_vertices_vec)
    points = [
        SVector{3, Float64}(cell_vertices_vec[i], h_vertices[i], 0.0) for i in 1:n_points
    ]
    lines = [MeshCell(PolyData.Lines(), 1:(n_points))]  # Create a single line connecting all points

    vtk_grid(fname, points, lines) do vtk
        vtk["Points"] = points

        # If pvd collection name is provided, add this file to the collection
        if !isnothing(pvd)
            paraview_collection(pvd; append = true) do pvd_collection
                collection_add_timestep(pvd_collection, vtk, t)
            end
        end
    end
    return nothing
end

"""
    save_particles(particles::Particles{B, 2}, pPhases; conversion = 1e3, fname::String = "./particles", pvd=nothing, t=0.0) where B

Save particle data and their material phase to a VTK file.

## Arguments
- `particles::Particles{B, 2}`: The particle data, where `B` is the type of the particle coordinates.
- `pPhases`: The phases of the particles.
- `conversion`: A conversion factor for the particle coordinates (default is 1e3).
- `fname::String`: The name of the VTK file to save (default is "./particles").
- `pvd::Union{Nothing, String}`: Optional ParaView collection filename for time series
- `t::Number`: Time value (default: 0.0)
"""
function save_particles(particles, pPhases; conversion = 1.0e3, fname::String = "./particles", pvd::Union{Nothing, String} = nothing, t::Number = 0.0, precision = Float32)
    N = length(size(particles.index))
    return if N == 2
        save_particles2D(particles, pPhases, precision; conversion = conversion, fname = fname, pvd = pvd, t = t)
    elseif N == 3
        save_particles3D(particles, pPhases, precision; conversion = conversion, fname = fname, pvd = pvd, t = t)
    else
        error("The dimension of the model is $N. It must be 2 or 3!")
    end
end

function save_particles2D(particles, pPhases, precision; conversion = 1.0e3, fname::String = "./particles", pvd::Union{Nothing, String} = nothing, t::Number = 0.0)
    p = particles.coords
    ppx, ppy = p
    pxv = precision.(Array(ppx.data)[:] ./ conversion)
    pyv = precision.(Array(ppy.data)[:] ./ conversion)
    clr = precision.(Array(pPhases.data)[:])
    idxv = Array(particles.index.data[:])

    x = pxv[idxv]
    y = pyv[idxv]
    phase = clr[idxv]
    npoints = length(x)
    z = zeros(precision, npoints)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:npoints]

    return vtk_grid(fname, x, y, z, cells) do vtk
        vtk["phase", VTKPointData()] = phase

        # If pvd collection name is provided, add this file to the collection
        if !isnothing(pvd)
            paraview_collection(pvd; append = true) do pvd_collection
                collection_add_timestep(pvd_collection, vtk, t)
            end
        end
    end
end

function save_particles3D(particles, pPhases, precision; conversion = 1.0e3, fname::String = "./particles", pvd::Union{Nothing, String} = nothing, t::Number = 0.0)
    p = particles.coords
    ppx, ppy, ppz = p
    pxv = precision.(Array(ppx.data)[:] ./ conversion)
    pyv = precision.(Array(ppy.data)[:] ./ conversion)
    pzv = precision.(Array(ppz.data)[:] ./ conversion)
    clr = precision.(Array(pPhases.data)[:])
    idxv = Array(particles.index.data[:])

    x = pxv[idxv]
    y = pyv[idxv]
    z = pzv[idxv]
    phase = clr[idxv]
    npoints = length(x)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:npoints]
    return vtk_grid(fname, x, y, z, cells) do vtk
        vtk["phase", VTKPointData()] = phase

        # If pvd collection name is provided, add this file to the collection
        if !isnothing(pvd)
            paraview_collection(pvd; append = true) do pvd_collection
                collection_add_timestep(pvd_collection, vtk, t)
            end
        end
    end
end

"""
    save_particles(particles::Particles{B, 2}; conversion = 1e3, fname::String = "./particles", pvd=nothing, t=0.0) where B

Save particle data to a VTK file.

## Arguments
- `particles::Particles{B, 2}`: The particle data, where `B` is the type of the particle coordinates.
- `conversion`: A conversion factor for the particle coordinates (default is 1e3).
- `fname::String`: The name of the VTK file to save (default is "./particles").
- `pvd::Union{Nothing, String}`: Optional ParaView collection filename for time series
- `t::Number`: Time value (default: 0.0)
"""
function save_particles(particles; conversion = 1.0e3, fname::String = "./particles", pvd::Union{Nothing, String} = nothing, t::Number = 0.0, precision = Float32)
    N = length(size(particles.index))
    return if N == 2
        save_particles2D(particles, precision; conversion = conversion, fname = fname, pvd = pvd, t = t)
    elseif N == 3
        save_particles3D(particles, precision; conversion = conversion, fname = fname, pvd = pvd, t = t)
    else
        error("The dimension of the model is $N. It must be 2 or 3!")
    end
end

function save_particles2D(particles, precision; conversion = 1.0e3, fname::String = "./particles", pvd::Union{Nothing, String} = nothing, t::Number = 0.0)
    p = particles.coords
    ppx, ppy = p
    pxv = precision.(Array(ppx.data)[:] ./ conversion)
    pyv = precision.(Array(ppy.data)[:] ./ conversion)
    idxv = Array(particles.index.data[:])

    x = pxv[idxv]
    y = pyv[idxv]
    npoints = length(x)
    z = zeros(precision, npoints)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:npoints]

    return vtk_grid(fname, x, y, z, cells) do vtk
        vtk["phase", VTKPointData()] = 1

        # If pvd collection name is provided, add this file to the collection
        if !isnothing(pvd)
            paraview_collection(pvd; append = true) do pvd_collection
                collection_add_timestep(pvd_collection, vtk, t)
            end
        end
    end
end

function save_particles3D(particles, precision; conversion = 1.0e3, fname::String = "./particles", pvd::Union{Nothing, String} = nothing, t::Number = 0.0)
    p = particles.coords
    ppx, ppy, ppz = p
    pxv = precision.(Array(ppx.data)[:] ./ conversion)
    pyv = precision.(Array(ppy.data)[:] ./ conversion)
    pzv = precision.(Array(ppz.data)[:] ./ conversion)
    idxv = Array(particles.index.data[:])

    x = pxv[idxv]
    y = pyv[idxv]
    z = pzv[idxv]
    npoints = length(x)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:npoints]
    return vtk_grid(fname, x, y, z, cells) do vtk
        vtk["phase", VTKPointData()] = 1

        # If pvd collection name is provided, add this file to the collection
        if !isnothing(pvd)
            paraview_collection(pvd; append = true) do pvd_collection
                collection_add_timestep(pvd_collection, vtk, t)
            end
        end
    end
end

"""
    save_particles(particles, igg::IGG; pPhases=nothing, conversion=1e3, fname="./particles", pvd=nothing, t=0.0, precision=Float32)

Parallel (MPI) counterpart of [`save_particles`](@ref): each rank writes its own
active particles as an unstructured `.vtu` piece, tied together by `<fname>.pvtu`.
Works for 2D and 3D `particles`. `pPhases` (a `CellArray` of phase ids) is written
as the `phase` point field when given, otherwise a constant is used. If `pvd` is
given, the `.pvtu` datasets are appended to `<pvd>.pvd` at time `t` to build a
time series (only rank 0 touches the collection).
"""
function save_particles(
        particles, igg::IGG;
        pPhases = nothing, conversion = 1.0e3, fname::String = "./particles",
        pvd::Union{Nothing, String} = nothing, t::Number = 0.0, precision = Float32,
    )
    coords = particles.coords
    idxv = Array(particles.index.data[:])
    xyz = map(coords) do p
        precision.(Array(p.data)[:] ./ conversion)[idxv]
    end
    length(xyz) == 2 && (xyz = (xyz..., zeros(precision, length(first(xyz)))))  # pad z for 2D
    phase = isnothing(pPhases) ? 1 : precision.(Array(pPhases.data)[:])[idxv]
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:length(first(xyz))]

    return pvtk_grid(
        fname, xyz..., cells; part = igg.me + 1, nparts = prod(igg.dims)
    ) do pvtk
        pvtk["phase", VTKPointData()] = phase
        if !isnothing(pvd) && igg.me == 0
            paraview_collection(pvd; append = true) do pvd_collection
                collection_add_timestep(pvd_collection, pvtk, t)
            end
        end
    end
end
