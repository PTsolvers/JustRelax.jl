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

Save VTK data with multiblock format containing both vertex and cell data.

## Arguments
- `fname::String`: The filename for the VTK file (without extension)
- `xvi`: Vertex coordinates (tuple of coordinate arrays)
- `xci`: Cell center coordinates (tuple of coordinate arrays)
- `data_v::NamedTuple`: Data defined at vertices
- `data_c::NamedTuple`: Data defined at cell centers
- `velocity::NTuple{N, T}`: Velocity field as a tuple of N-dimensional arrays
- `t::Number`: Time value (default: 0)
- `pvd::Union{Nothing, String}`: Optional ParaView collection filename. If provided, the VTK file will be added to a time series collection. WriteVTK.jl automatically handles creating new collections or appending to existing ones.

## Examples
```julia
# Basic usage (backward compatible)
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
        velocity::NTuple{N, T};
        precision = Float32,
        t::Number = 0,
        pvd::Union{Nothing, String} = nothing,
    ) where {N, T}

    # unpack data names and arrays
    data_names_v = string.(keys(data_v))
    data_arrays_v = values(data_v)
    data_names_c = string.(keys(data_c))
    data_arrays_c = values(data_c)

    velocity_field = rand(N, size(first(velocity))...)
    for (i, v) in enumerate(velocity)
        velocity_field[i, :, :, :] = precision.(Array(v))
    end

    vtk_multiblock(fname) do vtm
        # First block.
        # Variables stores in cell centers
        vtk_grid(vtm, xci...) do vtk
            for (name_i, array_i) in zip(data_names_c, data_arrays_c)
                vtk[name_i] = precision.(Array(array_i))
            end
        end
        # Second block.
        # Variables stores in cell vertices
        vtk_grid(vtm, xvi...) do vtk
            for (name_i, array_i) in zip(data_names_v, data_arrays_v)
                vtk[name_i] = precision.(Array(array_i))
            end
            vtk["Velocity"] = velocity_field
            isnothing(t) || (vtk["TimeValue"] = t)
        end

        # If pvd collection name is provided, add this file to the collection
        if !isnothing(pvd)
            paraview_collection(pvd; append = true) do pvd_collection
                collection_add_timestep(pvd_collection, vtm, t)
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
- `velocity::NTuple{N, T}`: Velocity field as a tuple of N-dimensional arrays
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
        velocity::NTuple{N, T};
        precision = Float32,
        t::Number = nothing,
        pvd::Union{Nothing, String} = nothing
    ) where {N, T}

    # unpack data names and arrays
    data_names_c = string.(keys(data_c))
    data_arrays_c = values(data_c)

    velocity_field = rand(N, size(first(velocity))...)
    for (i, v) in enumerate(velocity)
        velocity_field[i, :, :, :] = precision.(Array(v))
    end

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
    z = zeros(npoints)
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
    ppx, ppy = p
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
