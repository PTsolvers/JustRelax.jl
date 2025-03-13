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

function save_vtk(
        fname::String,
        xvi,
        xci,
        data_v::NamedTuple,
        data_c::NamedTuple,
        velocity::NTuple{N, T};
        t::Number = 0,
    ) where {N, T}

    # unpack data names and arrays
    data_names_v = string.(keys(data_v))
    data_arrays_v = values(data_v)
    data_names_c = string.(keys(data_c))
    data_arrays_c = values(data_c)

    velocity_field = rand(N, size(first(velocity))...)
    for (i, v) in enumerate(velocity)
        velocity_field[i, :, :, :] = v
    end

    vtk_multiblock(fname) do vtm
        # First block.
        # Variables stores in cell centers
        vtk_grid(vtm, xci...) do vtk
            for (name_i, array_i) in zip(data_names_c, data_arrays_c)
                vtk[name_i] = Array(array_i)
            end
        end
        # Second block.
        # Variables stores in cell vertices
        vtk_grid(vtm, xvi...) do vtk
            for (name_i, array_i) in zip(data_names_v, data_arrays_v)
                vtk[name_i] = Array(array_i)
            end
            vtk["Velocity"] = velocity_field
            isnothing(t) || (vtk["TimeValue"] = t)
        end
    end

    return nothing
end

function save_vtk(
        fname::String, xci, data_c::NamedTuple, velocity::NTuple{N, T}; t::Number = nothing
    ) where {N, T}

    # unpack data names and arrays
    data_names_c = string.(keys(data_c))
    data_arrays_c = values(data_c)

    velocity_field = rand(N, size(first(velocity))...)
    for (i, v) in enumerate(velocity)
        velocity_field[i, :, :, :] = v
    end

    # Variables stores in cell centers
    vtk_grid(fname, xci...) do vtk
        for (name_i, array_i) in zip(data_names_c, data_arrays_c)
            vtk[name_i] = Array(array_i)
        end
        vtk["Velocity"] = velocity_field
        isnothing(t) || (vtk["TimeValue"] = t)
    end

    return nothing
end

function save_vtk(fname::String, xi, data::NamedTuple)
    # unpack data names and arrays
    data_names = string.(keys(data))
    data_arrays = values(data)
    # make and save VTK file
    vtk_grid(fname, xi...) do vtk
        for (name_i, array_i) in zip(data_names, data_arrays)
            vtk[name_i] = Array(array_i)
        end
    end

    return nothing
end

"""
    save_marker_chain(fname::String, chain::MarkerChain)

Save a vector of points as a line in a VTK file.

## Arguments
- `fname::String`: The name of the VTK file to save. The extension `.vtk` will be appended to the name.
- `chain::MarkerChain`: Marker chain object from JustPIC.jl.
"""
save_marker_chain(fname::String, chain; conversion = 1.0e3) = save_marker_chain(fname, chain.cell_vertices ./ conversion, chain.h_vertices ./ conversion)

function save_marker_chain(
        fname::String, cell_vertices::LinRange{Float64}, h_vertices::Vector{Float64}
    )
    cell_vertices_vec = collect(cell_vertices)  # Convert LinRange to Vector
    n_points = length(cell_vertices_vec)
    points = [
        SVector{3, Float64}(cell_vertices_vec[i], h_vertices[i], 0.0) for i in 1:n_points
    ]
    lines = [MeshCell(PolyData.Lines(), 1:(n_points))]  # Create a single line connecting all points

    vtk_grid(fname, points, lines) do vtk
        vtk["Points"] = points
    end
    return nothing
end

"""
    save_particles(particles::Particles{B, 2}, pPhases; conversion = 1e3, fname::String = "./particles") where B

Save particle data and their material phase to a VTK file.

## Arguments
- `particles::Particles{B, 2}`: The particle data, where `B` is the type of the particle coordinates.
- `pPhases`: The phases of the particles.
- `conversion`: A conversion factor for the particle coordinates (default is 1e3).
- `fname::String`: The name of the VTK file to save (default is "./particles").
"""
function save_particles(particles, pPhases; conversion = 1.0e3, fname::String = "./particles")
    N = length(size(particles.index))
    return if N == 2
        save_particles2D(particles, pPhases; conversion = conversion, fname = fname)
    elseif N == 3
        save_particles3D(particles, pPhases; conversion = conversion, fname = fname)
    else
        error("The dimension of the model is $N. It must be 2 or 3!")
    end
end

function save_particles2D(particles, pPhases; conversion = 1.0e3, fname::String = "./particles")
    p = particles.coords
    ppx, ppy = p
    pxv = Array(ppx.data)[:] ./ conversion
    pyv = Array(ppy.data)[:] ./ conversion
    clr = Array(pPhases.data)[:]
    idxv = Array(particles.index.data[:])

    x = pxv[idxv]
    y = pyv[idxv]
    phase = clr[idxv]
    npoints = length(x)
    z = zeros(npoints)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:npoints]

    return vtk_grid(fname, x, y, z, cells) do vtk
        vtk["phase", VTKPointData()] = phase
    end
end

function save_particles3D(particles, pPhases; conversion = 1.0e3, fname::String = "./particles")
    p = particles.coords
    ppx, ppy = p
    pxv = Array(ppx.data)[:] ./ conversion
    pyv = Array(ppy.data)[:] ./ conversion
    pzv = Array(ppz.data)[:] ./ conversion
    clr = Array(pPhases.data)[:]
    idxv = Array(particles.index.data[:])

    x = pxv[idxv]
    y = pyv[idxv]
    z = pzv[idxv]
    phase = clr[idxv]
    npoints = length(x)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:npoints]
    return vtk_grid(fname, x, y, z, cells) do vtk
        vtk["phase", VTKPointData()] = phase
    end
end

"""
    save_particles(particles::Particles{B, 2}; conversion = 1e3, fname::String = "./particles") where B

Save particle data to a VTK file.

## Arguments
- `particles::Particles{B, 2}`: The particle data, where `B` is the type of the particle coordinates.
- `pPhases`: The phases of the particles.
- `conversion`: A conversion factor for the particle coordinates (default is 1e3).
- `fname::String`: The name of the VTK file to save (default is "./particles").
"""
function save_particles(particles; conversion = 1.0e3, fname::String = "./particles")
    N = length(size(particles.index))
    return if N == 2
        save_particles2D(particles; conversion = conversion, fname = fname)
    elseif N == 3
        save_particles3D(particles; conversion = conversion, fname = fname)
    else
        error("The dimension of the model is $N. It must be 2 or 3!")
    end
end

function save_particles2D(particles; conversion = 1.0e3, fname::String = "./particles")
    p = particles.coords
    ppx, ppy = p
    pxv = Array(ppx.data)[:] ./ conversion
    pyv = Array(ppy.data)[:] ./ conversion
    idxv = Array(particles.index.data[:])

    x = pxv[idxv]
    y = pyv[idxv]
    npoints = length(x)
    z = zeros(npoints)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:npoints]

    return vtk_grid(fname, x, y, z, cells) do vtk
        vtk["phase", VTKPointData()] = 1
    end
end

function save_particles3D(particles; conversion = 1.0e3, fname::String = "./particles")
    p = particles.coords
    ppx, ppy = p
    pxv = Array(ppx.data)[:] ./ conversion
    pyv = Array(ppy.data)[:] ./ conversion
    pzv = Array(ppz.data)[:] ./ conversion
    idxv = Array(particles.index.data[:])

    x = pxv[idxv]
    y = pyv[idxv]
    z = pzv[idxv]
    npoints = length(x)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:npoints]
    return vtk_grid(fname, x, y, z, cells) do vtk
        vtk["phase", VTKPointData()] = 1
    end
end
