"""
    VTKDataSeries{T,S,G}

A structure representing a series of VTK data files.

# Fields
- `series`: A Paraview collection object representing the series of VTK files.
- `path`: The path where the VTK files are stored.
- `name`: The name of the VTK series.
- `grid`: The grid coordinates.

# Description
The `VTKDataSeries` structure is used to manage a series of VTK files. It contains a Paraview collection object, which represents the series of VTK files, the path where the files are stored, the name of the series, and the grid coordinates.

"""
struct VTKDataSeries{T,S,G}
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
        series = paraview_collection(full_name; append=true)
        return new{typeof(series),String,typeof(xi)}(series, path, name, xi)
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
    for (name_i, arrary_i) in zip(data_names, data_arrays)
        vtk[name_i] = Array(arrary_i)
    end
    # close vtk file
    vtk_save(vtk)
    # open pvd file
    pvd_name = joinpath(data_series.path, data_series.name)
    pvd = paraview_collection(pvd_name; append=true)
    # add vtk file to time series
    collection_add_timestep(pvd, vtk, seconds)
    # close pvd file
    vtk_save(pvd)

    return nothing
end

"""
    save_vtk(fname::String, xvi, xci, data_v::NamedTuple, data_c::NamedTuple)

Save data to a VTK file.

# Arguments
- `fname`: The name of the file to save to.
- `xvi`: The coordinates of the cell vertices.
- `xci`: The coordinates of the cell centers.
- `data_v`: A named tuple containing the data to be stored at the cell vertices.
- `data_c`: A named tuple containing the data to be stored at the cell centers.

# Description
The function first unpacks the data names and arrays from `data_v` and `data_c`. It then creates a VTK multiblock file with two blocks. The first block contains the data at the cell centers, and the second block contains the data at the cell vertices. The data is stored in the VTK file with the names provided in `data_v` and `data_c`.

"""
function save_vtk(fname::String, xvi, xci, data_v::NamedTuple, data_c::NamedTuple)

    # unpack data names and arrays
    data_names_v = string.(keys(data_v))
    data_arrays_v = values(data_v)
    data_names_c = string.(keys(data_c))
    data_arrays_c = values(data_c)

    vtk_multiblock(fname) do vtm
        # First block.
        # Variables stores in cell centers
        vtk_grid(vtm, xci...) do vtk
            for (name_i, arrary_i) in zip(data_names_c, data_arrays_c)
                vtk[name_i] = Array(arrary_i)
            end
        end
        # Second block.
        # Variables stores in cell vertices
        vtk_grid(vtm, xvi...) do vtk
            for (name_i, arrary_i) in zip(data_names_v, data_arrays_v)
                vtk[name_i] = Array(arrary_i)
            end
        end
    end

    return nothing
end

"""
    save_vtk(fname::String, xi, data::NamedTuple)

Save data to a VTK file.

# Arguments
- `fname`: The name of the file to save to.
- `xi`: The coordinates of the cell vertices or centers.
- `data`: A named tuple containing the data to be stored.

# Description
The function first unpacks the data names and arrays from `data`. It then creates a VTK file and stores the data in it with the names provided in `data`.

"""
function save_vtk(fname::String, xi, data::NamedTuple)
    # unpack data names and arrays
    data_names = string.(keys(data))
    data_arrays = values(data)
    # make and save VTK file
    vtk_grid(fname, xi...) do vtk
        for (name_i, arrary_i) in zip(data_names, data_arrays)
            vtk[name_i] = Array(arrary_i)
        end
    end

    return nothing
end
