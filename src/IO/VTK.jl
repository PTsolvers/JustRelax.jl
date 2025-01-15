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
    for (name_i, array_i) in zip(data_names, data_arrays)
        vtk[name_i] = Array(array_i)
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

function save_vtk(
    fname::String,
    xvi,
    xci,
    data_v::NamedTuple,
    data_c::NamedTuple,
    velocity::NTuple{N,T};
    t::Number=nothing,
) where {N,T}

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
    fname::String, xci, data_c::NamedTuple, velocity::NTuple{N,T}; t::Number=nothing
) where {N,T}

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
    save_marker_chain(fname::String, cell_vertices::LinRange{Float64}, h_vertices::Vector{Float64})

Save a vector of points as a line in a VTK file. The X and Y coordinates of the points are given by `cell_vertices` and `h_vertices`, respectively.
The Z coordinate is set to zero as we are working in 2D.

## Arguments
- `fname::String`: The name of the VTK file to save. The extension `.vtk` will be appended to the name.
- `cell_vertices::LinRange{Float64}`: A range of X coordinates for the points.
- `h_vertices::Vector{Float64}`: A vector of Y coordinates for the points.

## Example
```julia
cell_vertices = LinRange(0.0, 1.0, 10)
h_vertices = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
save_marker_chain("Example", cell_vertices, h_vertices)
```
"""
function save_marker_chain(
    fname::String, cell_vertices::LinRange{Float64}, h_vertices::Vector{Float64}
)
    cell_vertices_vec = collect(cell_vertices)  # Convert LinRange to Vector
    n_points = length(cell_vertices_vec)
    points = [
        SVector{3,Float64}(cell_vertices_vec[i], h_vertices[i], 0.0) for i in 1:n_points
    ]
    lines = [MeshCell(PolyData.Lines(), 1:(n_points))]  # Create a single line connecting all points

    vtk_grid(fname, points, lines) do vtk
        vtk["Points"] = points
    end
    return nothing
end
