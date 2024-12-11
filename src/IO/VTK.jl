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
