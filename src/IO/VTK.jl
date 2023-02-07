function save_vtk(fname::String, xvi, xci, data_v::NamedTuple, data_c::NamedTuple)

    # unpack data names and arrays
    data_names_v  = string.(keys(data_v))
    data_arrays_v = values(data_v)
    data_names_c  = string.(keys(data_c))
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

