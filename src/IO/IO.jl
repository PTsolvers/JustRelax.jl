macro namevar(x)
    return :($(esc(string(x))))
end

function save_hdf5(fname, data)
    h5open("$(fname).h5", "w") do file
        for data_i in data
            save_data(file, data_i)
        end
    end
end

function save_data(file, data_i)
    data_name = @namevar data_i
    write(file, data_name, data_i)
    return nothing
end

function save_data(file, data::Geometry{N}) where {N}
    xci = center_coordinates(data)
    xvi = vertex_coordinates(data)

    write(file, "Xc", xci[1])
    write(file, "Yc", xci[2])
    write(file, "Xv", xvi[1])
    write(file, "Yv", xvi[2])
    if N == 3
        write(file, "Zc", xci[3])
        write(file, "Zv", xvi[3])
    end

    return nothing
end

center_coordinates(data::Geometry{N}) where {N} = ntuple(i -> collect(data.xci[i]), Val(N))
vertex_coordinates(data::Geometry{N}) where {N} = ntuple(i -> collect(data.xvi[i]), Val(N))
