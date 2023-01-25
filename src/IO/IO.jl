macro namevar(x) 
    # if input is a structure, take the innermost name
    # i.e. stokes.V.Vx => "Vx"
    name = split(string(x), ".")[end]
    return :($(esc(name)), $(esc(x)))
end

"""
    metadata(src, file, dst)

Copy `file`, Manifest.toml, and, Project.toml from `src` to `dst`
"""
function metadata(src, file, dst)
    if !ispath(dst) 
        println("Create $dst folder") 
        mkpath(dest)
    end
    for f in (file, "Manifest.toml", "Project.toml")
        cp(joinpath(src,f), dst)
    end
end

"""
    checkpointing(dst, stokes, T, η, time)

Save necessary data in `dst` as and HDF5 file to restart the model from the state at `time`
"""
function checkpointing(dst, stokes, T, η, time)
    fname = joinpath(dst, "checkpoint")
    h5open("$(fname).h5", "w") do file
        write(file, @namevar(time)...)
        write(file, @namevar(stokes.V.Vx)...)
        write(file, @namevar(stokes.V.Vy)...)
        write(file, @namevar(stokes.P)...)
        write(file, @namevar(T)...)
        write(file, "viscosity", η)
    end
end


"""
    function save_hdf5(fname, data)

Save `data` as and HDF5 file with name (in)
"""
function save_hdf5(fname, data::Vararg{Any, N}) where N
    h5open("$(fname).h5", "w") do file
        for data_i in data
            save_data(file, data_i)
        end
    end
end

@inline save_data(file, data) = write(file, @namevar(data)...)

function save_data(file, data::Geometry{N}) where N

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

center_coordinates(data::Geometry{N}) where N = ntuple(i-> collect(data.xci[i]), Val(N))
vertex_coordinates(data::Geometry{N}) where N = ntuple(i-> collect(data.xvi[i]), Val(N))