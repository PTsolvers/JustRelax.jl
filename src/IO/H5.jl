# if input is a structure, take the innermost name
# i.e. stokes.V.Vx => "Vx"
macro namevar(x)
    name = split(string(x), ".")[end]
    return quote
        tmp = $(esc(x))
        $(esc(name)), _tocpu(tmp)
    end
end

_tocpu(x) = x
_tocpu(x::T) where {T<:CuArray} = Array(x)
_tocpu(x::T) where {T<:ROCArray} = Array(x)

"""
    checkpointing(dst, stokes, T, η, time)

Save necessary data in `dst` as and HDF5 file to restart the model from the state at `time`
"""
function checkpointing(dst, stokes, T, η, time)
    !isdir(dst) && mkpath(dst) # creat folder in case it does not exist
    fname = joinpath(dst, "checkpoint")
    h5open("$(fname).h5", "w") do file
        write(file, @namevar(time)...)
        write(file, @namevar(stokes.V.Vx)...)
        write(file, @namevar(stokes.V.Vy)...)
        write(file, @namevar(stokes.P)...)
        write(file, @namevar(T)...)
        write(file, "viscosity", _tocpu(η))
    end
end

"""
    function save_hdf5(dst, fname, data)

Save `data` as the `fname.h5` HDF5 file in the folder `dst`
"""
function save_hdf5(dst, fname, data::Vararg{Any,N}) where {N}
    !isdir(dst) && mkpath(dst) # creat folder in case it does not exist
    pth_name = joinpath(dst, fname)
    return save_hdf5(pth_name, data)
end

# comm_cart, info comm_cart, MPI.Info()
function save_hdf5(fname, dim_g, I, comm_cart, info, data::Vararg{Any,N}) where {N}
    @assert HDF5.has_parallel()
    h5open("$(fname).h5", "w", comm_cart, info) do file
        for data_i in data
            name, field = @namevar data_i
            dset = create_dataset(
                file, "/" * name, datatype(eltype(field)), dataspace(dim_g)
            )
            dset[I.indices...] = Array(field)
        end
    end
    return nothing
end

"""
    function save_hdf5(fname, data)

Save `data` as the `fname.h5` HDF5 file
"""
function save_hdf5(fname, data::Vararg{Any,N}) where {N}
    h5open("$(fname).h5", "w") do file
        for data_i in data
            save_data(file, data_i)
        end
    end
end

@inline save_data(file, data) = write(file, @namevar(data)...)

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
