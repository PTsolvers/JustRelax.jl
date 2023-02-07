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
_tocpu(x::T) where T<:CuArray = Array(x)

"""
    metadata(src, file, dst)

Copy `file`, Manifest.toml, and, Project.toml from `src` to `dst`
"""
function metadata(src, file, dst)
    @assert dst != pwd()
    if !ispath(dst) 
        println("Created $dst folder") 
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
    function save_hdf5(fname, data)

Save `data` as the `fname.h5` HDF5 file
"""
function save_hdf5(fname, data::Vararg{Any, N}) where N
    h5open("$(fname).h5", "w") do file
        for data_i in data
            save_data(file, data_i)
        end
    end
end

"""
    function save_hdf5(dst, fname, data)

Save `data` as the `fname.h5` HDF5 file in the folder `dst`
"""
function save_hdf5(dst, fname, data::Vararg{Any, N}) where N
    !isdir(dst) && mkpath(dst) # creat folder in case it does not exist
    pth_name = joinpath(dst, fname)
    save_hdf5(pth_name, data)
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


# if do_save
#     dim_g = (nx_g()-2, ny_g()-2, nz_g()-2)
#     @parallel preprocess_visu!(Vn, τII, Vx, Vy, Vz, τxx, τyy, τzz, τxy, τxz, τyz)
#     @parallel apply_mask!(Vn, τII, ϕ)
#     out_h5 = joinpath(out_path,out_name)*".h5"
#     I = CartesianIndices(( (coords[1]*(nx-2) + 1):(coords[1]+1)*(nx-2),
#                            (coords[2]*(ny-2) + 1):(coords[2]+1)*(ny-2),
#                            (coords[3]*(nz-2) + 1):(coords[3]+1)*(nz-2) ))
#     fields = Dict("ϕ"=>inn(ϕ),"Vn"=>Vn,"τII"=>τII,"Pr"=>inn(Pt))
#     (me==0) && print("Saving HDF5 file...")
#     write_h5(out_h5,fields,dim_g,I,comm_cart,info) # comm_cart,MPI.Info() are varargs to exclude if using non-parallel HDF5 lib
#     # write_h5(out_h5,fields,dim_g,I) # comm_cart,MPI.Info() are varargs to exclude if using non-parallel HDF5 lib
#     (me==0) && println(" done")
#     # write XDMF
#     if me == 0
#         print("Saving XDMF file...")
#         write_xdmf(joinpath(out_path,out_name)*".xdmf3",out_name*".h5",fields,(xc[2],yc[2],zc[2]),(dx,dy,dz),dim_g)
#         println(" done")
#     end
# end

# comm_cart, MPI.Info() are varargs to exclude if using non-parallel HDF5 lib
function write_h5(path, fields, dim_g, I, comm_cart, info)
    if !HDF5.has_parallel() && (length(args)>0)
        @warn("HDF5 has no parallel support.")
    end
    h5open(path, "w", comm_cart, info) do io
        for (name,field) in fields
            dset               = create_dataset(io, "/$name", datatype(eltype(field)), dataspace(dim_g))
            dset[I.indices...] = Array(field)
        end
    end
    return
end