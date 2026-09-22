"""
    JustRelax.DataIO

Checkpointing and output writing.

`checkpointing_hdf5`/`load_checkpoint_hdf5` and `checkpointing_jld2`/`load_checkpoint_jld2`
save and restore the model state, while `save_vtk`, `save_particles`, and
`save_marker_chain` write fields, particles, and marker chains for ParaView. Every routine
moves the data to the CPU before writing, whatever backend the fields live on. The
submodule is loaded together with JustRelax; its names are reached as `JustRelax.DataIO.f`
or by `using JustRelax.DataIO`.
"""
module DataIO

using WriteVTK
using HDF5
using JLD2
using MPI
using StaticArrays

import ..JustRelax: Geometry
import ..JustRelax: IGG
import ..JustRelax: ImplicitGlobalGrid

include("H5.jl")

export save_hdf5,
    checkpointing_hdf5,
    load_checkpoint_hdf5,
    metadata,
    center_coordinates,
    vertex_coordinates,
    save_data

include("JLD2.jl")

export checkpointing_jld2, load_checkpoint_jld2

include("VTK.jl")

export VTKDataSeries, append!, save_vtk, save_pvtk, save_marker_chain, save_particles

export metadata

"""
    metadata(src, dst, files...)

Copy `files...`, Manifest.toml, and Project.toml from `src` to `dst`
"""
function metadata(src, dst, files...)
    @assert dst != pwd()
    if !ispath(dst)
        println("Created $dst folder")
        mkpath(dst)
    end
    for f in vcat(collect(files), ["Manifest.toml", "Project.toml"])
        srcfile = if isfile(joinpath(src, f))
            joinpath(src, f)
        elseif isfile(joinpath(src, "test", f))
            joinpath(src, "test", f)
        else
            continue
        end
        newfile = joinpath(dst, basename(f))
        isfile(newfile) && rm(newfile)
        cp(srcfile, newfile, force = true)
    end
    return
end

end
