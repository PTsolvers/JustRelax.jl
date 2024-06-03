module DataIO

using WriteVTK
using HDF5
using JLD2
using MPI

import ..JustRelax: Geometry
import ..JustRelax: IGG

include("H5.jl")

export save_hdf5, checkpointing_hdf5, load_checkpoint_hdf5, metadata

include("JLD2.jl")

export checkpointing_jld2, load_checkpoint_jld2

include("VTK.jl")

export VTKDataSeries, append!, save_vtk

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
        !isfile(joinpath(f)) && continue
        newfile = joinpath(dst, basename(f))
        isfile(newfile) && rm(newfile)
        cp(joinpath(src,f), newfile)
    end
end

end
