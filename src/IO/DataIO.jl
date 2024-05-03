module DataIO

using WriteVTK
using HDF5
using JLD2
using MPI

import ..JustRelax: Geometry

include("H5.jl")

export save_hdf5, checkpointing, load_checkpoint, metadata

include("JLD2.jl")

export checkpointing_jld2, load_checkpoint_jld2

include("VTK.jl")

export VTKDataSeries, append!, save_vtk

export metadata

"""
    metadata(src, file, dst)

Copy `file`, Manifest.toml, and, Project.toml from `src` to `dst`
"""
function metadata(src, file, dst)
    @assert dst != pwd()
    if !ispath(dst)
        println("Created $dst folder")
        mkpath(dst)
    end
    for f in (file, "Manifest.toml", "Project.toml")
        !isfile(f) && continue
        newfile = joinpath(dst, basename(f))
        isfile(newfile) && rm(newfile)
        cp(joinpath(src, f), newfile)
    end
end

end
