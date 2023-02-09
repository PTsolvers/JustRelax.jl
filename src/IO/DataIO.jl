module DataIO

using WriteVTK
using HDF5
using MPI
using CUDA

import ..JustRelax: Geometry

include("H5.jl")

export save_hdf5, checkpointing, metadata

include("VTK.jl")

export save_vtk

export metadata

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
        cp(joinpath(src, f), dst)
    end
end

end
