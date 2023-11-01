module DataIO

using WriteVTK
using HDF5
using MPI
using CUDA, AMDGPU

import ..JustRelax: Geometry

include("H5.jl")

export save_hdf5, checkpointing, metadata

include("VTK.jl")

export VTKDataSeries, append!, save_vtk

export metadata

"""
    metadata(src, file, dst)

Copy specific metadata files from a source directory to a destination directory.

# Arguments
- `src`: The source directory.
- `file`: The name of the file to be copied.
- `dst`: The destination directory.

# Description
The function first checks if the destination directory exists. If it doesn't, it creates the directory. Then it iterates over a list of files (`file`, `Manifest.toml`, `Project.toml`), and if the file exists in the source directory, it copies the file to the destination directory. If a file with the same name already exists in the destination directory, it is first deleted before the new file is copied.

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
