using LinearAlgebra, CairoMakie
using JustRelax
using MPI: MPI
using WriteVTK
# using GeophysicalModelGenerator
using StaticArrays

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 3)
environment!(model)

# plot slice of the result
viz = false

# save model as paraview file
Paraview = false

# model resolution (number of gridpoints)
nx, ny, nz = 16, 16, 16

# set MPI
finalize_MPI = false

# include benchmark
include("slab/DiffusionSlab.jl")

# run model
geometry, Temperature = DiffusionSlab(;
    ttot=5e6 * 3600 * 24 * 365,
    dt=500e3 * 3600 * 24 * 365,
    nx=64,
    ny=64,
    nz=64,
    lx=1000e3,
    ly=1000e3,
    lz=1000e3,
    b_width=(1, 1, 1),
    init_MPI=MPI.Initialized() ? false : true,
    finalize_MPI=false,
)

# plot results
if viz
    heatmap(xci[1], xci[2], (thermal.T)[:, 1, :]; colormap=:vik)
end

# save Paraview file
if paraview
    vtk_grid("fields", geometry.xc[1], geometry.xc[2], geometry.xc[3]) do vtk
        vtk["Phases"] = Phases
        vtk["T"] = Temp
    end
end
