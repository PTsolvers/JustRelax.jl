using LinearAlgebra, CairoMakie
using JustRelax

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)

# model resolution (number of gridpoints)
nx, ny, nz = 16, 16
