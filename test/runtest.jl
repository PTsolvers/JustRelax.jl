using ParallelStencil
using JustRelax
using Printf, LinearAlgebra, GLMakie
using ParallelStencil.FiniteDifferences2D

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)

# include benchmark functions
include("test2d.jl")

# run benchmark in double precission
geometry, stokes, ρ = solkz(nx=31, ny=31)

# lets switch to single precission
ParallelStencil.@reset_parallel_stencil

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float32, 2)
environment!(model)

# run benchmark in double precission
geometry, stokes, ρ = solkz(nx=31, ny=31)