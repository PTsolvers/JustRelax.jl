module JustRelax2D

using ..JustRelax
using JustPIC, JustPIC._2D
using StaticArrays
using CellArrays
using ParallelStencil, ParallelStencil.FiniteDifferences2D
using ImplicitGlobalGrid
using GeoParams, LinearAlgebra, Printf
using Statistics
using MPI

import JustRelax: IGG, BackendTrait, CPUBackendTrait, backend, CPUBackend
import JustRelax: PTStokesCoeffs
import JustRelax:
    AbstractBoundaryConditions,
    TemperatureBoundaryConditions,
    AbstractFlowBoundaryConditions,
    DisplacementBoundaryConditions,
    VelocityBoundaryConditions,
    apply_dirichlet,
    apply_dirichlet!

import JustRelax: normal_stress, shear_stress, shear_vorticity

import JustPIC._2D: numphases, nphases

@init_parallel_stencil(Threads, Float64, 2)

include("common.jl")
include("stokes/Stokes2D.jl")
export solve!

end

module JustRelax3D

using ..JustRelax
using JustPIC, JustPIC._3D
using StaticArrays
using CellArrays
using ParallelStencil, ParallelStencil.FiniteDifferences3D
using ImplicitGlobalGrid
using GeoParams, LinearAlgebra, Printf
using Statistics
using MPI

import JustRelax: IGG, BackendTrait, CPUBackendTrait, backend, CPUBackend
import JustRelax: PTStokesCoeffs
import JustRelax:
    AbstractBoundaryConditions,
    TemperatureBoundaryConditions,
    AbstractFlowBoundaryConditions,
    DisplacementBoundaryConditions,
    VelocityBoundaryConditions,
    apply_dirichlet,
    apply_dirichlet!
import JustRelax: normal_stress, shear_stress, shear_vorticity

import JustPIC._3D: numphases, nphases

@init_parallel_stencil(Threads, Float64, 3)

include("common.jl")
include("stokes/Stokes3D.jl")
export solve!

end

module JustRelax2D_AD

using ..JustRelax
using JustPIC, JustPIC._2D
using StaticArrays
using CellArrays
using ParallelStencil, ParallelStencil.FiniteDifferences2D
using ImplicitGlobalGrid
using GeoParams, LinearAlgebra, Printf
using Statistics
using MPI
using Enzyme

import JustRelax: IGG, BackendTrait, CPUBackendTrait, backend, CPUBackend
import JustRelax: PTStokesCoeffs
import JustRelax:
    AbstractBoundaryConditions,
    TemperatureBoundaryConditions,
    AbstractFlowBoundaryConditions,
    DisplacementBoundaryConditions,
    VelocityBoundaryConditions,
    apply_dirichlet,
    apply_dirichlet!

import JustRelax: normal_stress, shear_stress, shear_vorticity

import JustPIC._2D: numphases, nphases

@init_parallel_stencil(Threads, Float64, 2)

include("common.jl")
include("stokes/Stokes2D.jl")
export solve!
include("adjoint/Adjoint_Stokes2D.jl")
export adjoint_solve!
include("adjoint/Adjoint_VelocityKernels.jl")
export update_V!
include("adjoint/AdjointSolve.jl")
export adjoint_2D!
include("adjoint/AdjointSensitivities.jl")
calc_sensitivity_2D!

end
