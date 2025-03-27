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
export adjoint_solve!, adjoint_solveDot!
include("adjoint/Adjoint_VelocityKernels.jl")
export update_V!, compute_strain_rateAD!
include("adjoint/AdjointSolve.jl")
export adjoint_2D!
include("adjoint/AdjointSensitivities.jl")
export calc_sensitivity_2D!
include("adjoint/Adjoint_PressureKernel.jl")
export compute_P_kernelAD!, update_PAD!, ana_P!
include("adjoint/AdjointStressKernels.jl")
export update_stresses_center_vertex_psAD! 
export update_stresses_center_vertex_psADSens! 
export assemble_parameter_matrices!

end
