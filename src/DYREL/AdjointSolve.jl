include("Adjointkernels.jl")

using LinearAlgebra
using Printf

mutable struct DYRELAdjoint2DWork{A, S}
    λVx::A
    λVy::A
    λP::A
    ResλVx::A
    ResλVy::A
    ResλP::A
    λResVx0::A
    λResVy0::A
    λdVxdτ::A
    λdVydτ::A
    λdVx::A
    λdVy::A
    λdτVx::A
    λdτVy::A
    λαVx::A
    λαVy::A
    λβVx::A
    λβVy::A
    λcVx::A
    λcVy::A
    Schurx::A
    Schury::A
    dρgx::A
    dρgy::A
    dstokes::S
    err_evo_V::Vector{Float64}
    err_evo_P::Vector{Float64}
    err_evo_it::Vector{Int}
end
