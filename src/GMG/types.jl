"""
    GMGOptions(Val(N); coarsening, coarsest_cells = 16, max_levels = 32)

Internal hierarchy controls. Each coarsening direction is 1 or 2; the stopping
target is a cell count per direction. `max_levels` includes the fine level.
Solver and acceleration controls will be added with their implementation gates.
"""
struct GMGOptions{N}
    coarsening::NTuple{N, Int}
    coarsest_cells::NTuple{N, Int}
    max_levels::Int
end

"""Host setup metadata for the global physical domain of one GMG level."""
struct GMGDomain{N, T}
    ni::NTuple{N, Int}
    xvi::NTuple{N, Vector{T}}
    li::NTuple{N, T}
    origin::NTuple{N, T}
end

"""
    GMGPartition(global_ni, owned; periodic, halo, comm = nothing)

Explicit pressure-cell ownership in global, one-based indices. `storage` includes
neighbor overlap; `owned_local` excludes it. A periodic seam has one owner.
The communicator is borrowed; GMG does not initialize or free it. Callers must
supply a nonoverlapping, complete partition and consistent collective arguments.
"""
struct GMGPartition{N, C}
    global_ni::NTuple{N, Int}
    owned::NTuple{N, UnitRange{Int}}
    storage::NTuple{N, UnitRange{Int}}
    owned_local::NTuple{N, UnitRange{Int}}
    periodic::NTuple{N, Bool}
    halo::NTuple{N, Int}
    comm::C
end

"""Separable pressure-cell widths and normal-face dual widths, on the backend."""
struct GMGMeasures{N, V}
    cell::NTuple{N, V}
    face::NTuple{N, V}
end

"""Coupled correction/RHS fields in JR's native pressure and velocity storage."""
struct GMGCoupledFields{N, A}
    p::A
    v::NTuple{N, A}
end

"""One explicit level; numerical scratch never represents physical history."""
struct GMGLevel{G, D, P, M, F}
    grid::G
    domain::D
    partition::P
    measures::M
    correction::F
    rhs::F
    residual::F
end

"""Global parent-cell mapping for a nested level transition."""
struct GMGTransition{N}
    ratio::NTuple{N, Int}
end

"""Concrete homogeneous level storage, with separate adjacent-level transitions."""
struct GMGHierarchy{L, T, O}
    levels::Vector{L}
    transitions::Vector{T}
    options::O
end

"""Concrete GeoParams inputs; material laws and argument arrays remain borrowed."""
struct GMGMaterialContext{R, A, P}
    rheology::R
    args::A
    phase_ratios::P
end

@inline backend(fields::GMGCoupledFields) = backend(fields.p)
@inline backend(level::GMGLevel) = backend(level.correction)
