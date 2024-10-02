## Weno5 advection scheme. Implementation based on the repository from
# https://gmd.copernicus.org/preprints/gmd-2023-189/

abstract type AbstractWENO end

"""
    WENO5{T, N, A, M} <: AbstractWENO

The `WENO5` is a structure representing the Weighted Essentially Non-Oscillatory (WENO) scheme of order 5 for the solution of hyperbolic partial differential equations.

# Fields
- `d0L`, `d1L`, `d2L`: Upwind constants.
- `d0R`, `d1R`, `d2R`: Downwind constants.
- `c1`, `c2`: Constants for betas.
- `sc1`, `sc2`, `sc3`, `sc4`, `sc5`: Stencil weights.
- `ϵ`: Tolerance.
- `ni`: Grid size.
- `ut`: Intermediate buffer array.
- `fL`, `fR`, `fB`, `fT`: Fluxes.
- `method`: Method (1:JS, 2:Z).

# Examples
```julia
weno = WENO5(Val(2), (nx, ny))
```

# Description
The `WENO5` structure contains the parameters and temporary variables used in the WENO scheme. These include the upwind and downwind constants, the constants for betas, the stencil candidate weights, the tolerance, the grid size, the fluxes, and the method.
"""
# Define the WENO5 struct
struct WENO5{T,N,A,M} <: AbstractWENO
    # upwind constants
    d0L::T
    d1L::T
    d2L::T
    # downwind constants
    d0R::T
    d1R::T
    d2R::T
    # for betas
    c1::T
    c2::T
    # stencil weights
    sc1::T
    sc2::T
    sc3::T
    sc4::T
    sc5::T
    # tolerance
    ϵ::T
    # grid size
    ni::NTuple{N,Int64}
    # fluxes
    ut::A
    fL::A
    fR::A
    fB::A
    fT::A
    # method
    method::M # 1:JS, 2:Z
end
