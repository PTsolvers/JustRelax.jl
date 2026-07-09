"""
    WENO5(method::Val, ni::NTuple{N, Integer})
    WENO5(backend, method::Val, ni::NTuple{N, Integer})

Construct a `WENO5` fifth-order Weighted Essentially Non-Oscillatory advection scheme for a
grid of size `ni`. `method` selects the smoothness weights, `Val(1)` for Jiang-Shu (JS) and
`Val(2)` for the Z variant. The scheme's constants and temporary flux buffers are allocated
for the given backend (`CPUBackend` by default). Pass the result to [`WENO_advection!`](@ref).
"""
function WENO5(::Type{CPUBackend}, method::Val{T}, ni::NTuple{N, Integer}) where {N, T}
    return WENO5(method, tuple(ni...))
end

# Define the WENO5 constructor
function WENO5(method::Val{T}, ni::NTuple{N, Integer}) where {N, T}
    d0L = 1 / 10
    d1L = 3 / 5
    d2L = 3 / 10
    # downwind constants
    d0R = 3 / 10
    d1R = 3 / 5
    d2R = 1 / 10
    # for betas
    c1 = 13 / 12
    c2 = 1 / 4
    # stencil weights
    sc1 = 1 / 3
    sc2 = 7 / 6
    sc3 = 11 / 6
    sc4 = 1 / 6
    sc5 = 5 / 6
    # tolerance
    ϵ = 1.0e-6
    # fluxes
    ut = @zeros(ni...)
    fL = @zeros(ni...)
    fR = @zeros(ni...)
    fB = @zeros(ni...)
    fT = @zeros(ni...)
    # method = Val{1} # 1:JS, 2:Z
    return JustRelax.WENO5(
        d0L,
        d1L,
        d2L,
        d0R,
        d1R,
        d2R,
        c1,
        c2,
        sc1,
        sc2,
        sc3,
        sc4,
        sc5,
        ϵ,
        ni,
        ut,
        fL,
        fR,
        fB,
        fT,
        method,
    )
end
