## Velocity type

function Velocity(nx::Integer, ny::Integer)
    nVx = (nx + 1, ny + 2)
    nVy = (nx + 2, ny + 1)

    Vx, Vy = @zeros(nVx...), @zeros(nVy)
    return JustRelax.Velocity(Vx, Vy, nothing)
end

function Velocity(nx::Integer, ny::Integer, nz::Integer)
    nVx = (nx + 1, ny + 2, nz + 2)
    nVy = (nx + 2, ny + 1, nz + 2)
    nVz = (nx + 2, ny + 2, nz + 1)

    Vx, Vy, Vz = @zeros(nVx...), @zeros(nVy), @zeros(nVz)
    return JustRelax.Velocity(Vx, Vy, Vz)
end

## Displacement type

function Displacement(nx::Integer, ny::Integer)
    nUx = (nx + 1, ny + 2)
    nUy = (nx + 2, ny + 1)

    Ux, Uy = @zeros(nUx...), @zeros(nUy)
    return JustRelax.Displacement(Ux, Uy, nothing)
end

function Displacement(nx::Integer, ny::Integer, nz::Integer)
    nUx = (nx + 1, ny + 2, nz + 2)
    nUy = (nx + 2, ny + 1, nz + 2)
    nUz = (nx + 2, ny + 2, nz + 1)

    Ux, Uy, Uz = @zeros(nUx...), @zeros(nUy), @zeros(nUz)
    return JustRelax.Displacement(Ux, Uy, Uz)
end

## Vorticity type

function Vorticity(nx::Integer, ny::Integer)
    xy = @zeros(nx + 1, ny + 1)

    return JustRelax.Vorticity(nothing, nothing, xy)
end

function Vorticity(nx::Integer, ny::Integer, nz::Integer)
    yz = @zeros(nx, ny + 1, nz + 1)
    xz = @zeros(nx + 1, ny, nz + 1)
    xy = @zeros(nx + 1, ny + 1, nz)

    return JustRelax.Vorticity(yz, xz, xy)
end

## Viscosity type

function Viscosity(ni::NTuple{N, Integer}) where {N}
    η = @ones(ni...)
    η_vep = @ones(ni...)
    ητ = @zeros(ni...)
    return JustRelax.Viscosity(η, η_vep, ητ)
end

## SymmetricTensor type

function SymmetricTensor(nx::Integer, ny::Integer)
    return JustRelax.SymmetricTensor(
        @zeros(nx, ny), # xx
        @zeros(nx, ny), # yy
        @zeros(nx + 1, ny + 1), # xy
        @zeros(nx, ny), # xy @ cell center
        @zeros(nx, ny) # II (second invariant)
    )
end

function SymmetricTensor(nx::Integer, ny::Integer, nz::Integer)
    return JustRelax.SymmetricTensor(
        @zeros(nx, ny, nz), # xx
        @zeros(nx, ny, nz), # yy
        @zeros(nx, ny, nz), # zz
        @zeros(nx + 1, ny + 1, nz), # xy
        @zeros(nx, ny + 1, nz + 1), # yz
        @zeros(nx + 1, ny, nz + 1), # xz
        @zeros(nx, ny, nz), # yz @ cell center
        @zeros(nx, ny, nz), # xz @ cell center
        @zeros(nx, ny, nz), # xy @ cell center
        @zeros(nx, ny, nz), # II (second invariant)
    )
end

## Residual type

function Residual(nx::Integer, ny::Integer)
    Rx = @zeros(nx - 1, ny)
    Ry = @zeros(nx, ny - 1)
    RP = @zeros(nx, ny)
    return JustRelax.Residual(RP, Rx, Ry)
end

function Residual(nx::Integer, ny::Integer, nz::Integer)
    Rx = @zeros(nx - 1, ny, nz)
    Ry = @zeros(nx, ny - 1, nz)
    Rz = @zeros(nx, ny, nz - 1)
    RP = @zeros(nx, ny, nz)
    return JustRelax.Residual(RP, Rx, Ry, Rz)
end

## StokesArrays type
function StokesArrays(::Type{CPUBackend}, ni::NTuple{N, Integer}) where {N}
    return StokesArrays(ni)
end
"""
    StokesArrays(ni::NTuple{N,Integer}) where {N}

Create the Stokes arrays object in 2D or 3D.

## Fields
- `P`: Pressure field
- `P0`: Previous pressure field
- `∇V`: Velocity gradient
- `V`: Velocity fields
- `Q`: Volumetric source/sink term e.g. `ΔV/V_tot [m³/m³]`
- `U`: Displacement fields
- `ω`: Vorticity field
- `τ`: Stress tensors
- `τ_o`: Old stress tensors
- `ε`: Strain rate tensors
- `ε_pl`: Plastic strain rate tensors
- `EII_pl`: Second invariant of the accumulated plastic strain
- `viscosity`: Viscosity fields
- `R`: Residual fields

"""
function StokesArrays(ni::NTuple{N, Integer}) where {N}
    P = @zeros(ni...)
    P0 = @zeros(ni...)
    ∇V = @zeros(ni...)
    V = Velocity(ni...)
    Q = @zeros(ni...) # volumetric source/sink term
    U = Displacement(ni...)
    ω = Vorticity(ni...)
    τ = SymmetricTensor(ni...)
    τ_o = SymmetricTensor(ni...)
    ε = SymmetricTensor(ni...)
    ε_pl = SymmetricTensor(ni...)
    EII_pl = @zeros(ni...)
    viscosity = Viscosity(ni)
    R = Residual(ni...)

    return JustRelax.StokesArrays(P, P0, V, ∇V, Q, τ, ε, ε_pl, EII_pl, viscosity, τ_o, R, U, ω)
end

## StokesArraysAdjoint type
function StokesArraysAdjoint(::Type{CPUBackend}, ni::NTuple{N,Integer}) where {N}
    return StokesArraysAdjoint(ni)
end

function StokesArraysAdjoint(ni::NTuple{N,Integer}) where {N}
    P = @zeros(ni...)
    PA = @zeros(ni...)
    P0 = @zeros(ni...)
    ∇V = @zeros(ni...)
    V = Velocity(ni...)
    VA = Velocity(ni...)
    U = Displacement(ni...)
    ω = Vorticity(ni...)
    τ = SymmetricTensor(ni...)
    dτ = SymmetricTensor(ni...)
    τ_o = SymmetricTensor(ni...)
    ε = SymmetricTensor(ni...)
    ε_pl = SymmetricTensor(ni...)
    EII_pl = @zeros(ni...)
    viscosity = Viscosity(ni)
    R = Residual(ni...)
    η = @zeros(ni...)
    ρ = @zeros(ni...)
    G  = @zeros(ni...)
    fr = @zeros(ni...)
    C = @zeros(ni...)
    K = @zeros(ni...)
    return JustRelax.StokesArraysAdjoint(P, PA, P0, V, VA, ∇V, τ, dτ, ε, ε_pl, EII_pl, viscosity, τ_o, R, U, ω, η, ρ, G, fr, C,K)
end
