## Velocity type
"""
    Velocity(nx::Integer, ny::Integer)

Create the velocity arrays for the Stokes solver in 2D.
## Fields
- `Vx`: Velocity in x direction (nx + 1, ny + 2)
- `Vy`: Velocity in y direction (nx + 2, ny + 1)
"""
function Velocity(nx::Integer, ny::Integer)
    nVx = (nx + 1, ny + 2)
    nVy = (nx + 2, ny + 1)

    Vx, Vy = @zeros(nVx...), @zeros(nVy)
    return JustRelax.Velocity(Vx, Vy, nothing)
end

"""
    Velocity(nx::Integer, ny::Integer, nz::Integer)

Create the velocity arrays for the Stokes solver in 3D.
## Fields
- `Vx`: Velocity in x direction (nx + 1, ny + 2, nz + 2)
- `Vy`: Velocity in y direction (nx + 2, ny + 1, nz + 2)
- `Vz`: Velocity in z direction (nx + 2, ny + 2, nz + 1)
"""
function Velocity(nx::Integer, ny::Integer, nz::Integer)
    nVx = (nx + 1, ny + 2, nz + 2)
    nVy = (nx + 2, ny + 1, nz + 2)
    nVz = (nx + 2, ny + 2, nz + 1)

    Vx, Vy, Vz = @zeros(nVx...), @zeros(nVy), @zeros(nVz)
    return JustRelax.Velocity(Vx, Vy, Vz)
end

## Displacement type
"""
    Displacement(nx::Integer, ny::Integer)

Create the displacement arrays for the Stokes solver in 2D.
## Fields
- `Ux`: Displacement in x direction at their staggered location
- `Uy`: Displacement in y direction at their staggered location
"""
function Displacement(nx::Integer, ny::Integer)
    nUx = (nx + 1, ny + 2)
    nUy = (nx + 2, ny + 1)

    Ux, Uy = @zeros(nUx...), @zeros(nUy)
    return JustRelax.Displacement(Ux, Uy, nothing)
end

"""
    Displacement(nx::Integer, ny::Integer, nz::Integer)

Create the displacement arrays for the Stokes solver in 3D.
## Fields
- `Ux`: Displacement in x direction at their staggered location
- `Uy`: Displacement in y direction at their staggered location
- `Uz`: Displacement in z direction at their staggered location
"""
function Displacement(nx::Integer, ny::Integer, nz::Integer)
    nUx = (nx + 1, ny + 2, nz + 2)
    nUy = (nx + 2, ny + 1, nz + 2)
    nUz = (nx + 2, ny + 2, nz + 1)

    Ux, Uy, Uz = @zeros(nUx...), @zeros(nUy), @zeros(nUz)
    return JustRelax.Displacement(Ux, Uy, Uz)
end

## Vorticity type
"""
    Vorticity(nx::Integer, ny::Integer)

Create the vorticity arrays for the Stokes solver in 2D.
## Fields
- `xy`: Vorticity component xy at vertices
"""
function Vorticity(nx::Integer, ny::Integer)
    xy = @zeros(nx + 1, ny + 1)

    return JustRelax.Vorticity(nothing, nothing, xy)
end

"""
    Vorticity(nx::Integer, ny::Integer, nz::Integer)

Create the vorticity arrays for the Stokes solver in 3D.
## Fields
- `yz`: Vorticity component yz at their staggered location
- `xz`: Vorticity component xz at their staggered location
- `xy`: Vorticity component xy at their staggered location
"""
function Vorticity(nx::Integer, ny::Integer, nz::Integer)
    yz = @zeros(nx, ny + 1, nz + 1)
    xz = @zeros(nx + 1, ny, nz + 1)
    xy = @zeros(nx + 1, ny + 1, nz)

    return JustRelax.Vorticity(yz, xz, xy)
end

## Viscosity type
"""
    Viscosity(ni::NTuple{N, Integer}) where {N}

Create the viscosity arrays for the Stokes solver in 2D or 3D with the extents given by ni (`nx x ny` or `nx x ny x nz``).
## Fields
- `η`: Viscosity at cell centers
- `ηv`: Viscosity at vertices
- `η_vep`: Viscosity for visco-elastic-plastic rheology
- `ητ`: Pseudo-transient viscosity for stress update
"""
function Viscosity(ni::NTuple{N, Integer}) where {N}
    η = @ones(ni...)
    ηv = @ones(ni .+ 1...)
    η_vep = @ones(ni...)
    ητ = @zeros(ni...)
    return JustRelax.Viscosity(η, ηv, η_vep, ητ)
end

# Principal stress
"""
    PrincipalStress(ni::NTuple{N, Integer}) where {N}

Create the principal stress arrays for the Stokes solver in 2D or 3D with the extents given by ni (`nx x ny` or `nx x ny x nz``).
## Fields
- `σ1`: First principal stress
- `σ2`: Second principal stress
- `σ3`: Third principal stress (only in 3D). In 2D it is a placeholder array of size (2, 1, 1).
"""
function PrincipalStress(::Type{CPUBackend}, ni::NTuple{N, Integer}) where {N}
    return PrincipalStress(ni)
end

function PrincipalStress(ni::NTuple{2, Integer})
    σ1 = @zeros(2, ni...)
    σ2 = @zeros(2, ni...)
    σ3 = @zeros(2, 1, 1)
    return JustRelax.PrincipalStress(σ1, σ2, σ3)
end

function PrincipalStress(ni::NTuple{3, Integer})
    σ1 = @zeros(3, ni...)
    σ2 = @zeros(3, ni...)
    σ3 = @zeros(3, ni...)
    return JustRelax.PrincipalStress(σ1, σ2, σ3)
end

## SymmetricTensor type

"""
    SymmetricTensor(nx::Integer, ny::Integer)

Create the symmetric tensor arrays for the Stokes solver in 2D.
## Fields
- `xx`: xx component of the tensor at cell centers
- `yy`: yy component of the tensor at cell centers
- `xx_v`: xx component of the tensor at vertices
- `yy_v`: yy component of the tensor at vertices
- `xy`: xy component of the tensor at vertices
- `xy_c`: xy component of the tensor at cell centers
- `II`: second invariant of the tensor at cell centers
"""
function SymmetricTensor(nx::Integer, ny::Integer)
    return JustRelax.SymmetricTensor(
        @zeros(nx, ny), # xx
        @zeros(nx, ny), # yy
        @zeros(nx + 1, ny + 1), # xx_v
        @zeros(nx + 1, ny + 1), # yy_v
        @zeros(nx + 1, ny + 1), # xy
        @zeros(nx, ny), # xy @ cell center
        @zeros(nx, ny) # II (second invariant)
    )
end

"""
    SymmetricTensor(nx::Integer, ny::Integer, nz::Integer)

Create the symmetric tensor arrays for the Stokes solver in 3D.

## Fields
- `xx`: xx component of the tensor at cell centers
- `yy`: yy component of the tensor at cell centers
- `zz`: zz component of the tensor at cell centers
- `xx_v`: xx component of the tensor at vertices
- `yy_v`: yy component of the tensor at vertices
- `zz_v`: zz component of the tensor at vertices
- `xy`: xy component of the tensor at vertices
- `yz`: yz component of the tensor at vertices
- `xz`: xz component of the tensor at vertices
- `yz_c`: yz component of the tensor at cell centers
- `xz_c`: xz component of the tensor at cell centers
- `xy_c`: xy component of the tensor at cell centers
- `II`: second invariant of the tensor at cell centers
"""
function SymmetricTensor(nx::Integer, ny::Integer, nz::Integer)
    return JustRelax.SymmetricTensor(
        @zeros(nx, ny, nz), # xx
        @zeros(nx, ny, nz), # yy
        @zeros(nx, ny, nz), # zz
        @zeros(nx + 1, ny + 1, nz + 1), # xx_v
        @zeros(nx + 1, ny + 1, nz + 1), # yy_v
        @zeros(nx + 1, ny + 1, nz + 1), # zz_v
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
"""
    Residual(nx::Integer, ny::Integer)

Create the residual arrays for the Stokes solver in 2D.
## Fields
- `Rx`: Residual for the x-momentum equation
- `Ry`: Residual for the y-momentum equation
- `RP`: Residual for the continuity equation
"""
function Residual(nx::Integer, ny::Integer)
    Rx = @zeros(nx - 1, ny)
    Ry = @zeros(nx, ny - 1)
    RP = @zeros(nx, ny)
    return JustRelax.Residual(RP, Rx, Ry)
end

"""
    Residual(nx::Integer, ny::Integer, nz::Integer)

Create the residual arrays for the Stokes solver in 3D.
## Fields
- `Rx`: Residual for the x-momentum equation
- `Ry`: Residual for the y-momentum equation
- `Rz`: Residual for the z-momentum equation
- `RP`: Residual for the continuity equation
"""
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
- `Δε`: Strain increment tensor
- `∇U`: Displacement gradient
- `λ` : plastic multiplier @ centers
- `λv` : plastic multiplier @ vertices
- `ΔPψ` : pressure correction in dilatant case
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
    Δε = SymmetricTensor(ni...)
    ∇U = @zeros(ni...)
    λ = @zeros(ni...)
    λv = @zeros(ni .+ 1...)
    ΔPψ = @zeros(ni...)

    return JustRelax.StokesArrays(P, P0, V, ∇V, Q, τ, ε, ε_pl, EII_pl, viscosity, τ_o, R, U, ω, Δε, ∇U, λ, λv, ΔPψ)
end
