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
    return JustRelax.Velocity{PTArray}(Vx, Vy, Vz)
end

## Viscosity type

function Viscosity(ni::NTuple{N,Integer}) where {N}
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
        @zeros(nx + 1, ny + 1, nz), # xy
        @zeros(nx, ny, nz), # yy
        @zeros(nx + 1, ny, nz + 1), # xz
        @zeros(nx, ny + 1, nz + 1), # yz
        @zeros(nx, ny, nz), # zz
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
function StokesArrays(ni::NTuple{N,Integer}) where {N}
    P = @zeros(ni...)
    P0 = @zeros(ni...)
    ∇V = @zeros(ni...)
    V = Velocity(ni...)
    τ = SymmetricTensor(ni...)
    τ_o = SymmetricTensor(ni...)
    ε = SymmetricTensor(ni...)
    ε_pl = SymmetricTensor(ni...)
    EII_pl = @zeros(ni...)
    viscosity = Viscosity(ni)
    R = Residual(ni...)

    return JustRelax.StokesArrays(P, P0, V, ∇V, τ, ε, ε_pl, EII_pl, viscosity, τ_o, R)
end
