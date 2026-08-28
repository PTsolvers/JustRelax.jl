push!(LOAD_PATH, "..")

using Test
using LinearAlgebra
using JustRelax, JustRelax.JustRelax2D
import JustRelax.JustRelax2D as JR2

"""Return the interior x-face pressure gradient used by the 2D layout."""
function _pressure_gradient_x(p, wₚ, wᵤ, dx)
    Base.require_one_based_indexing(p, wₚ, wᵤ)
    nx, ny = size(p)
    gx = zeros(promote_type(eltype(p), eltype(wₚ), eltype(wᵤ)), nx - 1, ny)
    for j in 1:ny, i in 1:(nx - 1)
        # The padded production indices are P[i, j] and Vx[i + 1, j + 1],
        # while the corresponding masks live at center[i, j] and Vx[i + 1, j].
        active = wᵤ[i + 1, j] > 0
        gx[i, j] = active * dx * (-wₚ[i, j] * p[i, j] + wₚ[i + 1, j] * p[i + 1, j])
    end
    return gx
end

"""Return the interior y-face pressure gradient used by the 2D layout."""
function _pressure_gradient_y(p, wₚ, wᵥ, dy)
    Base.require_one_based_indexing(p, wₚ, wᵥ)
    nx, ny = size(p)
    gy = zeros(promote_type(eltype(p), eltype(wₚ), eltype(wᵥ)), nx, ny - 1)
    for j in 1:(ny - 1), i in 1:nx
        # The padded production indices are P[i, j] and Vy[i + 1, j + 1],
        # while the corresponding mask is Vy[i, j + 1].
        active = wᵥ[i, j + 1] > 0
        gy[i, j] = active * dy * (-wₚ[i, j] * p[i, j] + wₚ[i, j + 1] * p[i, j + 1])
    end
    return gy
end

function _dense_gradient_x(wₚ, wᵤ, dx)
    Base.require_one_based_indexing(wₚ, wᵤ)
    nx, ny = size(wₚ)
    G = zeros(eltype(wₚ), (nx - 1) * ny, nx * ny)
    row(i, j) = i + (j - 1) * (nx - 1)
    col(i, j) = i + (j - 1) * nx
    for j in 1:ny, i in 1:(nx - 1)
        r = row(i, j)
        active = wᵤ[i + 1, j] > 0
        G[r, col(i, j)] = -active * dx * wₚ[i, j]
        G[r, col(i + 1, j)] = active * dx * wₚ[i + 1, j]
    end
    return G
end

function _dense_gradient_y(wₚ, wᵥ, dy)
    Base.require_one_based_indexing(wₚ, wᵥ)
    nx, ny = size(wₚ)
    G = zeros(eltype(wₚ), nx * (ny - 1), nx * ny)
    row(i, j) = i + (j - 1) * nx
    col(i, j) = i + (j - 1) * nx
    for j in 1:(ny - 1), i in 1:nx
        r = row(i, j)
        active = wᵥ[i, j + 1] > 0
        G[r, col(i, j)] = -active * dy * wₚ[i, j]
        G[r, col(i, j + 1)] = active * dy * wₚ[i, j + 1]
    end
    return G
end

@testset "Variational Stokes 2D operator maps" begin
    nx, ny = 3, 2
    dx, dy = 0.7, 1.3
    p = [0.2 -1.1; 0.8 2.4; -0.5 0.6]
    wₚ = [1.0 0.4; 0.7 0.0; 0.3 0.9]
    wᵤ = [0.0 0.0; 0.6 0.8; 0.2 0.5; 0.0 0.0]
    wᵥ = [0.0 0.5 0.0; 0.3 0.0 0.7; 0.0 0.4 0.0]

    gx = _pressure_gradient_x(p, wₚ, wᵤ, dx)
    gy = _pressure_gradient_y(p, wₚ, wᵥ, dy)
    @test vec(gx) ≈ _dense_gradient_x(wₚ, wᵤ, dx) * vec(p)
    @test vec(gy) ≈ _dense_gradient_y(wₚ, wᵥ, dy) * vec(p)

    # The same explicit maps define the transpose action used by a coupled
    # pressure/velocity derivation; this catches row/column ordering errors.
    qx = [-0.3 1.2; 0.4 -0.8]
    qy = [0.5; -0.2; 1.1]
    Gx = _dense_gradient_x(wₚ, wᵤ, dx)
    Gy = _dense_gradient_y(wₚ, wᵥ, dy)
    @test dot(vec(gx), vec(qx)) ≈ dot(vec(p), Gx' * vec(qx))
    @test dot(vec(gy), vec(qy)) ≈ dot(vec(p), Gy' * vec(qy))
end

@testset "Variational Stokes 2D active-volume policy" begin
    @test JR2.variational_active(0.0) == false
    @test JR2.variational_active(nextfloat(0.0)) == true
    @test JR2.variational_active(1.0) == true
end

@testset "Variational Stokes 2D bounded face mass" begin
    for fraction in (0.0, nextfloat(0.0), 0.05, 0.1, 0.75, 1.0)
        mass = JR2.variational_face_mass(fraction)
        @test isfinite(mass)
        @test mass ≥ 0.1
    end
    @test JR2.variational_face_mass(1.0) == 1.0
end

@testset "Marker-chain filtering index safety" begin
    find_cells = JR2.find_minmax_cell_indices

    # Cell lookup must floor negative coordinates rather than truncate them
    # toward zero.
    @test find_cells([-0.2, -1.2], 0.0, (1.0, 1.0)) == (-1, 0)

    # An empty marker column must produce an empty interval without attempting
    # to convert infinities to integer indices.
    @test find_cells([NaN, NaN], 0.0, (1.0, 1.0)) == (1, 0)
end
