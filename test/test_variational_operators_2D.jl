using JustRelax
using LinearAlgebra
using Test

@testset "Variational Stokes legacy keyword bundle" begin
    legacy = (; nout = 2000, free_surface = true)
    options = JustRelax.JustRelax2D._variational_stokes_options((; kwargs = legacy, nout = 10))
    @test options == (; nout = 10, free_surface = true)
end

"""Return the interior x-face pressure gradient used by the 2D layout."""
function _pressure_gradient_x(p, wₚ, wᵤ, dx)
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

"""Assemble the 2D staggered symmetric-gradient map with zero boundary velocity."""
function _dense_deformation(nx, ny, dx, dy)
    nvx = (nx - 1) * ny
    nvy = nx * (ny - 1)
    nc = nx * ny
    nv = (nx - 1) * (ny - 1)
    D = zeros(2 * nc + nv, nvx + nvy)
    vx(i, j) = i + (j - 1) * (nx - 1)
    vy(i, j) = nvx + i + (j - 1) * nx
    cell(i, j) = i + (j - 1) * nx
    vertex(i, j) = 2 * nc + i + (j - 1) * (nx - 1)

    for j in 1:ny, i in 1:nx
        rxx = cell(i, j)
        ryy = nc + rxx
        i > 1 && (D[rxx, vx(i - 1, j)] -= dx)
        i < nx && (D[rxx, vx(i, j)] += dx)
        j > 1 && (D[ryy, vy(i, j - 1)] -= dy)
        j < ny && (D[ryy, vy(i, j)] += dy)
    end
    for j in 1:(ny - 1), i in 1:(nx - 1)
        r = vertex(i, j)
        D[r, vx(i, j)] -= 0.5 * dy
        D[r, vx(i, j + 1)] += 0.5 * dy
        D[r, vy(i, j)] -= 0.5 * dx
        D[r, vy(i + 1, j)] += 0.5 * dx
    end
    return D
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

@testset "Variational Stokes 2D deformation adjoint" begin
    nx, ny = 3, 3
    D = _dense_deformation(nx, ny, 0.7, 1.3)
    u = collect(range(-0.8, 1.1; length = size(D, 2)))
    τ = collect(range(0.2, 1.4; length = size(D, 1)))
    w_center = [1.0 0.4 0.0; 0.7 0.2 0.9; 0.3 0.8 0.6]
    w_vertex = [0.5 0.0; 0.25 0.9]
    # Tensor contraction counts the symmetric xy component twice.
    Wτ = vcat(vec(w_center), vec(w_center), 2 .* vec(w_vertex))

    @test dot(D * u, Wτ .* τ) ≈ dot(u, D' * (Wτ .* τ))
    @test all(iszero, (Wτ .* τ)[Wτ .== 0])
end

@testset "Variational Stokes 2D rigid-body modes" begin
    nx, ny = 4, 3
    dx, dy = 0.7, 1.3
    xfaces = (0:nx) .* dx
    yfaces = (0:ny) .* dy
    xcenters = ((0:(nx - 1)) .+ 0.5) .* dx
    ycenters = ((0:(ny - 1)) .+ 0.5) .* dy

    function strain(Vx, Vy)
        εxx = [(Vx[i + 1, j] - Vx[i, j]) / dx for i in 1:nx, j in 1:ny]
        εyy = [(Vy[i, j + 1] - Vy[i, j]) / dy for i in 1:nx, j in 1:ny]
        εxy = [
            0.5 * (
                    (Vx[i + 1, j + 1] - Vx[i + 1, j]) / dy +
                    (Vy[i + 1, j + 1] - Vy[i, j + 1]) / dx
                ) for i in 1:(nx - 1), j in 1:(ny - 1)
        ]
        return εxx, εyy, εxy
    end

    translation = strain(fill(2.0, nx + 1, ny), fill(-3.0, nx, ny + 1))
    @test all(all(iszero, ε) for ε in translation)

    ω = 1.7
    Vx = [-ω * y for _ in xfaces, y in ycenters]
    Vy = [ω * x for x in xcenters, _ in yfaces]
    rotation = strain(Vx, Vy)
    @test all(all(x -> isapprox(x, 0.0; atol = 10eps()), ε) for ε in rotation)
end

@testset "Variational Stokes 2D active-volume policy" begin
    @test JustRelax.JustRelax2D.variational_active(0.0) == false
    @test JustRelax.JustRelax2D.variational_active(nextfloat(0.0)) == true
    @test JustRelax.JustRelax2D.variational_active(1.0) == true

    ϕ = JustRelax.JustRelax2D.RockRatio(JustRelax.CPUBackend, (3, 3))
    fill!(ϕ.center, 0.0)
    fill!(ϕ.Vy, 0.0)
    ϕ.center[2, 2] = 1.0
    ϕ.Vy[2, 3] = 0.25
    @test JustRelax.JustRelax2D.isvalid_vy(ϕ, 2, 3)
end

@testset "Variational Stokes 2D bounded face mass" begin
    for fraction in (0.0, nextfloat(0.0), 0.05, 0.1, 0.75, 1.0)
        mass = JustRelax.JustRelax2D.variational_face_mass(fraction)
        @test isfinite(mass)
        @test mass ≥ 0.1
    end
    @test JustRelax.JustRelax2D.variational_face_mass(1.0) == 1.0
end

@testset "Variational DYREL weighted momentum diagonal" begin
    D = zeros(1, 1)
    λmax = zeros(1, 1)
    store! = JustRelax.JustRelax2D.set_preconditioner!

    store!(D, λmax, 4.0, 12.0, 0.05, 1, 1)
    @test D[1, 1] == 0.4
    @test λmax[1, 1] ≈ 30.0

    store!(D, λmax, 4.0, 12.0, 1.0, 1, 1)
    @test D[1, 1] == 4.0
    @test λmax[1, 1] ≈ 3.0
end

@testset "Variational DYREL weighted pressure residual" begin
    weighted_divergence = JustRelax.JustRelax2D.variational_pressure_divergence
    @test weighted_divergence(2.0, 1.0) == 2.0
    @test weighted_divergence(2.0, 0.25) == 0.5
    @test weighted_divergence(2.0, 0.0) == 0.0
end

@testset "Marker-chain filtering index safety" begin
    find_cells = JustRelax.JustRelax2D.find_minmax_cell_indices

    # Cell lookup must floor negative coordinates rather than truncate them
    # toward zero.
    @test find_cells([-0.2, -1.2], 0.0, (1.0, 1.0)) == (-1, 0)

    # An empty marker column must produce an empty interval without attempting
    # to convert infinities to integer indices.
    @test find_cells([NaN, NaN], 0.0, (1.0, 1.0)) == (1, 0)
end
