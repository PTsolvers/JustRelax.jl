using JustRelax
using JustRelax.JustRelax2D
using LinearAlgebra
using ParallelStencil
using Test

@init_parallel_stencil(Threads, Float64, 2)

@testset "Variational Stokes legacy keyword bundle" begin
    flatten = JustRelax.JustRelax2D.flatten_solver_kwargs
    legacy = (; nout = 2000, free_surface = true)
    @test flatten((; kwargs = legacy, nout = 10)) == (; nout = 10, free_surface = true)
    @test flatten((; nout = 10, free_surface = true)) == (; nout = 10, free_surface = true)
    @test_throws "must be a NamedTuple" flatten((; kwargs = 1))
end

@testset "Solver entry point keyword contract" begin
    # The CUDA/AMDGPU extensions declare their backend-trait methods as `(...; kwargs)` and splat
    # inside, so every trait method must take the options as that single bundle, and every public
    # entry point must slurp so it accepts both the bundled and the plain-keyword call form. A
    # generic entry that splats instead reaches the GPU methods with no `kwargs` at all.
    keywords_of(f, trait) = Base.kwarg_decl(only(methods(f, Tuple{trait, Any, Vararg{Any}})))

    for (M, name) in (
            (JustRelax.JustRelax2D, :solve_VariationalStokes!),
            (JustRelax.JustRelax2D, :solve_DYREL!),
            (JustRelax.JustRelax2D, :solve_VariationalDYREL!),
            (JustRelax.JustRelax3D, :solve_VariationalStokes!),
        )
        f = getfield(M, name)
        @test keywords_of(f, JustRelax.CPUBackendTrait) == [:kwargs]
        @test keywords_of(f, JustRelax.StokesArrays) == [Symbol("kwargs...")]
    end
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

    # Anchor the layout above to the kernel it models: with zero stress, zero buoyancy and
    # `dt = 0`, the masked momentum residual is exactly the negated ϕ-weighted pressure gradient.
    ni = nx, ny
    grid = Geometry(ni, (nx / dx, ny / dy))
    ϕ = JustRelax.JustRelax2D.RockRatio(JustRelax.CPUBackend, ni)
    copyto!(ϕ.center, wₚ)
    copyto!(ϕ.Vx, wᵤ)
    copyto!(ϕ.Vy, wᵥ)
    fill!(ϕ.vertex, 1.0)

    stokes = StokesArrays(JustRelax.CPUBackend, ni)
    copyto!(stokes.P, p)
    @parallel (JustRelax.JustRelax2D.@idx ni) JustRelax.JustRelax2D.compute_PH_residual_V!(
        stokes.R.Rx, stokes.R.Ry,
        JustRelax.JustRelax2D.@velocity(stokes)...,
        stokes.P, stokes.ΔPψ,
        JustRelax.JustRelax2D.@stress(stokes)...,
        zeros(ni), zeros(ni),
        ϕ, grid._di.center, grid._di.vertex, 0.0,
    )
    @test stokes.R.Rx ≈ -gx
    @test stokes.R.Ry ≈ -gy
end

@testset "Variational Stokes 2D rigid-body modes" begin
    # A rigid translation and a rigid rotation carry no strain, so the masked strain-rate kernel
    # must annihilate both — for a full rock domain and for one the RockRatio partially masks.
    ni = nx, ny = 4, 3
    grid = Geometry(ni, (nx * 0.7, ny * 1.3))
    xci, xvi = grid.xci, grid.xvi

    function kernel_strain!(stokes, ϕ)
        # compute_strain_rate! reads ∇V rather than computing it
        @parallel (JustRelax.JustRelax2D.@idx ni) JustRelax.JustRelax2D.compute_∇V!(
            stokes.∇V, JustRelax.JustRelax2D.@velocity(stokes), ϕ, grid._di.vertex
        )
        @parallel (JustRelax.JustRelax2D.@idx ni .+ 1) JustRelax.JustRelax2D.compute_strain_rate!(
            JustRelax.JustRelax2D.@strain(stokes)...,
            stokes.∇V,
            JustRelax.JustRelax2D.@velocity(stokes)...,
            ϕ,
            grid._di.vertex,
            grid._di.velocity...,
        )
        return (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy, stokes.∇V)
    end

    full_rock() = let ϕ = JustRelax.JustRelax2D.RockRatio(JustRelax.CPUBackend, ni)
        foreach(f -> fill!(getfield(ϕ, f), 1.0), (:center, :vertex, :Vx, :Vy))
        ϕ
    end
    ϕ_cut = full_rock()
    ϕ_cut.center[1, 1] = 0.0
    ϕ_cut.Vx[1, 1] = 0.0

    ω = 1.7
    for ϕ in (full_rock(), ϕ_cut), mode in (:translation, :rotation)
        stokes = StokesArrays(JustRelax.CPUBackend, ni)
        if mode === :translation
            fill!(stokes.V.Vx, 2.0)
            fill!(stokes.V.Vy, -3.0)
        else
            # Vx carries a ghost row in y only, Vy a ghost column in x only; the ghosts hold the
            # analytic field too, so the whole stencil sees the rigid mode.
            ghost(c) = (d = c[2] - c[1]; vcat(c[1] - d, c..., c[end] + d))
            stokes.V.Vx .= [-ω * y for _ in xvi[1], y in ghost(xci[2])]
            stokes.V.Vy .= [ω * x for x in ghost(xci[1]), _ in xvi[2]]
        end
        for field in kernel_strain!(stokes, ϕ)
            @test all(x -> isapprox(x, 0.0; atol = 1.0e3 * eps()), field)
        end
    end
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

    # A decoupled mask-boundary row — every surrounding stress weight vanishes — is preconditioned
    # with the identity.
    store!(D, λmax, 0.0, 0.0, 1.0, 1, 1)
    @test D[1, 1] == 1.0
    @test λmax[1, 1] == 1.0

    # A row the FSSA term inverts, or one fed a degenerate viscosity, is not silently rescued.
    store!(D, λmax, -2.0, 12.0, 1.0, 1, 1)
    @test isnan(D[1, 1])
    @test isnan(λmax[1, 1])

    store!(D, λmax, NaN, 12.0, 1.0, 1, 1)
    @test isnan(D[1, 1])
    @test isnan(λmax[1, 1])
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
