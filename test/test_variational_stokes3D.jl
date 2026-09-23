@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax3D
using ParallelStencil

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 3)
    JustRelax.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 3)
    JustRelax.CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 3)
    JustRelax.CPUBackend
end

using JustPIC
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDA.CUDABackend
else
    JustPIC.CPU
end

import JustRelax.JustRelax3D as JR3
import JustRelax.JustRelax2D as JR2

# The kernels are called from the CPU modules directly, and the 3D variational solver has
# no GPU implementation.
@static if ENV["JULIA_JUSTRELAX_BACKEND"] !== "CPU"
    @info "test_variational_stokes3D.jl: CPU-only; skipped on $(ENV["JULIA_JUSTRELAX_BACKEND"])"
else

    ## Grid and staggered coordinates -------------------------------------------------------

    # Anisotropic spacing, so a swapped axis changes the answer.
    const ni = (4, 5, 6)
    const li = (1.2, 2.0, 3.3)
    const dxi = li ./ ni
    const dx, dy, dz = dxi
    const nx, ny, nz = ni

    xv(i) = (i - 1) * dx
    yv(j) = (j - 1) * dy
    zv(k) = (k - 1) * dz
    xc(i) = (i - 0.5) * dx
    yc(j) = (j - 0.5) * dy
    zc(k) = (k - 0.5) * dz
    # Velocity arrays carry one ghost layer in the transverse directions.
    xg(i) = (i - 1.5) * dx
    yg(j) = (j - 1.5) * dy
    zg(k) = (k - 1.5) * dz

    fill_dev!(A, f) = copyto!(A, [f(I.I...) for I in CartesianIndices(size(A))])

    function fill_velocity!(stokes, fx, fy, fz)
        fill_dev!(stokes.V.Vx, (i, j, k) -> fx(xv(i), yg(j), zg(k)))
        fill_dev!(stokes.V.Vy, (i, j, k) -> fy(xg(i), yv(j), zg(k)))
        fill_dev!(stokes.V.Vz, (i, j, k) -> fz(xg(i), yg(j), zv(k)))
        return nothing
    end

    ## Rock-ratio masks ---------------------------------------------------------------------

    # Liquid volume fraction of a control volume [zlo, zlo + dz] below the free surface `h`.
    below(h, zlo) = clamp((h - zlo) / dz, 0.0, 1.0)

    function full_rock_ratio()
        ϕ = JR3.RockRatio(backend, ni)
        for f in (:center, :vertex, :Vx, :Vy, :Vz, :yz, :xz, :xy)
            getfield(ϕ, f) .= 1.0
        end
        return ϕ
    end

    # Rock below z = (kc + 1/4)dz, air above, plus one eliminated cell centre at (2, 3, 1).
    # Control volumes at cell-centre heights span [(k-1)dz, k dz]; those at vertex heights
    # span [(k-3/2)dz, (k-1/2)dz]. With this surface:
    #   centre-height entries (centre, Vx, Vy, xy): 1 for k ≤ kc, 1/4 at kc+1, 0 above;
    #   vertex-height entries (vertex, Vz, yz, xz): 1 for k ≤ kc, 3/4 at kc+1, 0 above.
    # A pressure row needs its centre and all six faces, so it survives for k ≤ kc
    # (the top face Vz[kc+2] of cell kc+1 is dry). An xy edge needs both vertices it
    # joins along z, so it survives for k ≤ kc. yz and xz edges lie in a vertex plane
    # and need faces from the levels k-1 and k only, so they survive for k ≤ kc+1.
    const kc = 3
    const hole = (2, 3, 1)
    const h_surface = (kc + 0.25) * dz

    function layered_rock_ratio()
        ϕ = JR3.RockRatio(backend, ni)
        centre_level(k) = below(h_surface, (k - 1) * dz)
        vertex_level(k) = below(h_surface, (k - 1.5) * dz)
        for f in (:center, :Vx, :Vy, :xy)
            fill_dev!(getfield(ϕ, f), (i, j, k) -> centre_level(k))
        end
        for f in (:vertex, :Vz, :yz, :xz)
            fill_dev!(getfield(ϕ, f), (i, j, k) -> vertex_level(k))
        end
        c = Array(ϕ.center)
        c[hole...] = 0.0
        copyto!(ϕ.center, c)
        return ϕ
    end

    active_c(i, j, k) = k ≤ kc && (i, j, k) != hole
    active_yz(i, j, k) = k ≤ kc + 1
    active_xz(i, j, k) = k ≤ kc + 1
    active_xy(i, j, k) = k ≤ kc

    ## Phase ratios ---------------------------------------------------------------------------

    @parallel_indices (I...) function set_phase_ratio!(ratios, r1)
        @index ratios[1, I...] = r1[I...]
        @index ratios[2, I...] = 1.0 - r1[I...]
        return nothing
    end

    @parallel_indices (I...) function set_single_phase!(ratios)
        @index ratios[1, I...] = 1.0
        return nothing
    end

    function single_phase_ratios()
        pr = PhaseRatios(backend_JP, 1, ni)
        for f in (:center, :vertex, :Vx, :Vy, :Vz, :yz, :xz, :xy)
            A = getfield(pr, f)
            @parallel (@idx size(A)) set_single_phase!(A)
        end
        return pr
    end

    # Phase-1 fraction given as a function of the entry's index.
    function two_phase_ratios(r1)
        pr = PhaseRatios(backend_JP, 2, ni)
        for f in (:center, :vertex, :Vx, :Vy, :Vz, :yz, :xz, :xy)
            A = getfield(pr, f)
            r = @zeros(size(A)...)
            fill_dev!(r, r1)
            @parallel (@idx size(A)) set_phase_ratio!(A, r)
        end
        return pr
    end

    ## Tensor helpers written out independently of the package -------------------------------

    # √(½(τxx² + τyy² + τzz²) + τyz² + τxz² + τxy²)
    invariant(t) = sqrt(0.5 * (t[1]^2 + t[2]^2 + t[3]^2) + t[4]^2 + t[5]^2 + t[6]^2)
    harmonic(xs...) = length(xs) / sum(inv, xs)
    cl(i, n) = clamp(i, 1, n)

    ## 1. Divergence and strain rate ---------------------------------------------------------

    # Quadratic velocity field: a centred difference of a quadratic is exact at the midpoint,
    # so every staggered entry must equal the analytic derivative *at its own location*.
    ux(x, y, z) = x^2 + 0.5y^2 + 0.3y * z + 0.2z^2
    uy(x, y, z) = 0.6x^2 + 0.7x * z - 0.4y^2 + 0.25z^2
    uz(x, y, z) = 0.35x^2 - 0.3x * y + 0.8y^2 + 0.5z^2
    divu(x, y, z) = 2x - 0.8y + z
    exx(x, y, z) = 2x - divu(x, y, z) / 3
    eyy(x, y, z) = -0.8y - divu(x, y, z) / 3
    ezz(x, y, z) = z - divu(x, y, z) / 3
    eyz(x, y, z) = 0.5 * ((0.7x + 0.5z) + (-0.3x + 1.6y))
    exz(x, y, z) = 0.5 * ((0.3y + 0.4z) + (0.7x - 0.3y))
    exy(x, y, z) = 0.5 * ((y + 0.3z) + (1.2x + 0.7z))

    function strain_rate!(stokes, ϕ)
        _di = inv.(dxi)
        @parallel (@idx ni) JR3.compute_∇V!(stokes.∇V, @velocity(stokes), ϕ, _di)
        @parallel (@idx ni .+ 1) JR3.compute_strain_rate!(
            stokes.∇V, @strain(stokes)..., @velocity(stokes)..., ϕ, _di
        )
        return nothing
    end

    @testset "3D masked divergence and strain rate" begin
        @testset "rigid-body motion is strain free" begin
            stokes = StokesArrays(backend, ni)
            U = (0.3, -1.1, 0.7)
            Ω = (0.4, -0.9, 1.3)
            # V = U + Ω × x
            fill_velocity!(
                stokes,
                (x, y, z) -> U[1] + Ω[2] * z - Ω[3] * y,
                (x, y, z) -> U[2] + Ω[3] * x - Ω[1] * z,
                (x, y, z) -> U[3] + Ω[1] * y - Ω[2] * x,
            )
            strain_rate!(stokes, full_rock_ratio())
            # the field itself is far from zero, so a zero strain rate is not trivial
            @test maximum(abs, Array(stokes.V.Vx)) > 1
            for A in (stokes.∇V, @strain(stokes)...)
                @test maximum(abs, Array(A)) < 1.0e-12
            end
        end

        @testset "quadratic field, full rock" begin
            stokes = StokesArrays(backend, ni)
            fill_velocity!(stokes, ux, uy, uz)
            strain_rate!(stokes, full_rock_ratio())
            ∇V = Array(stokes.∇V)
            εxx, εyy, εzz = Array(stokes.ε.xx), Array(stokes.ε.yy), Array(stokes.ε.zz)
            εyz, εxz, εxy = Array(stokes.ε.yz), Array(stokes.ε.xz), Array(stokes.ε.xy)
            for I in CartesianIndices(∇V)
                i, j, k = I.I
                x, y, z = xc(i), yc(j), zc(k)
                @test ∇V[I] ≈ divu(x, y, z) atol = 1.0e-12
                @test εxx[I] ≈ exx(x, y, z) atol = 1.0e-12
                @test εyy[I] ≈ eyy(x, y, z) atol = 1.0e-12
                @test εzz[I] ≈ ezz(x, y, z) atol = 1.0e-12
            end
            # yz edges sit at (xc, yv, zv), xz at (xv, yc, zv), xy at (xv, yv, zc)
            @test all(
                isapprox(εyz[I], eyz(xc(I[1]), yv(I[2]), zv(I[3])); atol = 1.0e-12)
                    for I in CartesianIndices(εyz)
            )
            @test all(
                isapprox(εxz[I], exz(xv(I[1]), yc(I[2]), zv(I[3])); atol = 1.0e-12)
                    for I in CartesianIndices(εxz)
            )
            @test all(
                isapprox(εxy[I], exy(xv(I[1]), yv(I[2]), zc(I[3])); atol = 1.0e-12)
                    for I in CartesianIndices(εxy)
            )
        end

        @testset "quadratic field, air above the surface" begin
            stokes = StokesArrays(backend, ni)
            fill_velocity!(stokes, ux, uy, uz)
            sentinel = 7.0
            for A in (stokes.∇V, @strain(stokes)...)
                A .= sentinel
            end
            strain_rate!(stokes, layered_rock_ratio())

            ∇V = Array(stokes.∇V)
            εxx, εyy, εzz = Array(stokes.ε.xx), Array(stokes.ε.yy), Array(stokes.ε.zz)
            εyz, εxz, εxy = Array(stokes.ε.yz), Array(stokes.ε.xz), Array(stokes.ε.xy)
            live = [I for I in CartesianIndices(∇V) if active_c(I.I...)]
            dead = [I for I in CartesianIndices(∇V) if !active_c(I.I...)]
            @test length(live) == nx * ny * kc - 1

            # active rows are unscaled by the (partial) rock fraction
            @test all(isapprox(∇V[I], divu(xc(I[1]), yc(I[2]), zc(I[3])); atol = 1.0e-12) for I in live)
            @test all(isapprox(εxx[I], exx(xc(I[1]), yc(I[2]), zc(I[3])); atol = 1.0e-12) for I in live)
            @test all(
                isapprox(εyz[I], eyz(xc(I[1]), yv(I[2]), zv(I[3])); atol = 1.0e-12)
                    for I in CartesianIndices(εyz) if active_yz(I.I...)
            )
            @test all(
                isapprox(εxz[I], exz(xv(I[1]), yc(I[2]), zv(I[3])); atol = 1.0e-12)
                    for I in CartesianIndices(εxz) if active_xz(I.I...)
            )
            @test all(
                isapprox(εxy[I], exy(xv(I[1]), yv(I[2]), zc(I[3])); atol = 1.0e-12)
                    for I in CartesianIndices(εxy) if active_xy(I.I...)
            )
            # the eliminated continuity rows are zero
            @test all(iszero(∇V[I]) for I in dead)
            # Eliminated strain-rate entries are zeroed, so no stale value leaks into the
            # vertex averages of the stress update.
            @test all(iszero(εxx[I]) for I in dead)
            @test all(iszero(εyy[I]) for I in dead)
            @test all(iszero(εzz[I]) for I in dead)
            @test all(iszero(εyz[I]) for I in CartesianIndices(εyz) if !active_yz(I.I...))
            @test all(iszero(εxz[I]) for I in CartesianIndices(εxz) if !active_xz(I.I...))
            @test all(iszero(εxy[I]) for I in CartesianIndices(εxy) if !active_xy(I.I...))
        end
    end

    ## 2. Rock-ratio bookkeeping ---------------------------------------------------------------

    @testset "3D rock ratio from phase ratios" begin
        # phase 2 is air; its fraction varies with all three indices
        air(i, j, k) = mod(i + 2j + 3k, 7) / 6
        pr = two_phase_ratios((i, j, k) -> 1 - air(i, j, k))
        ϕ = JR3.RockRatio(backend, ni)
        update_rock_ratio!(ϕ, pr, 2)
        for f in (:center, :vertex, :Vx, :Vy, :Vz, :yz, :xz, :xy)
            A = Array(getfield(ϕ, f))
            @test all(isapprox(A[I], 1 - air(I.I...); atol = 1.0e-14) for I in CartesianIndices(A))
        end

        @test all(
            JR3.compute_air_ratio(pr.center, 2, I.I...) ≈ air(I.I...) for I in CartesianIndices(ni)
        )

        # a trace of rock below 1e-5 counts as no rock at all
        pr_trace = two_phase_ratios((i, j, k) -> i == 1 ? 1 - 5.0e-6 : 0.5)
        update_rock_ratio!(ϕ, pr_trace, 1)
        c = Array(ϕ.center)
        @test all(iszero, c[1, :, :])
        @test all(≈(0.5), c[2:end, :, :])

        # without an air phase everything is rock
        update_rock_ratio!(ϕ, pr, 0)
        @test all(isone, Array(ϕ.vertex))
        @test all(isone, Array(ϕ.xy))
    end

    @testset "3D null-space rules" begin
        ϕ = layered_rock_ratio()
        for I in CartesianIndices(ϕ.center)
            @test JR3.isvalid_c(ϕ, I.I...) == active_c(I.I...)
        end
        @test all(JR3.isvalid_yz(ϕ, I.I...) == active_yz(I.I...) for I in CartesianIndices(ϕ.yz))
        @test all(JR3.isvalid_xz(ϕ, I.I...) == active_xz(I.I...) for I in CartesianIndices(ϕ.xz))
        @test all(JR3.isvalid_xy(ϕ, I.I...) == active_xy(I.I...) for I in CartesianIndices(ϕ.xy))
        # a velocity face is kept as long as its own control volume holds liquid
        @test all(JR3.isvalid_vx(ϕ, I.I...) == (I[3] ≤ kc + 1) for I in CartesianIndices(ϕ.Vx))
        @test all(JR3.isvalid_vy(ϕ, I.I...) == (I[3] ≤ kc + 1) for I in CartesianIndices(ϕ.Vy))
        @test all(JR3.isvalid_vz(ϕ, I.I...) == (I[3] ≤ kc + 1) for I in CartesianIndices(ϕ.Vz))
        # a vertex needs its own fraction and wet edges on both sides of it
        @test all(
            JR3.isvalid_v(ϕ, I.I...) == (I[3] ≤ kc + 1) for I in CartesianIndices(ϕ.vertex)
        )
        # all three faces sharing index (i, j, k) must be wet
        @test all(
            JR3.isvalid_velocity(ϕ, i, j, k) == (k ≤ kc + 1) for i in 1:nx, j in 1:ny, k in 1:nz
        )
    end

    ## 3. Pressure ---------------------------------------------------------------------------

    @testset "3D masked pressure update" begin
        ϕ = layered_rock_ratio()
        # two phases with different moduli and thermal expansivities, mixed non-uniformly;
        # phase 1 blends a solid and a melt expansivity by the melt fraction
        r1(i, j, k) = 0.2 + 0.6 * mod(i + j + k, 3) / 2
        pr = two_phase_ratios(r1)
        K1, K2, G1, G2 = 3.0, 8.0, 1.5, 0.5
        α1s, α1m, α2 = 2.0e-2, 9.0e-2, 5.0e-2
        el1 = ConstantElasticity(; G = G1, Kb = K1)
        el2 = ConstantElasticity(; G = G2, Kb = K2)
        rheology = (
            SetMaterialParams(;
                Phase = 1,
                CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), el1)),
                Elasticity = el1,
                Density = MeltDependent_Density(;
                    ρsolid = T_Density(; ρ0 = 1.0, α = α1s), ρmelt = T_Density(; ρ0 = 1.0, α = α1m)
                ),
            ),
            SetMaterialParams(;
                Phase = 2,
                CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), el2)),
                Elasticity = el2,
                Density = T_Density(; ρ0 = 1.0, α = α2),
            ),
        )
        dt, r, θ_dτ = 0.7, 0.9, 1.3
        P_h = [1.0 + 0.1i - 0.2j + 0.05k^2 for i in 1:nx, j in 1:ny, k in 1:nz]
        P0_h = [0.5 + 0.03i * j - 0.1k for i in 1:nx, j in 1:ny, k in 1:nz]
        ∇V_h = [0.2 * sin(i) + 0.1 * cos(j * k) for i in 1:nx, j in 1:ny, k in 1:nz]
        Q_h = [0.01 * (i - j + k) for i in 1:nx, j in 1:ny, k in 1:nz]
        η_h = [1.0 + 0.3 * mod(i * j + k, 4) for i in 1:nx, j in 1:ny, k in 1:nz]
        ΔT_h = [0.4 * cos(i + 2j - k) for i in 1:nx, j in 1:ny, k in 1:nz]
        melt_h = [0.1 * mod(i + j * k, 5) for i in 1:nx, j in 1:ny, k in 1:nz]
        dev(A) = copyto!(@zeros(ni...), A)

        # The update is the implicit pseudo-time step P' - P = ψ R(P') with
        # R(P) = -(P - P0)/(K dt) - ∇V + αΔT/dt + Q/dt, ψ = (r/θ_dτ)(1/η + 1/(G dt))⁻¹;
        # the stored residual is R(P) at the old pressure times the rock fraction.
        function check(P, RP, with_ΔT, with_melt)
            ϕc = Array(ϕ.center)
            P, RP = Array(P), Array(RP)
            for I in CartesianIndices(P)
                if !active_c(I.I...)
                    @test P[I] == 0 && RP[I] == 0
                    continue
                end
                w = r1(I.I...)
                K = w * K1 + (1 - w) * K2
                G = w * G1 + (1 - w) * G2
                φm = with_melt ? melt_h[I] : 0.0
                α = with_ΔT ? w * (φm * α1m + (1 - φm) * α1s) + (1 - w) * α2 : 0.0
                src = -∇V_h[I] + α * ΔT_h[I] / dt + Q_h[I] / dt
                R(p) = -(p - P0_h[I]) / (K * dt) + src
                ψ = r / θ_dτ / (inv(η_h[I]) + inv(G * dt))
                @test P[I] - P_h[I] ≈ ψ * R(P[I]) rtol = 1.0e-12
                @test RP[I] ≈ R(P_h[I]) * ϕc[I] rtol = 1.0e-12
            end
            return nothing
        end

        for (args, with_ΔT, with_melt) in (
                ((;), false, false),
                ((; ΔT = dev(ΔT_h)), true, false),
                ((; ΔT = dev(ΔT_h), melt_fraction = dev(melt_h)), true, true),
            )
            P, RP = dev(P_h), @zeros(ni...)
            JR3.compute_variational_P!(
                P, dev(P0_h), RP, dev(∇V_h), dev(Q_h), dev(η_h), rheology, pr, ϕ, dt, r, θ_dτ, args
            )
            check(P, RP, with_ΔT, with_melt)
        end

        # masking leaves live pressures alone and zeroes eliminated ones
        P, RP = dev(P_h), dev(P0_h)
        @parallel (@idx ni) JR3.mask_variational_pressure!(P, RP, ϕ)
        live = [active_c(I.I...) for I in CartesianIndices(P_h)]
        @test Array(P) == ifelse.(live, P_h, 0.0)
        @test Array(RP) == ifelse.(live, P0_h, 0.0)
    end

    ## 4. Stress update ------------------------------------------------------------------------

    # Non-uniform inputs for the stress kernels.
    function stress_state(; ε_scale = 1.0)
        stokes = StokesArrays(backend, ni)
        s = ε_scale
        fill_dev!(stokes.ε.xx, (i, j, k) -> s * (0.3 + 0.1i - 0.05j * k))
        fill_dev!(stokes.ε.yy, (i, j, k) -> s * (-0.2 + 0.07j - 0.02i * k))
        fill_dev!(stokes.ε.zz, (i, j, k) -> s * (-0.1 - 0.1i + 0.03j + 0.02k))
        fill_dev!(stokes.ε.yz, (i, j, k) -> s * (0.4 + 0.05i * j - 0.03k))
        fill_dev!(stokes.ε.xz, (i, j, k) -> s * (-0.25 + 0.04j + 0.02i * k))
        fill_dev!(stokes.ε.xy, (i, j, k) -> s * (0.15 - 0.03i + 0.06j - 0.01k^2))
        for (A, a) in zip(@tensor_center(stokes.τ_o), (0.2, -0.1, 0.05, 0.3, -0.2, 0.1))
            fill_dev!(A, (i, j, k) -> a * (1 + 0.1i - 0.07j + 0.05k))
        end
        for (A, a) in zip((stokes.τ_o.yz, stokes.τ_o.xz, stokes.τ_o.xy), (0.25, -0.15, 0.12))
            fill_dev!(A, (i, j, k) -> a * (1 - 0.08i + 0.06j + 0.04k))
        end
        fill_dev!(stokes.viscosity.η, (i, j, k) -> 1.0 + 0.5 * mod(i + 2j + 3k, 5))
        return stokes
    end

    # Launched over `ni .+ 1` by default, as the solver does, so every edge including
    # those on the far boundary planes is updated.
    function vs_stress!(stokes, Pin, λ, λv, rheology, pr, ϕ; dt, θ_dτ, relλ, range = ni .+ 1)
        @parallel (@idx range) JR3.update_stresses_center_vertex!(
            @strain(stokes),
            @plastic_strain(stokes),
            stokes.EII_pl,
            stokes.ε_vol_pl,
            stokes.EVol_pl,
            @tensor_center(stokes.τ),
            (stokes.τ.yz, stokes.τ.xz, stokes.τ.xy),
            @tensor_center(stokes.τ_o),
            (stokes.τ_o.yz, stokes.τ_o.xz, stokes.τ_o.xy),
            Pin,
            stokes.P,
            stokes.viscosity.η,
            λ,
            λv,
            stokes.τ.II,
            stokes.viscosity.η_vep,
            relλ,
            dt,
            θ_dτ,
            rheology,
            pr.center,
            pr.vertex,
            pr.xy,
            pr.yz,
            pr.xz,
            ϕ,
        )
        return nothing
    end

    function reference_stress!(stokes, Pin, λ, λv, rheology, pr; dt, θ_dτ, relλ)
        @parallel (@idx ni .+ 1) JR3.update_stresses_center_vertex_ps!(
            @strain(stokes),
            @plastic_strain(stokes),
            stokes.EII_pl,
            stokes.ε_vol_pl,
            stokes.EVol_pl,
            @tensor_center(stokes.τ),
            (stokes.τ.yz, stokes.τ.xz, stokes.τ.xy),
            @tensor_center(stokes.τ_o),
            (stokes.τ_o.yz, stokes.τ_o.xz, stokes.τ_o.xy),
            Pin,
            stokes.P,
            stokes.viscosity.η,
            λ,
            λv,
            stokes.τ.II,
            stokes.viscosity.η_vep,
            relλ,
            dt,
            θ_dτ,
            rheology,
            pr.center,
            pr.vertex,
            pr.xy,
            pr.yz,
            pr.xz,
        )
        return nothing
    end

    shear_multipliers() = (@zeros(nx, ny + 1, nz + 1), @zeros(nx + 1, ny, nz + 1), @zeros(nx + 1, ny + 1, nz))

    const G_el = 2.0
    const dt_el = 0.8

    function viscoelastic_rheology()
        el = ConstantElasticity(; G = G_el, Kb = Inf)
        return (
            SetMaterialParams(;
                Phase = 1,
                CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), el)),
                Elasticity = el,
            ),
        )
    end

    const C_dp = 0.2
    const φ_dp = 30.0
    const η_vp = 0.1

    function plastic_rheology()
        el = ConstantElasticity(; G = G_el, Kb = Inf)
        pl = DruckerPrager_regularised(; C = C_dp, ϕ = φ_dp, η_vp, Ψ = 0.0)
        return (
            SetMaterialParams(;
                Phase = 1,
                CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), el, pl)),
                Elasticity = el,
            ),
        )
    end

    pressure_field() = copyto!(@zeros(ni...), [0.5 + 0.1i + 0.2j - 0.15k for i in 1:nx, j in 1:ny, k in 1:nz])

    @testset "3D variational visco-elastic stress update" begin
        ϕ = layered_rock_ratio()
        pr = single_phase_ratios()
        rheology = viscoelastic_rheology()
        stokes = stress_state()
        Pin = pressure_field()
        λ = @zeros(ni...)
        λv = shear_multipliers()
        # pseudo-time iterations contract towards the Maxwell stress
        for _ in 1:60
            vs_stress!(stokes, Pin, λ, λv, rheology, pr, ϕ; dt = dt_el, θ_dτ = 1.0, relλ = 1.0)
        end

        η = Array(stokes.viscosity.η)
        ε = Array.(@strain(stokes))
        τo_c = Array.(@tensor_center(stokes.τ_o))
        τo_e = Array.((stokes.τ_o.yz, stokes.τ_o.xz, stokes.τ_o.xy))
        τ_c = Array.(@tensor_center(stokes.τ))
        τ_e = Array.((stokes.τ.yz, stokes.τ.xz, stokes.τ.xy))
        τII, η_vep, Pc = Array(stokes.τ.II), Array(stokes.viscosity.η_vep), Array(stokes.P)
        Gdt = G_el * dt_el
        # Maxwell body: τ = 2 η_ve (ε + τ_o / (2 G Δt)), η_ve = (1/η + 1/(G Δt))⁻¹
        maxwell(ηij, εij, τoij) = 2 / (inv(ηij) + inv(Gdt)) * (εij + τoij / (2Gdt))

        for I in CartesianIndices(η)
            i, j, k = I.I
            if !active_c(i, j, k)
                @test all(A -> A[I] == 0, τ_c)
                @test η_vep[I] == 0 && Pc[I] == 0
                continue
            end
            # shear strain rates at the centre are the mean of the four surrounding edges
            εc = (
                ε[1][I], ε[2][I], ε[3][I],
                (ε[4][i, j, k] + ε[4][i, j + 1, k] + ε[4][i, j, k + 1] + ε[4][i, j + 1, k + 1]) / 4,
                (ε[5][i, j, k] + ε[5][i + 1, j, k] + ε[5][i, j, k + 1] + ε[5][i + 1, j, k + 1]) / 4,
                (ε[6][i, j, k] + ε[6][i + 1, j, k] + ε[6][i, j + 1, k] + ε[6][i + 1, j + 1, k]) / 4,
            )
            τ_exp = ntuple(n -> maxwell(η[I], εc[n], τo_c[n][I]), 6)
            @test all(n -> isapprox(τ_c[n][I], τ_exp[n]; rtol = 1.0e-10), 1:6)
            @test τII[I] ≈ invariant(τ_exp) rtol = 1.0e-10
            @test η_vep[I] ≈ invariant(τ_exp) / (2 * invariant(εc)) rtol = 1.0e-10
            # incompressible: the corrected pressure is the input pressure
            @test Pc[I] == Array(Pin)[I]
        end

        # Edge viscosity is the harmonic mean of the four cells sharing the edge, clamped
        # onto the domain at the boundary.
        ηyz(i, j, k) = harmonic((η[i, cl(j + dj, ny), cl(k + dk, nz)] for dj in -1:0, dk in -1:0)...)
        ηxz(i, j, k) = harmonic((η[cl(i + di, nx), j, cl(k + dk, nz)] for di in -1:0, dk in -1:0)...)
        ηxy(i, j, k) = harmonic((η[cl(i + di, nx), cl(j + dj, ny), k] for di in -1:0, dj in -1:0)...)
        # `vs_stress!` is launched over `ni .+ 1`, so this covers every edge entry,
        # including those on the far boundary planes.
        for (n, ηe, active) in ((1, ηyz, active_yz), (2, ηxz, active_xz), (3, ηxy, active_xy))
            A = τ_e[n]
            @test all(
                active(I.I...) ?
                    isapprox(A[I], maxwell(ηe(I.I...), ε[3 + n][I], τo_e[n][I]); rtol = 1.0e-10) :
                    iszero(A[I])
                    for I in CartesianIndices(size(A))
            )
        end
    end

    @testset "3D variational Drucker-Prager stress update" begin
        pr = single_phase_ratios()
        rheology = plastic_rheology()
        Pin = pressure_field()
        kw = (; dt = dt_el, θ_dτ = 1.0, relλ = 1.0)

        @testset "full rock matches the unmasked kernel" begin
            vs = stress_state(; ε_scale = 3.0)
            ref = stress_state(; ε_scale = 3.0)
            λ_vs, λ_ref = @zeros(ni...), @zeros(ni...)
            λv_vs, λv_ref = shear_multipliers(), shear_multipliers()
            ϕ = full_rock_ratio()
            # neighbouring entries are read while the same launch updates them, so the two
            # kernels are compared at their converged, order-independent state
            for _ in 1:300
                vs_stress!(vs, Pin, λ_vs, λv_vs, rheology, pr, ϕ; kw...)
                reference_stress!(ref, Pin, λ_ref, λv_ref, rheology, pr; kw...)
            end
            @test Array(λ_vs) ≈ Array(λ_ref)
            @test maximum(Array(λ_vs)) > 0
            for (a, b) in zip(λv_vs, λv_ref)
                @test Array(a) ≈ Array(b)
            end
            for (a, b) in zip(
                    (@tensor(vs.τ)..., @tensor_center(vs.τ)..., @plastic_strain(vs)..., vs.τ.II, vs.viscosity.η_vep, vs.P),
                    (@tensor(ref.τ)..., @tensor_center(ref.τ)..., @plastic_strain(ref)..., ref.τ.II, ref.viscosity.η_vep, ref.P),
                )
                @test Array(a) ≈ Array(b)
            end
        end

        @testset "centre stress returns to the yield surface" begin
            stokes = stress_state(; ε_scale = 3.0)
            ϕ = full_rock_ratio()
            λ = @zeros(ni...)
            λv = shear_multipliers()
            for _ in 1:200
                vs_stress!(stokes, Pin, λ, λv, rheology, pr, ϕ; kw...)
            end
            η = Array(stokes.viscosity.η)
            ε = Array.(@strain(stokes))
            τo_c = Array.(@tensor_center(stokes.τ_o))
            τ_c = Array.(@tensor_center(stokes.τ))
            ε_pl = Array.(@plastic_strain(stokes))
            τII, λh, P = Array(stokes.τ.II), Array(λ), Array(Pin)
            Gdt = G_el * dt_el
            for I in CartesianIndices(η)
                i, j, k = I.I
                εc = (
                    ε[1][I], ε[2][I], ε[3][I],
                    (ε[4][i, j, k] + ε[4][i, j + 1, k] + ε[4][i, j, k + 1] + ε[4][i, j + 1, k + 1]) / 4,
                    (ε[5][i, j, k] + ε[5][i + 1, j, k] + ε[5][i, j, k + 1] + ε[5][i + 1, j, k + 1]) / 4,
                    (ε[6][i, j, k] + ε[6][i + 1, j, k] + ε[6][i, j + 1, k] + ε[6][i + 1, j + 1, k]) / 4,
                )
                τ_trial = ntuple(n -> 2 / (inv(η[I]) + inv(Gdt)) * (εc[n] + τo_c[n][I] / (2Gdt)), 6)
                # Drucker-Prager, F = τII - C cosφ - P sinφ, with viscoplastic regularisation:
                # the converged stress sits at τII = C cosφ + P sinφ + η_vp λ
                Y = C_dp * cosd(φ_dp) + P[I] * sind(φ_dp) + η_vp * λh[I]
                @test invariant(τ_trial) > Y
                @test τII[I] ≈ Y rtol = 1.0e-8
                # radial return: the stress keeps the direction of the visco-elastic trial
                @test all(n -> isapprox(τ_c[n][I], τ_trial[n] * Y / invariant(τ_trial); rtol = 1.0e-8), 1:6)
                # associated flow on the deviator, ε_pl = λ τ / (2 τII)
                @test all(n -> isapprox(ε_pl[n][I], λh[I] * τ_c[n][I] / (2τII[I]); rtol = 1.0e-8), 1:3)
            end
        end

        @testset "air above the surface" begin
            stokes = stress_state(; ε_scale = 3.0)
            ϕ = layered_rock_ratio()
            λ = @zeros(ni...)
            λv = shear_multipliers()
            for _ in 1:5
                vs_stress!(stokes, Pin, λ, λv, rheology, pr, ϕ; kw...)
            end
            τ_e = Array.((stokes.τ.yz, stokes.τ.xz, stokes.τ.xy))
            τ_c = Array.(@tensor_center(stokes.τ))
            for (A, active) in zip(τ_e, (active_yz, active_xz, active_xy))
                @test all(iszero(A[I]) != active(I.I...) for I in CartesianIndices(ni))
            end
            dead = [I for I in CartesianIndices(τ_c[1]) if !active_c(I.I...)]
            @test all(A -> all(iszero(A[I]) for I in dead), τ_c)
            @test all(iszero(Array(stokes.viscosity.η_vep)[I]) for I in dead)

            # A live, yielding yz edge follows the flow rule, ε_pl ∝ τ, so its plastic strain
            # rate is non-zero with the sign of τyz. The edges on level kc+1 and the edge at
            # the index of the dry cell `hole` are live even though the cell centre at the
            # same index is eliminated; their plastic strain rate is unaffected by that.
            λyz = Array(λv[1])
            εpl_yz = Array(stokes.ε_pl.yz)
            follows_flow_rule(I) = λyz[I] > 0 && sign(εpl_yz[I]) == sign(τ_e[1][I]) != 0
            shadowed = [CartesianIndex(hole); vec([CartesianIndex(i, j, kc + 1) for i in 1:nx, j in 1:ny])]
            unshadowed = [I for I in CartesianIndices((nx, ny, kc)) if I ∉ shadowed]
            @test all(I -> λyz[I] > 0, shadowed)
            @test all(follows_flow_rule, unshadowed)
            @test all(follows_flow_rule, shadowed)
        end
    end

    ## 5. Stress update in strain-increment form (2D) -----------------------------------------

    # The increment form of the variational stress update is only implemented in 2D.
    const ni2 = (5, 4)
    const kc2 = 2

    # Rock below y = (kc2 + 1/4)dy: rows at cell-centre height (centre, Vx) hold 1, 1/4, 0
    # at j ≤ kc2, j = kc2+1 and above; rows at vertex height (vertex, Vy) hold 1, 3/4, 0.
    # A centre needs its top face Vy[j+1], so it survives for j ≤ kc2; a vertex needs its
    # own fraction and the faces on both sides, so it survives for j ≤ kc2+1.
    function layered_rock_ratio2()
        ϕ = JR2.RockRatio(backend, ni2...)
        centre_row(j) = j ≤ kc2 ? 1.0 : j == kc2 + 1 ? 0.25 : 0.0
        vertex_row(j) = j ≤ kc2 ? 1.0 : j == kc2 + 1 ? 0.75 : 0.0
        for f in (:center, :Vx)
            fill_dev!(getfield(ϕ, f), (i, j) -> centre_row(j))
        end
        for f in (:vertex, :Vy)
            fill_dev!(getfield(ϕ, f), (i, j) -> vertex_row(j))
        end
        return ϕ
    end
    function full_rock_ratio2()
        ϕ = JR2.RockRatio(backend, ni2...)
        for f in (:center, :vertex, :Vx, :Vy)
            getfield(ϕ, f) .= 1.0
        end
        return ϕ
    end
    active2_c(i, j) = j ≤ kc2
    active2_v(i, j) = j ≤ kc2 + 1

    function single_phase_ratios2()
        pr = PhaseRatios(backend_JP, 1, ni2)
        for f in (:center, :vertex, :Vx, :Vy)
            getfield(pr, f).data .= 1.0
        end
        return pr
    end

    function stress_state2(dt; ε_scale = 1.0)
        stokes = StokesArrays(backend, ni2)
        s = ε_scale
        fill_dev!(stokes.ε.xx, (i, j) -> s * (0.3 + 0.1i - 0.07j))
        fill_dev!(stokes.ε.yy, (i, j) -> s * (-0.25 + 0.05i * j))
        fill_dev!(stokes.ε.xy, (i, j) -> s * (0.2 - 0.04i + 0.09j))
        for (Δ, e) in zip(@strain_increment(stokes), @strain(stokes))
            copyto!(Δ, Array(e) .* dt)
        end
        for (A, a) in zip(@tensor_center(stokes.τ_o), (0.2, -0.15, 0.1))
            fill_dev!(A, (i, j) -> a * (1 + 0.1i - 0.08j))
        end
        fill_dev!(stokes.τ_o.xy, (i, j) -> 0.12 * (1 - 0.05i + 0.1j))
        fill_dev!(stokes.viscosity.η, (i, j) -> 1.0 + 0.5 * mod(i + 2j, 5))
        return stokes
    end

    stress_args2(stokes, Pin, λ, λv) = (
        @plastic_strain(stokes), stokes.EII_pl, stokes.ε_vol_pl, stokes.EVol_pl,
        @tensor_center(stokes.τ), (stokes.τ.xy,), @tensor_center(stokes.τ_o), (stokes.τ_o.xy,),
        Pin, stokes.P, stokes.viscosity.η, λ, λv, stokes.τ.II, stokes.viscosity.η_vep,
    )

    function rate_stress2!(stokes, Pin, λ, λv, rheology, pr, ϕ; dt, θ_dτ = 1.0, relλ = 1.0)
        @parallel (@idx ni2 .+ 1) JR2.update_stresses_center_vertex!(
            @strain(stokes), stress_args2(stokes, Pin, λ, λv)...,
            relλ, dt, θ_dτ, rheology, pr.center, pr.vertex, ϕ,
        )
        return nothing
    end

    function increment_stress2!(stokes, Pin, λ, λv, rheology, pr, ϕ; dt, θ_dτ = 1.0, relλ = 1.0)
        @parallel (@idx ni2 .+ 1) JR2.update_stresses_center_vertex!(
            @strain(stokes), @strain_increment(stokes), stress_args2(stokes, Pin, λ, λv)...,
            relλ, dt, θ_dτ, rheology, pr.center, pr.vertex, ϕ,
        )
        return nothing
    end

    function reference_increment_stress2!(stokes, Pin, λ, λv, rheology, pr; dt, θ_dτ = 1.0, relλ = 1.0)
        @parallel (@idx ni2 .+ 1) JR2.update_stresses_center_vertex_ps!(
            @strain(stokes), @strain_increment(stokes), stress_args2(stokes, Pin, λ, λv)...,
            relλ, dt, θ_dτ, rheology, pr.center, pr.vertex,
        )
        return nothing
    end

    pressure_field2() = copyto!(@zeros(ni2...), [0.5 + 0.1i + 0.2j for i in 1:ni2[1], j in 1:ni2[2]])

    @testset "2D variational stress update, rate and increment forms" begin
        nx2, ny2 = ni2
        pr = single_phase_ratios2()
        Pin = pressure_field2()

        @testset "visco-elastic, air above the surface" begin
            ϕ = layered_rock_ratio2()
            rheology = viscoelastic_rheology()
            rate = stress_state2(dt_el)
            incr = stress_state2(dt_el)
            λr, λi = @zeros(ni2...), @zeros(ni2...)
            λvr, λvi = @zeros((ni2 .+ 1)...), @zeros((ni2 .+ 1)...)
            for _ in 1:60
                rate_stress2!(rate, Pin, λr, λvr, rheology, pr, ϕ; dt = dt_el)
                increment_stress2!(incr, Pin, λi, λvi, rheology, pr, ϕ; dt = dt_el)
            end

            η = Array(rate.viscosity.η)
            εxx, εyy, εxy = Array.(@strain(rate))
            τo = Array.(@tensor_center(rate.τ_o))
            τo_xy = Array(rate.τ_o.xy)
            τ = Array.(@tensor_center(rate.τ))
            τxy = Array(rate.τ.xy)
            Gdt = G_el * dt_el
            maxwell(ηij, εij, τoij) = 2 / (inv(ηij) + inv(Gdt)) * (εij + τoij / (2Gdt))
            for I in CartesianIndices(η)
                i, j = I.I
                if active2_c(i, j)
                    εc = (εxx[I], εyy[I], (εxy[i, j] + εxy[i + 1, j] + εxy[i, j + 1] + εxy[i + 1, j + 1]) / 4)
                    @test all(n -> isapprox(τ[n][I], maxwell(η[I], εc[n], τo[n][I]); rtol = 1.0e-10), 1:3)
                else
                    @test all(A -> iszero(A[I]), τ)
                end
            end
            ηv(i, j) = harmonic((η[cl(i + di, nx2), cl(j + dj, ny2)] for di in -1:0, dj in -1:0)...)
            @test all(
                active2_v(I.I...) ?
                    isapprox(τxy[I], maxwell(ηv(I.I...), εxy[I], τo_xy[I]); rtol = 1.0e-10) :
                    iszero(τxy[I])
                    for I in CartesianIndices(τxy)
            )

            # Δε = ε Δt makes the increment form the same update as the rate form
            for (a, b) in zip(
                    (@tensor_center(incr.τ)..., incr.τ.xy, incr.τ.II, incr.viscosity.η_vep, incr.P),
                    (@tensor_center(rate.τ)..., rate.τ.xy, rate.τ.II, rate.viscosity.η_vep, rate.P),
                )
                @test Array(a) ≈ Array(b) rtol = 1.0e-10
            end
        end

        # Neighbouring entries are read while the same launch updates them, so kernels are
        # compared at their converged state, which does not depend on the update order.
        @testset "Drucker-Prager, full rock" begin
            ϕ = full_rock_ratio2()
            rheology = plastic_rheology()
            function converge(update!, dt)
                stokes = stress_state2(dt; ε_scale = 3.0)
                λ, λv = @zeros(ni2...), @zeros((ni2 .+ 1)...)
                for _ in 1:300
                    update!(stokes, λ, λv, dt)
                end
                return stokes, Array(λ), Array(λv)
            end
            vs!(stokes, λ, λv, dt) = increment_stress2!(stokes, Pin, λ, λv, rheology, pr, ϕ; dt)
            ref!(stokes, λ, λv, dt) = reference_increment_stress2!(stokes, Pin, λ, λv, rheology, pr; dt)

            # with a unit time step the masked and unmasked increment forms coincide
            vs, λ_vs, λv_vs = converge(vs!, 1.0)
            ref, λ_ref, λv_ref = converge(ref!, 1.0)
            @test minimum(λ_ref) > 0 && minimum(λv_ref) > 0
            @test λ_vs ≈ λ_ref rtol = 1.0e-8
            @test λv_vs ≈ λv_ref rtol = 1.0e-8
            for (a, b) in zip(
                    (@tensor_center(vs.τ)..., vs.τ.xy, @plastic_strain(vs)..., vs.τ.II, vs.viscosity.η_vep),
                    (@tensor_center(ref.τ)..., ref.τ.xy, @plastic_strain(ref)..., ref.τ.II, ref.viscosity.η_vep),
                )
                @test Array(a) ≈ Array(b) rtol = 1.0e-8
            end

            # Converged centre stress sits at τII = C cosφ + P sinφ + η_vp λ, for any time
            # step, in both the unmasked kernel and the masked increment form.
            P = Array(Pin)
            on_yield_surface(stokes, λ) = all(
                isapprox(Array(stokes.τ.II)[I], C_dp * cosd(φ_dp) + P[I] * sind(φ_dp) + η_vp * λ[I]; rtol = 1.0e-8)
                    for I in CartesianIndices(P)
            )
            ref, λ_ref, _ = converge(ref!, dt_el)
            @test on_yield_surface(ref, λ_ref)
            vs, λ_vs, _ = converge(vs!, dt_el)
            @test on_yield_surface(vs, λ_vs)
        end

        @testset "strain rate from increments" begin
            stokes = StokesArrays(backend, ni2)
            fill_dev!(stokes.Δε.xx, (i, j) -> 0.1i - 0.3j)
            fill_dev!(stokes.Δε.yy, (i, j) -> 0.2 * i * j)
            fill_dev!(stokes.Δε.xy, (i, j) -> 0.05i + 0.15j^2)
            for A in @strain(stokes)
                A .= 7.0
            end
            _dt = inv(dt_el)
            @parallel (@idx ni2 .+ 1) JR2.compute_strain_rate_from_increment!(
                @strain(stokes)..., @strain_increment(stokes)..., layered_rock_ratio2(), _dt
            )
            for (ε, Δε, active) in zip(
                    Array.(@strain(stokes)), Array.(@strain_increment(stokes)), (active2_c, active2_c, active2_v)
                )
                @test all(ε[I] ≈ (active(I.I...) ? Δε[I] / dt_el : 0.0) for I in CartesianIndices(ε))
            end
        end
    end

    ## 6. Momentum update (2D) ----------------------------------------------------------------

    const di2 = (0.3, 0.45)

    function momentum_state2()
        stokes = StokesArrays(backend, ni2)
        dx2, dy2 = di2
        Ly = ni2[2] * dy2
        ρg0 = 2.5
        # hydrostatic pressure balances the uniform part of the body force; the shear stress
        # is linear in space and the density varies with depth
        fill_dev!(stokes.P, (i, j) -> ρg0 * (Ly - (j - 0.5) * dy2))
        fill_dev!(stokes.τ.xy, (i, j) -> 0.3 * (j - 1) * dy2 + 0.2 * (i - 1) * dx2)
        fill_dev!(stokes.V.Vx, (i, j) -> 0.1 * sin(i + j))
        fill_dev!(stokes.V.Vy, (i, j) -> 0.1 * cos(i - 2j))
        ρgx = @zeros(ni2...)
        ρgy = copyto!(@zeros(ni2...), [ρg0 * (1 + 0.04 * j^2) for i in 1:ni2[1], j in 1:ni2[2]])
        ητ = copyto!(@zeros(ni2...), [1.0 + 0.5 * mod(i + 2j, 3) for i in 1:ni2[1], j in 1:ni2[2]])
        return stokes, ρgx, ρgy, ητ
    end

    function momentum2!(stokes, ρgx, ρgy, ητ, ηdτ, ϕ; dt = nothing)
        _dc = inv.(di2)
        extra = isnothing(dt) ? () : (dt,)
        @parallel (@idx ni2 .+ 1) JR2.compute_V!(
            @velocity(stokes)..., @residuals(stokes.R)..., stokes.P, @stress(stokes)...,
            ηdτ, ρgx, ρgy, ητ, ϕ, _dc, _dc, extra...,
        )
        return nothing
    end

    @testset "2D variational momentum update" begin
        nx2, ny2 = ni2
        dx2, dy2 = di2
        ηdτ = 0.2
        dt = 0.7

        @testset "full-rock residual" begin
            stokes, ρgx, ρgy, ητ = momentum_state2()
            momentum2!(stokes, ρgx, ρgy, ητ, 0.0, full_rock_ratio2())
            Rx, Ry = Array.(@residuals(stokes.R))
            ρ = Array(ρgy)
            # Rx = ∂τxy/∂y; Ry = ∂τxy/∂x - ∂P/∂y - ρg, with the face body force the mean of
            # the two cells it separates
            @test all(≈(0.3), Rx)
            @test all(
                Ry[i, j] ≈ 0.2 + 2.5 - (ρ[i, j] + ρ[i, j + 1]) / 2 for i in 1:nx2, j in 1:(ny2 - 1)
            )
        end

        @testset "face mass and eliminated rows" begin
            ϕ = layered_rock_ratio2()
            stokes, ρgx, ρgy, ητ = momentum_state2()
            momentum2!(stokes, ρgx, ρgy, ητ, 0.0, ϕ)
            R0 = Array.(@residuals(stokes.R))
            V0 = Array.(@velocity(stokes))

            stokes, ρgx, ρgy, ητ = momentum_state2()
            momentum2!(stokes, ρgx, ρgy, ητ, ηdτ, ϕ)
            R = Array.(@residuals(stokes.R))
            V = Array.(@velocity(stokes))
            η = Array(ητ)
            fVx, fVy = Array(ϕ.Vx), Array(ϕ.Vy)
            # A face row is kept while its control volume holds liquid; its pseudo-time step
            # is divided by the liquid face mass, floored at 0.1.
            for i in 1:(nx2 - 1), j in 1:ny2
                f = fVx[i + 1, j]
                expected = f > 0 ? V0[1][i + 1, j + 1] + R0[1][i, j] * ηdτ / (max(f, 0.1) * (η[i, j] + η[i + 1, j]) / 2) : 0.0
                @test V[1][i + 1, j + 1] ≈ expected
                @test R[1][i, j] == (f > 0 ? R0[1][i, j] : 0.0)
            end
            for i in 1:nx2, j in 1:(ny2 - 1)
                f = fVy[i, j + 1]
                expected = f > 0 ? V0[2][i + 1, j + 1] + R0[2][i, j] * ηdτ / (max(f, 0.1) * (η[i, j] + η[i, j + 1]) / 2) : 0.0
                @test V[2][i + 1, j + 1] ≈ expected
                @test R[2][i, j] == (f > 0 ? R0[2][i, j] : 0.0)
            end
            # the layered surface puts partial faces on both components
            @test any(f -> 0 < f < 1, fVx) && any(f -> 0 < f < 1, fVy)
        end

        @testset "implicit density-gradient correction" begin
            ϕ = full_rock_ratio2()
            stokes, ρgx, ρgy, ητ = momentum_state2()
            momentum2!(stokes, ρgx, ρgy, ητ, 0.0, ϕ)
            R0 = Array.(@residuals(stokes.R))
            V0 = Array.(@velocity(stokes))

            stokes, ρgx, ρgy, ητ = momentum_state2()
            momentum2!(stokes, ρgx, ρgy, ητ, ηdτ, ϕ; dt)
            R = Array.(@residuals(stokes.R))
            V = Array.(@velocity(stokes))
            η, ρ = Array(ητ), Array(ρgy)
            # The Vy row is advanced implicitly in the surface-loading term Vy ∂(ρg)/∂y Δt:
            #   (Vy' - Vy) η/ηdτ = R + Vy' ∂(ρg)/∂y Δt, and the stored residual uses Vy.
            for i in 1:nx2, j in 1:(ny2 - 1)
                g = (ρ[i, j + 1] - ρ[i, j]) / dy2
                ηf = (η[i, j] + η[i, j + 1]) / 2
                Vy, Vy′ = V0[2][i + 1, j + 1], V[2][i + 1, j + 1]
                @test (Vy′ - Vy) * ηf / ηdτ ≈ R0[2][i, j] + Vy′ * g * dt
                @test R[2][i, j] ≈ R0[2][i, j] + Vy * g * dt
            end
            # the Vx rows carry no correction
            @test R[1] ≈ R0[1]

            # the single-component kernels perform the same updates
            split, ρgx, ρgy, ητ = momentum_state2()
            _dc = inv.(di2)
            @parallel (@idx ni2 .+ 1) JR2.compute_Vx!(
                split.V.Vx, split.R.Rx, split.P, split.τ.xx, split.τ.xy, ηdτ, ρgx, ητ, ϕ, _dc, _dc
            )
            @parallel (@idx ni2 .+ 1) JR2.compute_Vy!(
                split.V.Vy, split.V.Vx, split.R.Ry, split.P, split.τ.yy, split.τ.xy, ηdτ, ρgy, ητ, ϕ, _dc, _dc, dt
            )
            @test Array(split.V.Vx) ≈ V[1]
            @test Array(split.V.Vy) ≈ V[2]
            @test Array(split.R.Ry) ≈ R[2]
        end
    end

    ## 7. Momentum residual ------------------------------------------------------------------

    function vs_momentum!(stokes, ρg, ητ, ηdτ, ϕ)
        @parallel (@idx ni .+ 1) JR3.compute_V!(
            @velocity(stokes)...,
            @residuals(stokes.R)...,
            stokes.P,
            ρg...,
            @stress(stokes)...,
            ητ,
            ηdτ,
            ϕ,
            inv.(dxi),
        )
        return nothing
    end

    @testset "3D masked momentum residual" begin
        @testset "fully eliminated rows" begin
            stokes = StokesArrays(backend, ni)
            fill_velocity!(stokes, ux, uy, uz)
            Vx0, Vy0, Vz0 = Array.(@velocity(stokes))
            for R in @residuals(stokes.R)
                R .= 1.0
            end
            ρg = ntuple(_ -> @ones(ni...), Val(3))
            vs_momentum!(stokes, ρg, @ones(ni...), 0.5, JR3.RockRatio(backend, ni))
            @test all(R -> all(iszero, Array(R)), @residuals(stokes.R))
            # interior velocity unknowns are zeroed, boundary and ghost values are not unknowns
            for (V, V0, n) in zip(Array.(@velocity(stokes)), (Vx0, Vy0, Vz0), 1:3)
                interior = ntuple(d -> d == n ? (2:(ni[d])) : (2:(ni[d] + 1)), 3)
                @test all(iszero, V[interior...])
                mask = trues(size(V))
                mask[interior...] .= false
                @test V[mask] == V0[mask]
            end
        end

        @testset "hydrostatic state with a linear shear stress" begin
            # P = ρg (Lz - z) balances the body force fz = ρg; a stress that is linear in
            # space has divergence ∂τxz/∂z along x, and the full-rock rows must see only that.
            ρg0 = 2.5
            a = 0.3
            stokes = StokesArrays(backend, ni)
            fill_dev!(stokes.P, (i, j, k) -> ρg0 * (li[3] - zc(k)))
            fill_dev!(stokes.τ.xz, (i, j, k) -> a * zv(k) + 0.2 * xv(i))
            fill_dev!(stokes.τ.xy, (i, j, k) -> 0.4 * zc(k))
            ρg = (@zeros(ni...), @zeros(ni...), fill!(@zeros(ni...), ρg0))
            resid = function ()
                vs_momentum!(stokes, ρg, @ones(ni...), 0.0, full_rock_ratio())
                Rx, Ry, Rz = Array.(@residuals(stokes.R))
                return all(≈(a), Rx) && all(x -> abs(x) < 1.0e-12, Ry) &&
                    all(z -> isapprox(z, 0.2; atol = 1.0e-12), Rz)
            end
            @test resid()
        end

        @testset "masked edge differences" begin
            # Linear edge stresses: the derivative across a face row is the coefficient of
            # that direction alone, whatever the dependence on the other two.
            τxy = [0.7 * yv(j) + 0.9 * zc(k) for i in 1:(nx + 1), j in 1:(ny + 1), k in 1:nz]
            w = ones(size(τxy))
            @test JR3._d_yi(τxy, w, inv(dy), 2, 2, 2) ≈ 0.7
            τxz = [0.7 * xv(i) + 0.9 * zv(k) for i in 1:(nx + 1), j in 1:ny, k in 1:(nz + 1)]
            @test JR3._d_xi(τxz, ones(size(τxz)), inv(dx), 2, 2, 2) ≈ 0.7
        end
    end

    ## 8. Solver --------------------------------------------------------------------------------

    @testset "3D variational Stokes solver" begin
        n = (6, 6, 6)
        init_mpi = !JustRelax.MPI.Initialized()
        igg = IGG(init_global_grid(n...; init_MPI = init_mpi)...)
        grid = Geometry(n, (1.0, 1.2, 1.4))

        rheology = (
            SetMaterialParams(;
                Phase = 1,
                Density = ConstantDensity(; ρ = 1.0),
                Gravity = ConstantGravity(; g = 1.0),
                CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
            ),
            SetMaterialParams(;
                Phase = 2,
                Density = ConstantDensity(; ρ = 2.0),
                Gravity = ConstantGravity(; g = 1.0),
                CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
            ),
        )
        # a dense block off the centre line drives a non-trivial flow
        dense(i, j, k) = (2 ≤ i ≤ 3 && 3 ≤ j ≤ 4 && 3 ≤ k ≤ 5) ? 0.0 : 1.0
        function phase_ratios_for(n)
            pr = PhaseRatios(backend_JP, 2, n)
            for f in (:center, :vertex, :Vx, :Vy, :Vz, :yz, :xz, :xy)
                A = getfield(pr, f)
                r = @zeros(size(A)...)
                fill_dev!(r, (i, j, k) -> dense(min(i, n[1]), min(j, n[2]), min(k, n[3])))
                @parallel (@idx size(A)) set_phase_ratio!(A, r)
            end
            return pr
        end
        flow_bcs = VelocityBoundaryConditions(;
            free_slip = (left = true, right = true, top = true, bot = true, front = true, back = true),
            no_slip = (left = false, right = false, top = false, bot = false, front = false, back = false),
        )

        function run(solver, ϕ; grid_arg = grid, kwargs...)
            stokes = StokesArrays(backend, n)
            pr = phase_ratios_for(n)
            ρg = ntuple(_ -> @zeros(n...), Val(3))
            pt = PTStokesCoeffs(grid.li, grid.di.center; ϵ_rel = 1.0e-9, ϵ_abs = 1.0e-12, CFL = 0.95 / √3.1)
            args = (; T = @zeros(n .+ 2...), P = stokes.P, dt = Inf)
            out = if isnothing(ϕ)
                solver(stokes, pt, grid_arg, flow_bcs, ρg, pr, rheology, args, Inf, igg; kwargs...)
            else
                solver(stokes, pt, grid_arg, flow_bcs, ρg, pr, ϕ, rheology, args, Inf, igg; kwargs...)
            end
            return stokes, out
        end

        opts = (; iterMax = 20_000, nout = 200, verbose = false)
        ref, _ = run(solve!, nothing; kwargs = opts)
        # the reference flow is not trivial
        @test maximum(abs, Array(ref.V.Vz)) > 1.0e-3

        full_rock() = begin
            ϕ = JR3.RockRatio(backend, n)
            for f in (:center, :vertex, :Vx, :Vy, :Vz, :yz, :xz, :xy)
                getfield(ϕ, f) .= 1.0
            end
            ϕ
        end
        matches_reference(stokes) = all(
            isapprox(Array(a), Array(b); rtol = 1.0e-5, atol = 1.0e-9)
                for (a, b) in zip(@velocity(stokes), @velocity(ref))
        )

        # With every fraction equal to one the variational problem is the standard one.
        @test matches_reference(first(run(solve_VariationalStokes!, full_rock(); opts...)))
        @test matches_reference(first(run(solve_VariationalStokes!, full_rock(); kwargs = opts)))
        # the spacing-only call form rebuilds the same uniform grid
        @test matches_reference(
            first(run(solve_VariationalStokes!, full_rock(); grid_arg = grid.di, kwargs = opts))
        )

        # Air in the top two cell layers: dry pressure cells and faces are eliminated, the
        # rest converges.
        function cut_mask_solution_is_consistent()
            ϕ = full_rock()
            for f in (:center, :Vx, :Vy, :xy)
                A = Array(getfield(ϕ, f))
                A[:, :, (n[3] - 1):end] .= 0
                copyto!(getfield(ϕ, f), A)
            end
            for f in (:vertex, :Vz, :yz, :xz)
                A = Array(getfield(ϕ, f))
                A[:, :, n[3]:end] .= 0
                copyto!(getfield(ϕ, f), A)
            end
            stokes, out = run(solve_VariationalStokes!, ϕ; kwargs = opts)
            P, Vz = Array(stokes.P), Array(stokes.V.Vz)
            return all(isfinite, Vz) && all(iszero, P[:, :, (n[3] - 1):end]) &&
                all(iszero, Vz[2:(end - 1), 2:(end - 1), n[3]:end]) &&
                last(out.err_evo1) < first(out.err_evo1)
        end
        @test cut_mask_solution_is_consistent()

        finalize_global_grid(; finalize_MPI = init_mpi)
    end
end
