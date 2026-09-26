push!(LOAD_PATH, "..")
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test
using LinearAlgebra, Statistics
using StaticArrays
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

# kernels launched with `@parallel` must come from the module compiled for the active backend
const JR3K = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    Base.get_extension(JustRelax, :JustRelaxAMDGPUExt).JustRelax3D
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    Base.get_extension(JustRelax, :JustRelaxCUDAExt).JustRelax3D
else
    JustRelax.JustRelax3D
end

const JR3 = JustRelax.JustRelax3D

dev(A) = PTArray(backend)(A)

# ---------------------------------------------------------------------------------------
# Reference constitutive relations
# ---------------------------------------------------------------------------------------

# second invariant of a tensor in Voigt order (xx, yy, zz, yz, xz, xy)
τII3(t) = sqrt(0.5 * (t[1]^2 + t[2]^2 + t[3]^2) + t[4]^2 + t[5]^2 + t[6]^2)
ηve(η, G, dt) = inv(inv(η) + inv(G * dt))
# one implicit pseudo-time step θ (τⁿ⁺¹ - τⁿ) = 2ηε - η (τⁿ⁺¹ - τo) / (G dt) - τⁿ⁺¹
pt_step(τ, τo, ε, η, G, dt, θ) = (θ * τ + 2η * ε + η * τo / (G * dt)) / (θ + 1 + η / (G * dt))

# cells adjacent to an edge that exist on a grid of `ni` cells; an edge spans one cell
# along its own axis and sits on vertices in the two others
const EDGE_OFFSETS = (
    yz = [(0, dj, dk) for dj in (-1, 0), dk in (-1, 0)],
    xz = [(di, 0, dk) for di in (-1, 0), dk in (-1, 0)],
    xy = [(di, dj, 0) for di in (-1, 0), dj in (-1, 0)],
)
edge_cells(ni, e, I) = [I .+ o for o in EDGE_OFFSETS[e] if all(1 .≤ I .+ o .≤ ni)]
emean(A, e, I) = mean(A[c...] for c in edge_cells(size(A), e, I))
eharm(A, e, I) = (cs = edge_cells(size(A), e, I); length(cs) / sum(inv(A[c...]) for c in cs))

sample(f, X, Y, Z) = [f(x, y, z) for x in X, y in Y, z in Z]

@parallel_indices (I...) function fill_phase1!(phases)
    @index phases[1, I...] = 1.0
    return nothing
end
function single_phase_ratios(ni)
    phase_ratios = PhaseRatios(backend_JP, 1, ni)
    for (A, n) in (
            (phase_ratios.center, ni), (phase_ratios.vertex, ni .+ 1),
            (phase_ratios.yz, size(phase_ratios.yz)), (phase_ratios.xz, size(phase_ratios.xz)),
            (phase_ratios.xy, size(phase_ratios.xy)),
        )
        @parallel (@idx n) fill_phase1!(A)
    end
    return phase_ratios
end

# uniform, anisotropic grid for the kernel tests: the 3D stencils take scalar spacings
const nx, ny, nz = 5, 4, 6
const ni = (nx, ny, nz)
const grid = Geometry(ni, (1.0, 0.6, 1.5); origin = (0.0, -0.6, 0.2))
const xc, yc, zc = Array.(grid.xci)
const xv, yv, zv = Array.(grid.xvi)
# locations of the staggered components, Voigt order
const LOC = (
    c = (xc, yc, zc), yz = (xc, yv, zv), xz = (xv, yc, zv), xy = (xv, yv, zc),
)

# affine fields f(x) = a0 + a ⋅ x: every arithmetic average of neighbours on a uniform
# grid reproduces f at the stencil midpoint
affine(a0, a) = (x, y, z) -> a0 + a[1] * x + a[2] * y + a[3] * z

@testset "Stokes kernels 3D" begin

    @testset "tensor caching at cell centers" begin
        A(k) = [k + 0.1 * i + 0.01 * j - 0.001 * l for i in 1:nx, j in 1:ny, l in 1:nz]
        E(k, n) = [k + 0.1 * i + 0.01 * j^2 - 0.001 * l for i in 1:n[1], j in 1:n[2], l in 1:n[3]]
        τ, τo = ntuple(A, 6), ntuple(k -> A(6 + k), 6)
        ε = (A(13), A(14), A(15), E(16, (nx, ny + 1, nz + 1)), E(17, (nx + 1, ny, nz + 1)), E(18, (nx + 1, ny + 1, nz)))
        I = (2, 3, 4)
        τij, τij_o, εij = JR3.cache_tensors(τ, τo, ε, I...)
        @test τij == getindex.(τ, I...)
        @test τij_o == getindex.(τo, I...)
        # shear strain rates live on edges and are averaged over the four edges of the cell
        @test εij[1:3] == getindex.(ε[1:3], I...)
        @test εij[4] ≈ mean(ε[4][I[1], I[2]:(I[2] + 1), I[3]:(I[3] + 1)])
        @test εij[5] ≈ mean(ε[5][I[1]:(I[1] + 1), I[2], I[3]:(I[3] + 1)])
        @test εij[6] ≈ mean(ε[6][I[1]:(I[1] + 1), I[2]:(I[2] + 1), I[3]])

        τw = ntuple(_ -> zeros(ni), 6)
        vals = (1.0, -2.0, 0.5, 0.25, -0.75, 3.0)
        JR3.correct_stress!(τw..., vals, I...)
        @test getindex.(τw, I...) == vals
        @test sum(sum, τw) ≈ sum(vals)
    end

    @testset "divergence and strain rate" begin
        stokes = StokesArrays(backend, ni)
        # V = A x + v0: ∇V = tr A, ε = sym(A) - tr(A)/3 I; a rigid rotation gives ε = 0
        A1 = [0.7 -0.3 0.5; 1.1 -0.4 0.2; -0.6 0.9 0.25]
        ω = [0.0 0.8 -0.3; -0.8 0.0 0.5; 0.3 -0.5 0.0]
        for A in (A1, ω)
            v0 = (0.3, -1.0, 0.6)
            Vf = ntuple(d -> affine(v0[d], A[d, :]), 3)
            for (Vd, f, X) in zip(@velocity(stokes), Vf, grid.xi_vel)
                copyto!(Vd, sample(f, Array.(X)...))
            end
            @parallel (@idx ni) JR3K.compute_∇V!(stokes.∇V, @velocity(stokes), grid._di.center)
            @parallel (@idx ni .+ 1) JR3K.compute_strain_rate!(
                stokes.∇V, @strain(stokes)..., @velocity(stokes)..., grid._di.center
            )
            trA = tr(A)
            S = (A + A') / 2
            @test all(isapprox.(Array(stokes.∇V), trA; atol = 1.0e-12))
            for (εd, ref) in zip(
                    @strain(stokes),
                    (S[1, 1] - trA / 3, S[2, 2] - trA / 3, S[3, 3] - trA / 3, S[2, 3], S[1, 3], S[1, 2]),
                )
                @test all(isapprox.(Array(εd), ref; atol = 1.0e-12))
            end
        end
    end

    # smooth non-uniform material fields
    η_c = sample((x, y, z) -> 1.0 + 0.4 * sin(3x + 2y - z), LOC.c...)
    G_c = sample((x, y, z) -> 0.9 + 0.3 * cos(x - 2y + z), LOC.c...)
    dt = 0.6
    # affine strain rates and old stresses, sampled at each component's own location
    εf = (
        affine(0.3, (0.2, -0.1, 0.05)), affine(-0.1, (0.1, 0.3, -0.05)), affine(-0.15, (-0.2, 0.1, 0.1)),
        affine(0.2, (0.1, -0.2, 0.15)), affine(-0.25, (0.05, 0.1, -0.1)), affine(0.15, (-0.1, 0.2, 0.1)),
    )
    τof = (
        affine(0.05, (0.1, 0.0, -0.05)), affine(-0.1, (0.0, 0.1, 0.05)), affine(0.02, (-0.05, 0.05, 0.0)),
        affine(0.03, (0.02, -0.04, 0.01)), affine(-0.02, (0.01, 0.03, -0.02)), affine(0.04, (-0.03, 0.0, 0.02)),
    )
    shear_loc = (LOC.yz, LOC.xz, LOC.xy)
    ε_stag = (ntuple(d -> sample(εf[d], LOC.c...), 3)..., ntuple(d -> sample(εf[3 + d], shear_loc[d]...), 3)...)
    τo_c = ntuple(d -> sample(τof[d], LOC.c...), 6)
    τo_edge = ntuple(d -> sample(τof[3 + d], shear_loc[d]...), 3)

    @testset "visco-elastic compute_τ! and viscous compute_τ_vertex!" begin
        θ = 0.7
        τ0 = (ntuple(d -> sample(affine(0.1 * d, (0.0, 0.1, -0.1)), LOC.c...), 3)..., ntuple(d -> sample(affine(-0.05 * d, (0.1, 0.0, 0.1)), shear_loc[d]...), 3)...)
        τ = dev.(τ0)
        τo = (dev.(τo_c[1:3])..., dev.(τo_edge)...)
        @parallel (@idx ni .+ 1) JR3K.compute_τ!(τ..., τo..., dev.(ε_stag)..., dev(η_c), dev(G_c), dt, θ)
        for d in 1:3
            @test Array(τ[d]) ≈ pt_step.(τ0[d], τo_c[d], ε_stag[d], η_c, G_c, dt, θ)
        end
        # shear components on edges: arithmetic means of η and G over the adjacent cells
        for (d, e) in zip(4:6, (:yz, :xz, :xy))
            τh = Array(τ[d])
            ref = [
                pt_step(τ0[d][I], τo_edge[d - 3][I], ε_stag[d][I], emean(η_c, e, Tuple(I)), emean(G_c, e, Tuple(I)), dt, θ)
                    for I in CartesianIndices(τh)
            ]
            @test τh ≈ ref
        end

        # viscous edge update with the harmonic mean of the adjacent cells' viscosity
        τs = dev.(τ0[4:6])
        @parallel (@idx ni .+ 1) JR3K.compute_τ_vertex!(τs..., dev.(ε_stag[4:6])..., dev(η_c), θ)
        for (d, e) in zip(4:6, (:yz, :xz, :xy))
            τh = Array(τs[d - 3])
            ref = [(θ * τ0[d][I] + 2eharm(η_c, e, Tuple(I)) * ε_stag[d][I]) / (θ + 1) for I in CartesianIndices(τh)]
            @test τh ≈ ref
        end
    end

    # Visco-elasto-plastic Drucker-Prager rheology. With θ_dτ = 0 the trial stress is the
    # Maxwell stress τM = 2ηve ε_ve, ε_ve = ε + τo / (2G dt), and the converged state satisfies
    #   τII = C cosϕ + Pc sinϕ + η_vp λ,   Pc = P + K dt λ sinψ
    #   τ ∥ τM,   ε_pl = ε_ve - τ / (2ηve),   εII_pl = λ / 2
    C, ϕ, ψ, ηvp, G0, K0 = 0.4, 30.0, 8.0, 0.05, 1.0, 4.0
    pl = DruckerPrager_regularised(; C, ϕ, Ψ = ψ, η_vp = ηvp)
    visc = LinearViscous(; η = 1.0)
    el = ConstantElasticity(; G = G0, Kb = K0)
    rheology = (SetMaterialParams(; Phase = 1, CompositeRheology = CompositeRheology((visc, el, pl)), Elasticity = el),)
    Pf = affine(0.1, (0.2, -0.1, 0.05))
    P_c = sample(Pf, LOC.c...)

    function check_dp_state(τ, λ, Pc, εpl, ε, τo, η, P)
        ε_ve = ε .+ τo ./ (2G0 * dt)
        ηe = ηve(η, G0, dt)
        τM = 2ηe .* ε_ve
        yields = τII3(τM) > C * cosd(ϕ) + P * sind(ϕ)
        if yields
            τII = τII3(τ)
            @test λ > 0
            @test Pc ≈ P + K0 * dt * λ * sind(ψ)
            @test τII ≈ C * cosd(ϕ) + Pc * sind(ϕ) + ηvp * λ
            @test all(τ ./ τII .≈ τM ./ τII3(τM))
            @test all(isapprox.(εpl, ε_ve .- τ ./ (2ηe); atol = 1.0e-12))
            @test τII3(εpl) ≈ λ / 2
        else
            @test λ == 0
            @test Pc ≈ P
            @test all(τ .≈ τM)
            @test all(abs.(εpl) .< 1.0e-14)
        end
        return yields
    end

    function nonlinear_state()
        stokes = StokesArrays(backend, ni)
        for (A, B) in zip(@strain(stokes), ε_stag)
            copyto!(A, B)
        end
        for (A, B) in zip(@tensor_center(stokes.τ_o), τo_c)
            copyto!(A, B)
        end
        for (A, B) in zip((stokes.τ_o.yz, stokes.τ_o.xz, stokes.τ_o.xy), τo_edge)
            copyto!(A, B)
        end
        copyto!(stokes.P, P_c)
        copyto!(stokes.viscosity.η, η_c)
        return stokes
    end
    εc(I) = ntuple(d -> εf[d](xc[I[1]], yc[I[2]], zc[I[3]]), 6)
    τoc(I) = ntuple(d -> τof[d](xc[I[1]], yc[I[2]], zc[I[3]]), 6)

    @testset "compute_τ_nonlinear! (single and multi phase)" begin
        phase_ratios = single_phase_ratios(ni)
        results = map((false, true)) do multiphase
            stokes = nonlinear_state()
            θ, λ = @zeros(ni...), @zeros(ni...)
            args = (; T = @zeros(ni...))
            for _ in 1:80
                if multiphase
                    @parallel (@idx ni) JR3K.compute_τ_nonlinear!(
                        @tensor_center(stokes.τ), stokes.τ.II, @tensor_center(stokes.τ_o),
                        @strain(stokes), @plastic_strain(stokes), stokes.EII_pl,
                        stokes.ε_vol_pl, stokes.EVol_pl, stokes.P, θ, stokes.viscosity.η,
                        stokes.viscosity.η_vep, λ, phase_ratios.center, rheology, dt, 0.0, args,
                    )
                else
                    @parallel (@idx ni) JR3K.compute_τ_nonlinear!(
                        @tensor_center(stokes.τ), stokes.τ.II, @tensor_center(stokes.τ_o),
                        @strain(stokes), @plastic_strain(stokes), stokes.EII_pl,
                        stokes.P, θ, stokes.viscosity.η, stokes.viscosity.η_vep, λ, rheology, dt, 0.0, args,
                    )
                end
            end
            τh = Array.(@tensor_center(stokes.τ))
            εplh = Array.(@plastic_strain(stokes))
            λh, θh, ηveph = Array(λ), Array(θ), Array(stokes.viscosity.η_vep)
            nyield = 0
            for I in CartesianIndices(ni)
                # the plastic strain rate of cell I is stored at index I of every component
                nyield += check_dp_state(
                    getindex.(τh, Ref(I)), λh[I], θh[I], getindex.(εplh, Ref(I)),
                    εc(Tuple(I)), τoc(Tuple(I)), η_c[I], P_c[I],
                )
                @test ηveph[I] ≈ τII3(getindex.(τh, Ref(I))) / (2 * τII3(εc(Tuple(I))))
            end
            @test 0 < nyield < prod(ni)
            τh
        end
        @test all(results[1] .≈ results[2])
    end

    @testset "update_stresses_center_vertex_ps!" begin
        phase_ratios = single_phase_ratios(ni)
        stokes = nonlinear_state()
        λ, Pc = @zeros(ni...), @zeros(ni...)
        λv = (@zeros(size(stokes.τ.yz)...), @zeros(size(stokes.τ.xz)...), @zeros(size(stokes.τ.xy)...))
        for _ in 1:80
            @parallel (@idx ni .+ 1) JR3K.update_stresses_center_vertex_ps!(
                @strain(stokes), @plastic_strain(stokes), stokes.EII_pl, stokes.ε_vol_pl, stokes.EVol_pl,
                @tensor_center(stokes.τ), (stokes.τ.yz, stokes.τ.xz, stokes.τ.xy),
                @tensor_center(stokes.τ_o), (stokes.τ_o.yz, stokes.τ_o.xz, stokes.τ_o.xy),
                stokes.P, Pc, stokes.viscosity.η, λ, λv, stokes.τ.II, stokes.viscosity.η_vep,
                0.5, dt, 0.0, rheology, phase_ratios.center, phase_ratios.vertex,
                phase_ratios.xy, phase_ratios.yz, phase_ratios.xz,
            )
        end

        # centers
        τh = Array.(@tensor_center(stokes.τ))
        εplh = Array.(@plastic_strain(stokes))
        λh, Pch, εvolh = Array(λ), Array(Pc), Array(stokes.ε_vol_pl)
        nyield = 0
        for I in CartesianIndices(ni)
            τI = getindex.(τh, Ref(I))
            ε_ve = εc(Tuple(I)) .+ τoc(Tuple(I)) ./ (2G0 * dt)
            # normal plastic strain rates are stored at centers; shear ones live on edges
            εpl = (getindex.(εplh[1:3], Ref(I))..., (ε_ve[4:6] .- τI[4:6] ./ (2ηve(η_c[I], G0, dt)))...)
            nyield += check_dp_state(τI, λh[I], Pch[I], εpl, εc(Tuple(I)), τoc(Tuple(I)), η_c[I], P_c[I])
            @test εvolh[I] ≈ λh[I] * sind(ψ) atol = 1.0e-14
        end
        @test 0 < nyield < prod(ni)

        # interior edges: every neighbour the edge interpolates from exists, so for affine
        # fields the edge sees the exact tensor at its own location
        edges = (
            (4, :yz, stokes.τ.yz, λv[1], stokes.ε_pl.yz, (1:(nx - 1), 2:ny, 2:nz)),
            (5, :xz, stokes.τ.xz, λv[2], stokes.ε_pl.xz, (2:nx, 1:(ny - 1), 2:nz)),
            (6, :xy, stokes.τ.xy, λv[3], stokes.ε_pl.xy, (2:nx, 2:ny, 1:(nz - 1))),
        )
        for (d, e, τe, λe, εple, ranges) in edges
            τeh, λeh, εpleh = Array(τe), Array(λe), Array(εple)
            X = getfield(LOC, e)
            nyield_e = 0
            for I in CartesianIndices(ranges)
                x = getindex.(X, Tuple(I))
                ηe = ηve(eharm(η_c, e, Tuple(I)), G0, dt)
                ε_ve = ntuple(c -> εf[c](x...) + τof[c](x...) / (2G0 * dt), 6)
                τM = 2ηe .* ε_ve
                P = Pf(x...)
                if τII3(τM) > C * cosd(ϕ) + P * sind(ϕ)
                    nyield_e += 1
                    τII = τII3(τM) * τeh[I] / τM[d]
                    @test τII ≈ C * cosd(ϕ) + (P + K0 * dt * λeh[I] * sind(ψ)) * sind(ϕ) + ηvp * λeh[I]
                    @test εpleh[I] ≈ ε_ve[d] - τeh[I] / (2ηe)
                    @test εpleh[I] ≈ λeh[I] * τM[d] / (2τII3(τM))
                else
                    @test λeh[I] == 0
                    @test τeh[I] ≈ τM[d]
                end
            end
            @test nyield_e > 0
        end
    end

    @testset "edge interpolation on the domain boundary (yz edge, x boundary)" begin
        # εxz = s x is the only spatially varying field and every cell yields. A yz edge on
        # the x = lx face lies midway between the xz edges at the last two x-vertices, so it
        # must see εxz = s xc[nx].
        s, εyz = 0.4, 0.6
        phase_ratios = single_phase_ratios(ni)
        stokes = StokesArrays(backend, ni)
        stokes.ε.yz .= εyz
        copyto!(stokes.ε.xz, sample((x, y, z) -> s * x, LOC.xz...))
        stokes.viscosity.η .= 1.0
        stokes.P .= 0.0
        λ, Pc = @zeros(ni...), @zeros(ni...)
        λv = (@zeros(size(stokes.τ.yz)...), @zeros(size(stokes.τ.xz)...), @zeros(size(stokes.τ.xy)...))
        for _ in 1:80
            @parallel (@idx ni .+ 1) JR3K.update_stresses_center_vertex_ps!(
                @strain(stokes), @plastic_strain(stokes), stokes.EII_pl, stokes.ε_vol_pl, stokes.EVol_pl,
                @tensor_center(stokes.τ), (stokes.τ.yz, stokes.τ.xz, stokes.τ.xy),
                @tensor_center(stokes.τ_o), (stokes.τ_o.yz, stokes.τ_o.xz, stokes.τ_o.xy),
                stokes.P, Pc, stokes.viscosity.η, λ, λv, stokes.τ.II, stokes.viscosity.η_vep,
                0.5, dt, 0.0, rheology, phase_ratios.center, phase_ratios.vertex,
                phase_ratios.xy, phase_ratios.yz, phase_ratios.xz,
            )
        end
        ηe = ηve(1.0, G0, dt)
        τy = C * cosd(ϕ)
        # radial return of τM = 2ηe (0, 0, 0, εyz, εxz, 0): λ from the consistency condition
        function τyz_expected(εxz)
            τM = 2ηe .* (0.0, 0.0, 0.0, εyz, εxz, 0.0)
            λ = (τII3(τM) - τy) / (ηe + ηvp + K0 * dt * sind(ϕ) * sind(ψ))
            @test λ > 0
            return τM[4] * (1 - ηe * λ / τII3(τM))
        end
        τyz = Array(stokes.τ.yz)
        @test τyz[nx - 1, 2, 2] ≈ τyz_expected(s * xc[nx - 1])
        @test τyz[nx, 2, 2] ≈ τyz_expected(s * xc[nx])
    end

    @testset "edge interpolation on the domain boundary (xz edge, y boundary)" begin
        # εyz = s y is the only spatially varying field and every cell yields. An xz edge on
        # the y = ly face lies midway between the yz edges at the last two y-vertices, so it
        # must see εyz = s yc[ny].
        s, εxz = 0.4, 0.6
        phase_ratios = single_phase_ratios(ni)
        stokes = StokesArrays(backend, ni)
        stokes.ε.xz .= εxz
        copyto!(stokes.ε.yz, sample((x, y, z) -> s * y, LOC.yz...))
        stokes.viscosity.η .= 1.0
        stokes.P .= 0.0
        λ, Pc = @zeros(ni...), @zeros(ni...)
        λv = (@zeros(size(stokes.τ.yz)...), @zeros(size(stokes.τ.xz)...), @zeros(size(stokes.τ.xy)...))
        for _ in 1:80
            @parallel (@idx ni .+ 1) JR3K.update_stresses_center_vertex_ps!(
                @strain(stokes), @plastic_strain(stokes), stokes.EII_pl, stokes.ε_vol_pl, stokes.EVol_pl,
                @tensor_center(stokes.τ), (stokes.τ.yz, stokes.τ.xz, stokes.τ.xy),
                @tensor_center(stokes.τ_o), (stokes.τ_o.yz, stokes.τ_o.xz, stokes.τ_o.xy),
                stokes.P, Pc, stokes.viscosity.η, λ, λv, stokes.τ.II, stokes.viscosity.η_vep,
                0.5, dt, 0.0, rheology, phase_ratios.center, phase_ratios.vertex,
                phase_ratios.xy, phase_ratios.yz, phase_ratios.xz,
            )
        end
        ηe = ηve(1.0, G0, dt)
        τy = C * cosd(ϕ)
        # radial return of τM = 2ηe (0, 0, 0, εyz, εxz, 0): λ from the consistency condition
        function τxz_expected(εyz)
            τM = 2ηe .* (0.0, 0.0, 0.0, εyz, εxz, 0.0)
            λ = (τII3(τM) - τy) / (ηe + ηvp + K0 * dt * sind(ϕ) * sind(ψ))
            @test λ > 0
            return τM[5] * (1 - ηe * λ / τII3(τM))
        end
        τxz = Array(stokes.τ.xz)
        @test τxz[2, ny - 1, 2] ≈ τxz_expected(s * yc[ny - 1])
        @test τxz[2, ny, 2] ≈ τxz_expected(s * yc[ny])
    end

    @testset "edge interpolation on the domain boundary (xy edge, z boundary)" begin
        # εyz = s z is the only spatially varying field and every cell yields. An xy edge on
        # the z = lz face lies midway between the yz edges at the last two z-vertices, so it
        # must see εyz = s zc[nz].
        s, εxy = 0.4, 0.6
        phase_ratios = single_phase_ratios(ni)
        stokes = StokesArrays(backend, ni)
        stokes.ε.xy .= εxy
        copyto!(stokes.ε.yz, sample((x, y, z) -> s * z, LOC.yz...))
        stokes.viscosity.η .= 1.0
        stokes.P .= 0.0
        λ, Pc = @zeros(ni...), @zeros(ni...)
        λv = (@zeros(size(stokes.τ.yz)...), @zeros(size(stokes.τ.xz)...), @zeros(size(stokes.τ.xy)...))
        for _ in 1:80
            @parallel (@idx ni .+ 1) JR3K.update_stresses_center_vertex_ps!(
                @strain(stokes), @plastic_strain(stokes), stokes.EII_pl, stokes.ε_vol_pl, stokes.EVol_pl,
                @tensor_center(stokes.τ), (stokes.τ.yz, stokes.τ.xz, stokes.τ.xy),
                @tensor_center(stokes.τ_o), (stokes.τ_o.yz, stokes.τ_o.xz, stokes.τ_o.xy),
                stokes.P, Pc, stokes.viscosity.η, λ, λv, stokes.τ.II, stokes.viscosity.η_vep,
                0.5, dt, 0.0, rheology, phase_ratios.center, phase_ratios.vertex,
                phase_ratios.xy, phase_ratios.yz, phase_ratios.xz,
            )
        end
        ηe = ηve(1.0, G0, dt)
        τy = C * cosd(ϕ)
        # radial return of τM = 2ηe (0, 0, 0, εyz, 0, εxy): λ from the consistency condition
        function τxy_expected(εyz)
            τM = 2ηe .* (0.0, 0.0, 0.0, εyz, 0.0, εxy)
            λ = (τII3(τM) - τy) / (ηe + ηvp + K0 * dt * sind(ϕ) * sind(ψ))
            @test λ > 0
            return τM[6] * (1 - ηe * λ / τII3(τM))
        end
        τxy = Array(stokes.τ.xy)
        @test τxy[2, 2, nz - 1] ≈ τxy_expected(s * zc[nz - 1])
        @test τxy[2, 2, nz] ≈ τxy_expected(s * zc[nz])
    end

    @testset "edge-to-edge averages reproduce affine fields" begin
        # Exact at every target edge, including the last vertex plane of each axis, except
        # where the stencil steps from cell centers onto a boundary vertex: there one cell
        # is missing and the clamped average is one-sided by design.
        f = affine(0.3, (1.3, -0.7, 2.1))
        src = (; yz = sample(f, LOC.yz...), xz = sample(f, LOC.xz...), xy = sample(f, LOC.xy...))
        # (helper, source edge, target edge, axis stepping center -> vertex)
        cases = (
            (JR3.av_clamped_yz_z, :xy, :yz, 3), (JR3.av_clamped_yz_y, :xz, :yz, 2),
            (JR3.av_clamped_xz_z, :xy, :xz, 3), (JR3.av_clamped_xz_x, :yz, :xz, 1),
            (JR3.av_clamped_xy_y, :xz, :xy, 2), (JR3.av_clamped_xy_x, :yz, :xy, 1),
        )
        for (fn, s, t, d) in cases
            n = length.(LOC[t])
            for I in CartesianIndices(n)
                I[d] in (1, n[d]) && continue
                Ic = JR3.clamped_indices(ni, Tuple(I)...)
                @test fn(src[s], Tuple(I), Ic...) ≈ f(getindex.(LOC[t], Tuple(I))...)
            end
        end
    end

    @testset "momentum residual and velocity update" begin
        # affine P and τ: exact staggered derivatives. Body forces vary only transversally
        # to their own direction, so their face averages are exact too
        a = ntuple(d -> ((0.3, -0.2, 0.5), (0.1, 0.4, -0.3), (-0.2, 0.1, 0.6), (0.25, -0.15, 0.35), (0.05, 0.2, -0.4), (-0.3, 0.45, 0.1))[d], 6)
        pa = (0.2, -0.5, 0.35)
        τ = (ntuple(d -> sample(affine(0.0, a[d]), LOC.c...), 3)..., ntuple(d -> sample(affine(0.0, a[3 + d]), shear_loc[d]...), 3)...)
        P = sample(affine(0.0, pa), LOC.c...)
        f = (sample((x, y, z) -> 1 + y^2 - z, LOC.c...), sample((x, y, z) -> 0.5 * x + z^2, LOC.c...), sample((x, y, z) -> x * y - 1, LOC.c...))
        # ∂j τij - ∂i P - f_i with the Voigt slots of τxy (6), τxz (5), τyz (4)
        R = (
            a[1][1] + a[6][2] + a[5][3] - pa[1],
            a[6][1] + a[2][2] + a[4][3] - pa[2],
            a[5][1] + a[4][2] + a[3][3] - pa[3],
        )
        Rx_ref = [R[1] - (1 + y^2 - z) for x in xv[2:(end - 1)], y in yc, z in zc]
        Ry_ref = [R[2] - (0.5 * x + z^2) for x in xc, y in yv[2:(end - 1)], z in zc]
        Rz_ref = [R[3] - (x * y - 1) for x in xc, y in yc, z in zv[2:(end - 1)]]

        stokes = StokesArrays(backend, ni)
        V0 = ntuple(d -> [0.1 * sin(I[1] + d * I[2] - I[3]) for I in CartesianIndices(size(@velocity(stokes)[d]))], 3)
        foreach(copyto!, @velocity(stokes), V0)
        ητ = sample((x, y, z) -> 1.0 + x + 0.5 * y^2 + z, LOC.c...)
        ηdτ = 0.03
        @parallel JR3K.compute_V!(
            @velocity(stokes)..., @residuals(stokes.R)..., dev(P), dev.(f)..., dev.(τ)...,
            dev(ητ), ηdτ, grid._di.center,
        )
        @test Array(stokes.R.Rx) ≈ Rx_ref
        @test Array(stokes.R.Ry) ≈ Ry_ref
        @test Array(stokes.R.Rz) ≈ Rz_ref
        # V += R ηdτ / ητ with ητ averaged onto the face
        av(A, d) = d == 1 ? (A[1:(end - 1), :, :] .+ A[2:end, :, :]) ./ 2 :
            d == 2 ? (A[:, 1:(end - 1), :] .+ A[:, 2:end, :]) ./ 2 : (A[:, :, 1:(end - 1)] .+ A[:, :, 2:end]) ./ 2
        for (d, Rref) in enumerate((Rx_ref, Ry_ref, Rz_ref))
            Vh = Array(@velocity(stokes)[d])
            inner = ntuple(k -> 2:(size(Vh, k) - 1), 3)
            @test Vh[inner...] ≈ V0[d][inner...] .+ Rref .* ηdτ ./ av(ητ, d)
        end
    end

    @testset "principal stresses" begin
        stokes = StokesArrays(backend, ni)
        σ = JR3.PrincipalStress(backend, ni)
        function principal_error(comps)
            foreach(copyto!, @tensor_center(stokes.τ), comps)
            JR3.compute_principal_stresses!(stokes, σ)
            σh = Array.((σ.σ1, σ.σ2, σ.σ3))
            err = 0.0
            for I in CartesianIndices(ni)
                xx, yy, zz, yz, xz, xy = getindex.(comps, Ref(I))
                E = eigen(Symmetric([xx xy xz; xy yy yz; xz yz zz]))
                scale = maximum(abs, E.values)
                # σk = λk e_k in descending order of λ, eigenvectors defined up to sign
                for (k, m) in zip(1:3, 3:-1:1)
                    v = σh[k][:, I]
                    err = max(err, abs(norm(v) - abs(E.values[m])) / scale)
                    err = max(err, abs(abs(dot(v, E.vectors[:, m])) - abs(E.values[m])) / scale)
                end
            end
            return err
        end
        tensor_field(diag0, amp) = ntuple(6) do d
            [(d ≤ 3 ? diag0[d] : 0.0) + amp[d ≤ 3 ? 1 : 2] * sin(d * I[1] + I[2] - 2I[3]) for I in CartesianIndices(ni)]
        end
        # principal values far from symmetric about any one of them; one cell with
        # τxy = τxz = 0 makes the Householder reduction trivial
        comps = tensor_field((0.1, 1.0, 4.0), (0.05, 0.1))
        comps[5][1, 1, 1] = comps[6][1, 1, 1] = 0.0
        @test principal_error(comps) < 1.0e-8
        # principal values close together but not exactly equal, typical of deviatoric stress
        @test principal_error(tensor_field((-1.5, 0.2, 1.8), (0.1, 0.15))) < 1.0e-8
        # simple shear τxy = 1 has principal values 1, 0, -1
        shear = ntuple(d -> fill(d == 6 ? 1.0 : 0.0, ni), 6)
        @test principal_error(shear) < 1.0e-8
    end

    @testset "eigen_symmetric_3x3" begin
        # eigenvalue magnitudes, sorted, always match LinearAlgebra regardless of degeneracy
        function eigval_match(A; atol = 1.0e-8)
            E = eigen(Symmetric(Matrix(A)))
            σ = JR3.eigen_symmetric_3x3(SMatrix{3, 3}(A))
            λs = sort(collect(norm.(σ)); rev = true)
            λref = sort(abs.(E.values); rev = true)
            return maximum(abs, λs .- λref) < atol
        end
        # for a simple (non-repeated) eigenvalue the eigenvector is unique up to sign, so it
        # must also align with LinearAlgebra's choice
        function eigvec_match(A; atol = 1.0e-8)
            E = eigen(Symmetric(Matrix(A)))
            σ = JR3.eigen_symmetric_3x3(SMatrix{3, 3}(A))
            ok = true
            for (k, m) in zip(1:3, 3:-1:1)
                ok &= isapprox(norm(σ[k]), abs(E.values[m]); atol)
                ok &= isapprox(abs(dot(σ[k], E.vectors[:, m])), abs(E.values[m]); atol)
            end
            return ok
        end
        # for a repeated eigenvalue the eigenvector within its eigenspace is not unique, so
        # this checks self-consistency instead: each ek is a unit eigenvector of A (the
        # component of A ek orthogonal to ek vanishes) and the three ek are orthonormal
        function eigen_selfconsistent(A; atol = 1.0e-8)
            σ = JR3.eigen_symmetric_3x3(SMatrix{3, 3}(A))
            Am = Matrix(A)
            es = map(v -> v / norm(v), σ)
            ok = all(e -> norm(Am * e - dot(Am * e, e) * e) < atol, es)
            for k in 1:3, l in (k + 1):3
                ok &= abs(dot(es[k], es[l])) < atol
            end
            return ok
        end

        # simple shear: eigenvalues 1, 0, -1
        @test eigvec_match([0.0 1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0])
        # isotropic stress: a triple eigenvalue, eigenvectors undetermined
        A = 2.0 * Matrix(I, 3, 3)
        @test eigval_match(A)
        @test eigen_selfconsistent(A)
        # two equal eigenvalues (3, 3, 1) in a frame rotated about x, so A is not diagonal
        θ = π / 6
        R = [1.0 0.0 0.0; 0.0 cos(θ) -sin(θ); 0.0 sin(θ) cos(θ)]
        A = R * Diagonal([3.0, 3.0, 1.0]) * R'
        @test eigval_match(A)
        @test eigen_selfconsistent(A)
        # near-degenerate: two eigenvalues very close but distinct
        @test eigvec_match([2.0 0.001 0.0; 0.001 2.0005 0.0; 0.0 0.0 -1.0])
        # large anisotropy: eigenvalues spanning six orders of magnitude
        @test eigvec_match([1.0e3 5.0 0.0; 5.0 1.0 0.0; 0.0 0.0 1.0e-3]; atol = 1.0e-4)
        # dimensional stresses (Pa): the convergence tolerance is relative to ‖A‖
        A = 1.0e8 .* [1.0 0.3 -0.2; 0.3 -0.5 0.1; -0.2 0.1 0.4]
        @test eigvec_match(A; atol = 1.0e-4)
        # a matrix that is not yet diagonal after the allowed sweeps is an error
        @test_throws "did not converge" JR3.eigen_symmetric_3x3(SMatrix{3, 3}(A); max_sweeps = 0)
    end
end

# ---------------------------------------------------------------------------------------
# Pseudo-transient solver: homogeneous plane-strain pure shear, whose exact discrete solution
# is V = (εbg x, -εbg y, 0), P = P0 and a spatially uniform stress
# ---------------------------------------------------------------------------------------

const ni_s = (32, 16, 16)
const εbg = 1.0
quiet(f) = redirect_stdout(f, devnull)

function pure_shear_state(grid)
    stokes = StokesArrays(backend, ni_s)
    X = Array.(grid.xi_vel[1]), Array.(grid.xi_vel[2])
    # exact field on the boundary, perturbed in the interior so the solver has work to do
    Vx = sample((x, y, z) -> εbg * x, X[1]...)
    Vy = sample((x, y, z) -> -εbg * y, X[2]...)
    δx = sample((x, y, z) -> 0.05 * sin(3x) * cos(2y) * cos(z), X[1]...)
    δy = sample((x, y, z) -> 0.05 * cos(2x) * sin(3y) * cos(z), X[2]...)
    inner = (2:(size(Vx, 1) - 1), 2:(size(Vx, 2) - 1), 2:(size(Vx, 3) - 1))
    Vx[inner...] .+= δx[inner...]
    inner = (2:(size(Vy, 1) - 1), 2:(size(Vy, 2) - 1), 2:(size(Vy, 3) - 1))
    Vy[inner...] .+= δy[inner...]
    copyto!(stokes.V.Vx, Vx)
    copyto!(stokes.V.Vy, Vy)
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true, front = true, back = true),
        no_slip = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)
    return stokes, flow_bcs
end

function check_pure_shear(stokes, grid, τxx_expected; rtol = 1.0e-6)
    X = Array.(grid.xi_vel[1]), Array.(grid.xi_vel[2])
    Vx, Vy, Vz = Array(stokes.V.Vx), Array(stokes.V.Vy), Array(stokes.V.Vz)
    @test Vx[:, 2:(end - 1), 2:(end - 1)] ≈ sample((x, y, z) -> εbg * x, X[1][1], X[1][2][2:(end - 1)], X[1][3][2:(end - 1)]) rtol = rtol
    @test Vy[2:(end - 1), :, 2:(end - 1)] ≈ sample((x, y, z) -> -εbg * y, X[2][1][2:(end - 1)], X[2][2], X[2][3][2:(end - 1)]) rtol = rtol
    @test maximum(abs, Vz) < rtol
    @test all(isapprox.(Array(stokes.τ.xx), τxx_expected; rtol))
    @test all(isapprox.(Array(stokes.τ.yy), -τxx_expected; rtol))
    for A in (stokes.τ.zz, stokes.τ.yz, stokes.τ.xz, stokes.τ.xy)
        @test maximum(abs, Array(A)) < rtol * τxx_expected
    end
    return nothing
end

@testset "Stokes solve! 3D" begin
    igg = IGG(init_global_grid(ni_s...; init_MPI = JustRelax.MPI.Initialized() ? false : true, select_device = false, quiet = true)...)
    li = (1.0, 0.5, 0.5)
    grid = Geometry(ni_s, li; origin = (0.0, -0.5, 0.0))
    pt_stokes = PTStokesCoeffs(li, li ./ ni_s; ϵ_rel = 1.0e-12, ϵ_abs = 1.0e-10)
    ρg = @zeros(ni_s...), @zeros(ni_s...), @zeros(ni_s...)
    η0, G0, K0, dt = 1.3, 1.0, 5.0, 0.8
    kw = (; iterMax = 20.0e3, nout = 100, verbose = false)

    @testset "visco-elastic K, G form" begin
        stokes, flow_bcs = pure_shear_state(grid)
        stokes.viscosity.η .= η0
        quiet(() -> solve!(stokes, pt_stokes, grid, flow_bcs, ρg, @fill(K0, ni_s...), @fill(G0, ni_s...), dt, igg; kwargs = kw))
        check_pure_shear(stokes, grid, 2 * ηve(η0, G0, dt) * εbg)
    end

    visc = LinearViscous(; η = η0)
    el = ConstantElasticity(; G = G0, Kb = K0)
    rheo_ve = SetMaterialParams(;
        Phase = 1, Density = ConstantDensity(; ρ = 0.0), Gravity = ConstantGravity(; g = 0.0),
        CompositeRheology = CompositeRheology((visc, el)), Elasticity = el,
    )

    @testset "phase-ratio form, visco-elastic" begin
        stokes, flow_bcs = pure_shear_state(grid)
        phase_ratios = single_phase_ratios(ni_s)
        args = (; T = @zeros(ni_s .+ 2...), P = stokes.P, dt)
        quiet(() -> solve!(stokes, pt_stokes, grid, flow_bcs, ρg, phase_ratios, (rheo_ve,), args, dt, igg; kwargs = kw))
        check_pure_shear(stokes, grid, 2 * ηve(η0, G0, dt) * εbg)
        @test maximum(abs, Array(stokes.P)) < 1.0e-8
    end

    @testset "phase-ratio form, regularised Drucker-Prager" begin
        C, ϕ, ηvp = 0.4, 30.0, 0.2
        pl = DruckerPrager_regularised(; C, ϕ, Ψ = 0.0, η_vp = ηvp)
        rheo_pl = SetMaterialParams(;
            Phase = 1, Density = ConstantDensity(; ρ = 0.0), Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el, pl)), Elasticity = el,
        )
        stokes, flow_bcs = pure_shear_state(grid)
        phase_ratios = single_phase_ratios(ni_s)
        args = (; T = @zeros(ni_s .+ 2...), P = stokes.P, dt)
        quiet(() -> solve!(stokes, pt_stokes, grid, flow_bcs, ρg, phase_ratios, (rheo_pl,), args, dt, igg; kwargs = kw))
        # P = 0, ψ = 0: τII = τy + η_vp λ and τII = τII_M - ηve λ
        ηe = ηve(η0, G0, dt)
        τy, τM = C * cosd(ϕ), 2ηe * εbg
        @test τM > τy
        check_pure_shear(stokes, grid, (ηe * τy + ηvp * τM) / (ηe + ηvp))
        λ = (τM - τy) / (ηe + ηvp)
        @test all(isapprox.(Array(stokes.EII_pl), dt * λ / 2; rtol = 1.0e-6))
    end

    @testset "grid-spacing call forms agree with the Geometry forms" begin
        phase_ratios = single_phase_ratios(ni_s)
        forms = (
            s -> (s.viscosity.η .= η0; (@fill(K0, ni_s...), @fill(G0, ni_s...), dt)),
            s -> (phase_ratios, (rheo_ve,), (; T = @zeros(ni_s .+ 2...), P = s.P, dt), dt),
        )
        for form_args in forms
            results = map((grid, grid.di, grid.di.center)) do g
                stokes, flow_bcs = pure_shear_state(grid)
                quiet(() -> solve!(stokes, pt_stokes, g, flow_bcs, ρg, form_args(stokes)..., igg; kwargs = (; kw..., verbose = true)))
                Array.((stokes.V.Vx, stokes.V.Vy, stokes.τ.xx, stokes.τ.xy))
            end
            @test all(results[2] .≈ results[1])
            @test all(results[3] .≈ results[1])
            @test all(isapprox.(results[1][3], 2 * ηve(η0, G0, dt) * εbg; rtol = 1.0e-6))
        end
    end

    @testset "update_τ_o!" begin
        stokes = StokesArrays(backend, ni_s)
        for (k, A) in enumerate(@stress(stokes))
            copyto!(A, [k + 0.1 * I[1] - 0.01 * I[2] + 0.001 * I[3] for I in CartesianIndices(size(A))])
        end
        JR3K.update_τ_o!(stokes)
        @test all(Array.(@tensor(stokes.τ_o)) .== Array.(@stress(stokes)))
    end

    @testset "single MaterialParams form" begin
        # visco-elastic pure shear through the single-rheology call form
        stokes, flow_bcs = pure_shear_state(grid)
        args = (; T = @zeros(ni_s .+ 2...), P = stokes.P, dt)
        quiet(() -> solve!(stokes, pt_stokes, grid, flow_bcs, ρg, rheo_ve, args, dt, igg; kwargs = kw))
        @test all(isapprox.(Array(stokes.τ.xx), 2 * ηve(η0, G0, dt) * εbg; rtol = 1.0e-6))
    end

    finalize_global_grid(; finalize_MPI = true)
end
