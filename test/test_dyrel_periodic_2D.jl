push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    JustRelax.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    JustRelax.CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
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

# phase-1 fraction as a function of the x index, one full period across the domain. Defining it on
# indices rather than coordinates is what makes a shift by `m` cells exact.
@inline _periodic_frac(ix, nx) = 0.5 * (1 + sinpi(2 * ix / nx))

@parallel_indices (i, j) function _init_x_center_phases_2D!(phases, nx, m)
    f = _periodic_frac(mod(i - 1 - m, nx) + 0.5, nx)
    @index phases[1, i, j] = f
    @index phases[2, i, j] = 1 - f
    return nothing
end

@parallel_indices (i, j) function _init_x_vertex_phases_2D!(phases, nx, m)
    f = _periodic_frac(mod(i - 1 - m, nx), nx)
    @index phases[1, i, j] = f
    @index phases[2, i, j] = 1 - f
    return nothing
end

_periodic_flow_bcs() = VelocityBoundaryConditions(;
    free_slip = (left = false, right = false, top = false, bot = false),
    no_slip = (left = false, right = false, top = false, bot = false),
    periodic = (left = true, right = true, top = false, bot = false),
)

_linear_phase(η) = SetMaterialParams(;
    Phase = 1,
    Density = ConstantDensity(; ρ = 0.0),
    Gravity = ConstantGravity(; g = 0.0),
    CompositeRheology = CompositeRheology((LinearViscous(; η = η),)),
)

@testset "DYREL 2D periodic" begin
    init_mpi = !JustRelax.MPI.Initialized()
    igg = IGG(init_global_grid(24, 24, 1; init_MPI = init_mpi)...)

    @testset "periodic directions get a momentum row" begin
        ni = (6, 5) .* 6 
        nx, ny = ni
        bcs = _periodic_flow_bcs()
        @test JustRelax.periodic_dims(bcs) == (true, false)

        stokes = StokesArrays(backend, ni, bcs)
        # x carries one row per cell (the seam face is an extra unknown), y one per interior face
        @test size(stokes.R.Rx) == (nx, ny)
        @test size(stokes.R.Ry) == (nx, ny - 1)
        @test JustRelax.periodic_dims(stokes) == (true, false)

        dyrel = JustRelax2D.DYREL(backend, ni, JustRelax.periodic_dims(stokes))
        @test size(dyrel.Dx) == (nx, ny)
        @test size(dyrel.Dy) == (nx, ny - 1)

        # containers built without the boundary conditions leave the seam row out
        plain = StokesArrays(backend, ni)
        @test JustRelax.periodic_dims(plain) == (false, false)
        @test_throws "do not match the StokesArrays allocation" JustRelax.check_periodic_bcs(
            plain, bcs, igg, ntuple(_ -> 1.0, 2)
        )
    end

    @testset "shift invariance across the seam" begin
        # A material field that varies in x, translated by `m` cells, must translate the solution
        # by `m` cells. Any stencil that special-cases the seam instead of wrapping breaks this.
        n, m = 32, 7
        ni = n, n
        li = 1.0, 1.0
        εbg, ly = 1.0, li[2]
        grid = Geometry(ni, li; origin = (0.0, 0.0))
        dt = Inf
        rheology = (_linear_phase(1.0), _linear_phase(100.0))

        function solve_shifted(shift)
            phase_ratios = PhaseRatios(backend_JP, 2, ni)
            @parallel (@idx size(phase_ratios.center)) _init_x_center_phases_2D!(
                phase_ratios.center, n, shift
            )
            @parallel (@idx size(phase_ratios.vertex)) _init_x_vertex_phases_2D!(
                phase_ratios.vertex, n, shift
            )

            flow_bcs = _periodic_flow_bcs()
            stokes = StokesArrays(backend, ni, flow_bcs)
            args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
            compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

            # simple shear driven by the two y-boundary planes, which carry no boundary condition
            # and so keep whatever is written on them
            yVx = grid.xi_vel[1][2]
            stokes.V.Vx .= PTArray(backend)(
                [2 * εbg * (y - ly / 2) for _i in 1:(n + 1), y in yVx]
            )
            @views stokes.V.Vx[:, 2:(end - 1)] .= 0.0
            fill!(stokes.V.Vy, 0.0)
            flow_bcs!(stokes, flow_bcs)
            update_halo!(@velocity(stokes)...)

            ρg = @zeros(ni...), @zeros(ni...)
            dyrel = JustRelax2D.DYREL(
                backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-10, CFL = 0.99
            )
            solve_DYREL!(
                stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
                kwargs = (;
                    verbose_PH = false, verbose_DR = false, iterMax = 100.0e3, nout = 50,
                    rel_drop = 1.0e-3, linear_viscosity = true, viscosity_cutoff = (-Inf, Inf),
                )
            )
            return (
                Vx = Array(stokes.V.Vx), Vy = Array(stokes.V.Vy),
                η = Array(stokes.viscosity.η),
                τxy = Array(stokes.τ.xy), εxy = Array(stokes.ε.xy),
            )
        end

        a = solve_shifted(0)
        b = solve_shifted(m)

        # sanity: the input really is an exact translation
        @test circshift(a.η, (m, 0)) ≈ b.η

        # drop the duplicated seam plane / the ghost columns so a plain circshift is the comparison
        unwrap_vertex(A) = A[1:(end - 1), :]
        unwrap_ghosted(A) = A[2:(end - 1), :]
        scale = maximum(abs, a.Vx)

        @test maximum(
            abs, circshift(unwrap_vertex(a.Vx), (m, 0)) .- unwrap_vertex(b.Vx)
        ) / scale < 1.0e-10
        @test maximum(
            abs, circshift(unwrap_ghosted(a.Vy), (m, 0)) .- unwrap_ghosted(b.Vy)
        ) / scale < 1.0e-10

        # the two x-seam planes carry the same strain rate and the same stress
        @test a.εxy[1, :] ≈ a.εxy[end, :]
        @test a.τxy[1, :] ≈ a.τxy[end, :]
    end

    @testset "variational solver" begin
        # Same two checks for `solve_VariationalDYREL!`, once with the domain fully rock and once
        # with a volume fraction that varies in x and wraps across the seam, which is what
        # exercises the ϕ-weighted wrapped stencils.
        n, m = 24, 7
        ni = n, n
        li = 1.0, 1.0
        εbg, ly = 1.0, li[2]
        grid = Geometry(ni, li; origin = (0.0, 0.0))
        dt = Inf

        # ϕ varying in x only and exactly periodic across the seam: on the x-vertex arrays index
        # 1 and index end are the same plane, so they must carry the same fraction. Bounded well
        # away from zero so every row stays in the reduced space and the comparison is not just
        # matching eliminated rows.
        function fill_phi!(ϕ, shift, cut)
            if !cut
                foreach(A -> fill!(A, 1.0), (ϕ.center, ϕ.vertex, ϕ.Vx, ϕ.Vy))
                return ϕ
            end
            f(ix) = 0.5 + 0.5 * _periodic_frac(mod(ix - shift, n), n)
            # Built on the host and assigned as a whole: writing a device array element by element
            # is scalar indexing, which the GPU backends disallow.
            function assign!(A, x0)
                A .= PTArray(backend)([f(i - x0) for i in axes(A, 1), _j in axes(A, 2)])
                return A
            end
            assign!(ϕ.center, 0.5)
            assign!(ϕ.vertex, 1)
            assign!(ϕ.Vx, 1)
            assign!(ϕ.Vy, 0.5)
            return ϕ
        end

        function solve_variational(shift; contrast = true, cut = false)
            rheology = contrast ? (_linear_phase(1.0), _linear_phase(100.0)) :
                (_linear_phase(1.0), _linear_phase(1.0))
            phase_ratios = PhaseRatios(backend_JP, 2, ni)
            @parallel (@idx size(phase_ratios.center)) _init_x_center_phases_2D!(
                phase_ratios.center, n, shift
            )
            @parallel (@idx size(phase_ratios.vertex)) _init_x_vertex_phases_2D!(
                phase_ratios.vertex, n, shift
            )

            flow_bcs = _periodic_flow_bcs()
            stokes = StokesArrays(backend, ni, flow_bcs)
            args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
            compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

            ϕ = RockRatio(backend, ni)
            fill_phi!(ϕ, shift, cut)

            yVx = grid.xi_vel[1][2]
            stokes.V.Vx .= PTArray(backend)(
                [2 * εbg * (y - ly / 2) for _i in 1:(n + 1), y in yVx]
            )
            @views stokes.V.Vx[:, 2:(end - 1)] .= 0.0
            fill!(stokes.V.Vy, 0.0)
            flow_bcs!(stokes, flow_bcs)
            update_halo!(@velocity(stokes)...)

            ρg = @zeros(ni...), @zeros(ni...)
            dyrel = JustRelax2D.DYREL(
                backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; ϵ = 1.0e-10
            )
            out = solve_VariationalDYREL!(
                stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg;
                kwargs = (;
                    verbose_PH = false, verbose_DR = false, iterMax = 100.0e3,
                    total_iterMax = 200.0e3, nout = 50, rel_drop = 1.0e-3,
                    linear_viscosity = true, viscosity_cutoff = (-Inf, Inf),
                )
            )
            return (
                Vx = Array(stokes.V.Vx), Vy = Array(stokes.V.Vy),
                η = Array(stokes.viscosity.η),
                τxy = Array(stokes.τ.xy), εxy = Array(stokes.ε.xy),
                converged = out.converged,
            )
        end

        # uniform viscosity over a full-rock domain reproduces the exact simple shear
        uniform = solve_variational(0; contrast = false)
        @test uniform.converged
        yVx = grid.xi_vel[1][2]
        exact = [2 * εbg * (y - ly / 2) for _i in 1:(n + 1), y in yVx]
        @test maximum(abs, uniform.Vx .- exact) / (εbg * ly) < 1.0e-6
        @test uniform.Vx[1, :] ≈ uniform.Vx[end, :]

        unwrap_vertex(A) = A[1:(end - 1), :]
        unwrap_ghosted(A) = A[2:(end - 1), :]

        for cut in (false, true)
            a = solve_variational(0; cut = cut)
            b = solve_variational(m; cut = cut)
            @test a.converged && b.converged
            @test circshift(a.η, (m, 0)) ≈ b.η

            scale = maximum(abs, a.Vx)
            @test maximum(
                abs, circshift(unwrap_vertex(a.Vx), (m, 0)) .- unwrap_vertex(b.Vx)
            ) / scale < 1.0e-10
            @test maximum(
                abs, circshift(unwrap_ghosted(a.Vy), (m, 0)) .- unwrap_ghosted(b.Vy)
            ) / scale < 1.0e-10
            @test a.εxy[1, :] ≈ a.εxy[end, :]
            @test a.τxy[1, :] ≈ a.τxy[end, :]
        end
    end
end
