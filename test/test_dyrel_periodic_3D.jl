push!(LOAD_PATH, "..")

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

# phase-1 fraction as a function of the x index, one full period across the domain. Defining it on
# indices rather than coordinates is what makes a shift by `m` cells exact.
@inline _periodic_frac(ix, nx) = 0.5 * (1 + sinpi(2 * ix / nx))

@parallel_indices (i, j, k) function _init_single_phase_periodic_3D!(phases)
    @index phases[1, i, j, k] = 1.0
    return nothing
end

@parallel_indices (i, j, k) function _init_x_center_phases_3D!(phases, nx, m)
    f = _periodic_frac(mod(i - 1 - m, nx) + 0.5, nx)
    @index phases[1, i, j, k] = f
    @index phases[2, i, j, k] = 1 - f
    return nothing
end

@parallel_indices (i, j, k) function _init_x_vertex_phases_3D!(phases, nx, m)
    f = _periodic_frac(mod(i - 1 - m, nx), nx)
    @index phases[1, i, j, k] = f
    @index phases[2, i, j, k] = 1 - f
    return nothing
end

_periodic_flow_bcs() = VelocityBoundaryConditions(;
    free_slip = (left = false, right = false, top = false, bot = false, front = false, back = false),
    no_slip = (left = false, right = false, top = false, bot = false, front = false, back = false),
    periodic = (left = true, right = true, top = false, bot = false, front = true, back = true),
)

# Simple shear driven by the two z-boundary planes, which carry no boundary condition and so keep
# whatever is written on them. `εbg` is the background shear rate.
function _init_simple_shear!(stokes, grid, ni, lz, εbg)
    nx, ny, _ = ni
    zVx = grid.xi_vel[1][3]
    stokes.V.Vx .= PTArray(backend)(
        [2 * εbg * (z - lz / 2) for _i in 1:(nx + 1), _j in 1:(ny + 2), z in zVx]
    )
    # wipe the interior so the solver has to reconstruct it
    @views stokes.V.Vx[:, 2:(end - 1), 2:(end - 1)] .= 0.0
    fill!(stokes.V.Vy, 0.0)
    fill!(stokes.V.Vz, 0.0)
    return nothing
end

_linear_phase(η) = SetMaterialParams(;
    Phase = 1,
    Density = ConstantDensity(; ρ = 0.0),
    Gravity = ConstantGravity(; g = 0.0),
    CompositeRheology = CompositeRheology((LinearViscous(; η = η),)),
)

@testset "DYREL 3D periodic" begin
    # Kept small on purpose. ParallelStencil sizes a GPU block to the range it launches over, and
    # on a cubic range around 12^3 that comes out at 338 threads (ceil(256/13) rows, then ceil(256/169)
    # layers) -- above the 256 the register-heavy fused stress kernel can launch with. A range of
    # at most 256 cells is a single block of exactly that size, so it always fits.
    nx, ny, nz = ni = 6, 5, 4
    init_mpi = !JustRelax.MPI.Initialized()
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI = init_mpi)...)

    @testset "periodic directions get a momentum row" begin
        ni = 6, 5, 4
        nx, ny, nz = ni
        bcs = _periodic_flow_bcs()
        @test JustRelax.periodic_dims(bcs) == (true, true, false)

        stokes = StokesArrays(backend, ni, bcs)
        # x and y carry one row per cell (the seam face is an extra unknown), z one per interior face
        @test size(stokes.R.Rx) == (nx, ny, nz)
        @test size(stokes.R.Ry) == (nx, ny, nz)
        @test size(stokes.R.Rz) == (nx, ny, nz - 1)
        @test JustRelax.periodic_dims(stokes) == (true, true, false)

        dyrel = JustRelax3D.DYREL(backend, ni, JustRelax.periodic_dims(stokes))
        @test size(dyrel.Dx) == (nx, ny, nz)
        @test size(dyrel.Dy) == (nx, ny, nz)
        @test size(dyrel.Dz) == (nx, ny, nz - 1)

        # containers built without the boundary conditions leave the seam rows out
        plain = StokesArrays(backend, ni)
        @test JustRelax.periodic_dims(plain) == (false, false, false)
        @test_throws "do not match the StokesArrays allocation" JustRelax.check_periodic_bcs(
            plain, bcs, igg, ntuple(_ -> 1.0, 3)
        )
    end

    @testset "uniform viscosity recovers exact simple shear" begin
        li = 1.0, 1.0, 1.0
        εbg, lz = 1.0, li[3]
        grid = Geometry(ni, li; origin = (0.0, 0.0, 0.0))
        dt = Inf

        rheology = (_linear_phase(1.0),)
        phase_ratios = PhaseRatios(backend_JP, 1, ni)
        for ratios in (
                phase_ratios.center, phase_ratios.vertex,
                phase_ratios.yz, phase_ratios.xz, phase_ratios.xy,
            )
            @parallel (@idx size(ratios)) _init_single_phase_periodic_3D!(ratios)
        end

        flow_bcs = _periodic_flow_bcs()
        stokes = StokesArrays(backend, ni, flow_bcs)
        args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
        compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

        _init_simple_shear!(stokes, grid, ni, lz, εbg)
        flow_bcs!(stokes, flow_bcs)
        update_halo!(@velocity(stokes)...)

        ρg = ntuple(_ -> @zeros(ni...), Val(3))
        dyrel = JustRelax3D.DYREL(
            backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-8, CFL = 0.99
        )
        solve_DYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
            kwargs = (;
                verbose_PH = false, verbose_DR = false, iterMax = 50.0e3, nout = 50,
                rel_drop = 1.0e-2, linear_viscosity = true, viscosity_cutoff = (-Inf, Inf),
            )
        )

        Vx = Array(stokes.V.Vx)
        zVx = grid.xi_vel[1][3]
        exact = [2 * εbg * (z - lz / 2) for _i in 1:(nx + 1), _j in 1:(ny + 2), z in zVx]
        @test maximum(abs, Vx .- exact) / (εbg * lz) < 1.0e-6
        # the two x-seam planes are the same unknown
        @test Vx[1, :, :] ≈ Vx[end, :, :]
    end

    @testset "shift invariance across the seam" begin
        # A material field that varies in x, translated by `m` cells, must translate the solution
        # by `m` cells. Any stencil that special-cases the seam instead of wrapping breaks this.
        m = 2
        li = 1.0, 1.0, 1.0
        εbg, lz = 1.0, li[3]
        grid = Geometry(ni, li; origin = (0.0, 0.0, 0.0))
        dt = Inf
        rheology = (_linear_phase(1.0), _linear_phase(100.0))

        function solve_shifted(shift)
            phase_ratios = PhaseRatios(backend_JP, 2, ni)
            # `center` and `yz` sit at cell centres in x; `vertex`, `xz`, `xy` at x-vertices
            for ratios in (phase_ratios.center, phase_ratios.yz)
                @parallel (@idx size(ratios)) _init_x_center_phases_3D!(ratios, nx, shift)
            end
            for ratios in (phase_ratios.vertex, phase_ratios.xz, phase_ratios.xy)
                @parallel (@idx size(ratios)) _init_x_vertex_phases_3D!(ratios, nx, shift)
            end

            flow_bcs = _periodic_flow_bcs()
            stokes = StokesArrays(backend, ni, flow_bcs)
            args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
            compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

            _init_simple_shear!(stokes, grid, ni, lz, εbg)
            flow_bcs!(stokes, flow_bcs)
            update_halo!(@velocity(stokes)...)

            ρg = ntuple(_ -> @zeros(ni...), Val(3))
            dyrel = JustRelax3D.DYREL(
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
                Vx = Array(stokes.V.Vx), Vz = Array(stokes.V.Vz),
                η = Array(stokes.viscosity.η),
                τxz = Array(stokes.τ.xz), εxz = Array(stokes.ε.xz),
            )
        end

        a = solve_shifted(0)
        b = solve_shifted(m)

        # sanity: the input really is an exact translation
        @test circshift(a.η, (m, 0, 0)) ≈ b.η

        # drop the duplicated seam plane / the ghost columns so a plain circshift is the comparison
        unwrap_vertex(A) = A[1:(end - 1), :, :]
        unwrap_ghosted(A) = A[2:(end - 1), :, :]
        scale = maximum(abs, a.Vx)

        @test maximum(
            abs, circshift(unwrap_vertex(a.Vx), (m, 0, 0)) .- unwrap_vertex(b.Vx)
        ) / scale < 1.0e-10
        @test maximum(
            abs, circshift(unwrap_ghosted(a.Vz), (m, 0, 0)) .- unwrap_ghosted(b.Vz)
        ) / scale < 1.0e-10

        # the two x-seam planes carry the same strain rate *and* the same stress, which is what
        # fails when the vertex viscosity is averaged over one side of the seam only
        @test a.εxz[1, :, :] ≈ a.εxz[end, :, :]
        @test a.τxz[1, :, :] ≈ a.τxz[end, :, :]
    end
end
