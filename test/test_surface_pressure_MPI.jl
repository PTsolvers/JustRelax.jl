using JustRelax, JustRelax.JustRelax2D
using ParallelStencil, Test

@init_parallel_stencil(Threads, Float64, 2)
const JR = JustRelax.JustRelax2D

function load_residual!(stokes, pressure, ϕ, grid)
    ni = size(stokes.P)
    ρg = @zeros(ni...), @zeros(ni...)
    @parallel (@idx ni) JR.compute_PH_residual_V!(
        @residuals(stokes.R)..., @velocity(stokes)..., pressure, stokes.ΔPψ,
        @stress(stokes)..., ρg..., ϕ, grid._di.center, grid._di.vertex, 0.0,
    )
    return map(copy, @residuals(stokes.R))
end

@testset "Surface pressure MPI halos" begin
    ni = (8, 8)
    for direction in (1, 2)
        igg = IGG(
            init_global_grid(
                ni..., 1; dimx = direction == 1 ? 0 : 1, dimy = direction == 2 ? 0 : 1,
                dimz = 1, init_MPI = !JustRelax.MPI.Initialized(), select_device = false,
            )...
        )
        try
            grid = Geometry(ni, Float64.((nx_g(), ny_g())))
            stokes = StokesArrays(CPUBackend, ni)
            ϕ = RockRatio(CPUBackend, ni)
            offset = igg.coords[2] * (ni[2] - 2)
            h = direction == 2 ? ni[2] - 2 + 0.75 : 4.75
            for f in (:center, :Vx, :Vy, :vertex)
                A = getfield(ϕ, f)
                shift = f in (:Vy, :vertex) ? 1.5 : 1.0
                A .= [clamp(h - (offset + j - shift), 0.0, 1.0) for i in axes(A, 1), j in axes(A, 2)]
            end
            prescribed = [
                2.0 + 0.1 * (igg.coords[1] * (ni[1] - 2) + i) + 0.2 * (offset + j)
                    for i in 1:ni[1], j in 1:ni[2]
            ]
            reference = load_residual!(stokes, JR.SurfacePressure(stokes.P, prescribed), ϕ, grid)
            # Invalidate only rank-interior halos. prepare_surface_pressure must
            # repair its snapshot and all masks read by the reduced load stencil.
            for A in (prescribed, ϕ.center, ϕ.Vx, ϕ.Vy)
                if igg.coords[direction] > 0
                    selectdim(A, direction, firstindex(A, direction)) .= 0.0
                end
                if igg.coords[direction] < igg.dims[direction] - 1
                    selectdim(A, direction, lastindex(A, direction)) .= 0.0
                end
            end
            caller_field = copy(prescribed)
            bcs = VelocityBoundaryConditions(; free_slip = (left = true, right = true, top = true, bot = true))
            pressure = JR.prepare_surface_pressure(stokes.P, prescribed, ϕ, bcs)
            actual = load_residual!(stokes, pressure, ϕ, grid)
            @test prescribed == caller_field
            @test actual[1] ≈ reference[1] atol = 1.0e-12
            @test actual[2] ≈ reference[2] atol = 1.0e-12
        finally
            finalize_global_grid(; finalize_MPI = false)
        end
    end
end
