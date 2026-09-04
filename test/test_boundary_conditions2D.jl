@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D
using Test, Suppressor
using ParallelStencil

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDABackend
else
    CPUBackend
end

@testset "Boundary Conditions 2D" begin
    @suppress begin
        @testset "TemperatureBoundaryConditions" begin
            T = reshape(collect(Float64, 1:(6 * 7)), 6, 7)
            T0 = copy(T)
            thermal_bcs!(
                T,
                TemperatureBoundaryConditions(;
                    no_flux = (left = false, right = false, top = false, bot = false),
                    constant_value = (left = true, right = true, top = true, bot = true),
                ),
            )

            @test @views T[2:(end - 1), 1] == 2 .- T0[2:(end - 1), 2]
            @test @views T[2:(end - 1), end] == 2 .- T0[2:(end - 1), end - 1]
            @test @views T[1, 2:(end - 1)] == 2 .- T0[2, 2:(end - 1)]
            @test @views T[end, 2:(end - 1)] == 2 .- T0[end - 1, 2:(end - 1)]

            T = reshape(collect(Float64, 1:(6 * 7)), 6, 7)
            T0 = copy(T)
            thermal_bcs!(
                T,
                TemperatureBoundaryConditions(;
                    no_flux = (left = false, right = false, top = false, bot = false),
                    periodic = (left = true, right = true, top = true, bot = true),
                ),
            )

            @test @views T[2:(end - 1), 1] == T0[2:(end - 1), end - 1]
            @test @views T[2:(end - 1), end] == T0[2:(end - 1), 2]
            @test @views T[1, 2:(end - 1)] == T0[end - 1, 2:(end - 1)]
            @test @views T[end, 2:(end - 1)] == T0[2, 2:(end - 1)]

            inactive = (left = false, right = false, top = false, bot = false)
            @test_throws "Periodic boundary conditions must be paired" TemperatureBoundaryConditions(;
                no_flux = inactive,
                periodic = (left = true, right = false, top = false, bot = false),
            )
            @test_throws "Incompatible boundary conditions on the left boundary" TemperatureBoundaryConditions(;
                no_flux = (left = true, right = false, top = false, bot = false),
                periodic = (left = true, right = true, top = false, bot = false),
            )
            @test_throws "Incompatible boundary conditions on the right boundary" TemperatureBoundaryConditions(;
                no_flux = inactive,
                constant_flux = (left = false, right = 1.0, top = false, bot = false),
                periodic = (left = true, right = true, top = false, bot = false),
            )

            # only 4-face (2D) and 6-face (3D) boundary tuples are accepted
            @test_throws "must use 4 (2D) or 6 (3D) faces" TemperatureBoundaryConditions(;
                no_flux = (left = true, right = true, top = false, bot = false, front = false),
            )
            @test_throws "must use 4 (2D) or 6 (3D) faces" TemperatureBoundaryConditions(;
                no_flux = inactive,
                constant_value = (
                    left = false, right = false, front = false, back = false,
                    top = 273.0, bot = 1573.0, extra = false,
                ),
            )
            @test typeof(
                TemperatureBoundaryConditions(; no_flux = inactive)
            ).parameters[end] == 2
        end

        @testset "VelocityBoundaryConditions" begin
            if backend === CPUBackend
                # test incompatible boundary conditions
                @test_throws ErrorException VelocityBoundaryConditions(;
                    no_slip = (left = true, right = false, top = false, bot = false),
                    free_slip = (left = true, right = true, top = true, bot = true),
                )
                @test_throws ErrorException VelocityBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = true),
                    free_slip = (left = true, right = true, top = true, bot = true),
                )
                # a boundary that is neither no_slip nor free_slip carries a prescribed velocity
                @test VelocityBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = false),
                    free_slip = (left = false, right = false, top = false, bot = false),
                ) isa VelocityBoundaryConditions

                n = 5 # number of elements
                Vx, Vy = PTArray(backend)(rand(n + 1, n + 2)), PTArray(backend)(rand(n + 2, n + 1))
                # free-slip
                bcs = VelocityBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = false),
                    free_slip = (left = true, right = true, top = true, bot = true),
                )
                flow_bcs!(bcs, Vx, Vy)

                @test @views Vx[:, 1] == Vx[:, 2]
                @test @views Vx[:, end] == Vx[:, end - 1]
                @test @views Vy[1, :] == Vy[2, :]
                @test @views Vy[end, :] == Vy[end - 1, :]
                @test typeof(bcs) <: AbstractFlowBoundaryConditions
                @test typeof(bcs) <: VelocityBoundaryConditions

                Vx, Vy = PTArray(backend)(reshape(collect(Float64, 1:42), 6, 7)),
                    PTArray(backend)(reshape(collect(Float64, 1:42), 7, 6))
                Vx0, Vy0 = copy(Vx), copy(Vy)
                bcs = VelocityBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = false),
                    free_slip = (left = false, right = false, top = false, bot = false),
                    periodic = (left = true, right = true, top = false, bot = false),
                )
                flow_bcs!(bcs, Vx, Vy)
                @test @views Vx[1, :] == Vx0[end, :]
                @test @views Vx[end, :] == Vx0[end, :]
                @test @views Vy[1, :] == Vy0[end - 1, :]
                @test @views Vy[end, :] == Vy0[2, :]

                @test_throws "Periodic boundary conditions must be paired" VelocityBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = false),
                    free_slip = (left = false, right = false, top = false, bot = false),
                    periodic = (left = true, right = false, top = false, bot = false),
                )
                @test_throws "Incompatible boundary conditions on the left boundary" VelocityBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = false),
                    free_slip = (left = true, right = false, top = false, bot = false),
                    periodic = (left = true, right = true, top = false, bot = false),
                )
                @test_throws "top can't be both periodic and free_surface" VelocityBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = false),
                    free_slip = (left = false, right = false, top = false, bot = false),
                    periodic = (left = false, right = false, top = true, bot = true),
                    free_surface = true,
                )
                # no-slip
                Vx, Vy = PTArray(backend)(rand(n + 1, n + 2)), PTArray(backend)(rand(n + 2, n + 1))
                bcs = VelocityBoundaryConditions(;
                    no_slip = (left = true, right = true, top = true, bot = true),
                    free_slip = (left = false, right = false, top = false, bot = false),
                )
                flow_bcs!(bcs, Vx, Vy)
                @test sum(!iszero(Vx[1, i]) for i in axes(Vx, 2)) == 0
                @test sum(!iszero(Vx[end, i]) for i in axes(Vx, 2)) == 0
                @test sum(!iszero(Vy[i, 1]) for i in axes(Vy, 1)) == 0
                @test sum(!iszero(Vy[i, 1]) for i in axes(Vy, 1)) == 0
                @test @views Vy[1, :] == -Vy[2, :]
                @test @views Vy[end, :] == -Vy[end - 1, :]
                @test @views Vx[:, 1] == -Vx[:, 2]
                @test @views Vx[:, end] == -Vx[:, end - 1]

                # test with StokesArrays
                ni = 5, 5
                stokes = StokesArrays(backend, ni)
                stokes.V.Vx .= PTArray(backend)(rand(n + 1, n + 2))
                stokes.V.Vy .= PTArray(backend)(rand(n + 2, n + 1))
                # free-slip
                flow_bcs = VelocityBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = false),
                    free_slip = (left = true, right = true, top = true, bot = true),
                )
                flow_bcs!(stokes, flow_bcs)

                @test @views stokes.V.Vx[:, 1] == stokes.V.Vx[:, 2]
                @test @views stokes.V.Vx[:, end] == stokes.V.Vx[:, end - 1]
                @test @views stokes.V.Vy[1, :] == stokes.V.Vy[2, :]
                @test @views stokes.V.Vy[end, :] == stokes.V.Vy[end - 1, :]

                # no-slip
                flow_bcs = VelocityBoundaryConditions(;
                    no_slip = (left = true, right = true, top = true, bot = true),
                    free_slip = (left = false, right = false, top = false, bot = false),
                )
                flow_bcs!(stokes, flow_bcs)

                @test sum(!iszero(stokes.V.Vx[1, i]) for i in axes(Vx, 2)) == 0
                @test sum(!iszero(stokes.V.Vx[end, i]) for i in axes(Vx, 2)) == 0
                @test sum(!iszero(stokes.V.Vy[i, 1]) for i in axes(Vy, 1)) == 0
                @test sum(!iszero(stokes.V.Vy[i, 1]) for i in axes(Vy, 1)) == 0
                @test @views stokes.V.Vy[1, :] == -stokes.V.Vy[2, :]
                @test @views stokes.V.Vy[end, :] == -stokes.V.Vy[end - 1, :]
                @test @views stokes.V.Vx[:, 1] == -stokes.V.Vx[:, 2]
                @test @views stokes.V.Vx[:, end] == -stokes.V.Vx[:, end - 1]

                # traction-free top boundary on a non-unit grid
                ni = (4, 3)
                dx, dy = 2.0, 3.0
                stokes = StokesArrays(backend, ni)
                ηeff = fill(5.0, size(stokes.P))
                stokes.P[:, end] .= 10.0
                stokes.V.Vy[:, end - 1] .= 7.0
                for i in axes(stokes.V.Vx, 1)
                    stokes.V.Vx[i, end - 1] = 4.0 * (i - 1) * dx
                end
                flow_bcs = VelocityBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = false),
                    free_slip = (left = false, right = false, top = true, bot = false),
                    free_surface = true,
                )
                JustRelax2D.free_surface_stress_bcs!(stokes, flow_bcs, Val(2))
                JustRelax2D.free_surface_bcs!(
                    stokes, flow_bcs, ηeff, (dx, dy), (dx, dy), Val(2)
                )
                @test stokes.τ.yy[:, end] == stokes.P[:, end]
                @test stokes.V.Vy[2:(end - 1), end] ≈
                    fill(7.0 + (4.0 / 2 + 3 * 10.0 / (4 * 5.0)) * dy, ni[1])
            else
                @test true === true
            end
        end

        @testset "DisplacementBoundaryConditions" begin
            if backend === CPUBackend
                # test incompatible boundary conditions
                @test_throws ErrorException DisplacementBoundaryConditions(;
                    no_slip = (left = true, right = false, top = false, bot = false),
                    free_slip = (left = true, right = true, top = true, bot = true),
                )
                @test_throws ErrorException DisplacementBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = true),
                    free_slip = (left = true, right = true, top = true, bot = true),
                )
                # a boundary that is neither no_slip nor free_slip carries a prescribed velocity
                @test DisplacementBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = false),
                    free_slip = (left = false, right = false, top = false, bot = false),
                ) isa DisplacementBoundaryConditions
                n = 5 # number of elements
                Ux, Uy = PTArray(backend)(rand(n + 1, n + 2)), PTArray(backend)(rand(n + 2, n + 1))
                # free-slip
                bcs1 = DisplacementBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = false),
                    free_slip = (left = true, right = true, top = true, bot = true),
                )
                flow_bcs!(bcs1, Ux, Uy)
                @test @views Ux[:, 1] == Ux[:, 2]
                @test @views Ux[:, end] == Ux[:, end - 1]
                @test @views Uy[1, :] == Uy[2, :]
                @test @views Uy[end, :] == Uy[end - 1, :]
                @test typeof(bcs1) <: AbstractFlowBoundaryConditions
                @test typeof(bcs1) <: DisplacementBoundaryConditions

                Ux, Uy = PTArray(backend)(reshape(collect(Float64, 1:42), 6, 7)),
                    PTArray(backend)(reshape(collect(Float64, 1:42), 7, 6))
                Ux0, Uy0 = copy(Ux), copy(Uy)
                bcs_periodic = DisplacementBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = false),
                    free_slip = (left = false, right = false, top = false, bot = false),
                    periodic = (left = true, right = true, top = false, bot = false),
                )
                flow_bcs!(bcs_periodic, Ux, Uy)
                @test @views Ux[1, :] == Ux0[end, :]
                @test @views Ux[end, :] == Ux0[end, :]
                @test @views Uy[1, :] == Uy0[end - 1, :]
                @test @views Uy[end, :] == Uy0[2, :]

                # no-slip
                Ux, Uy = PTArray(backend)(rand(n + 1, n + 2)), PTArray(backend)(rand(n + 2, n + 1))
                bcs2 = DisplacementBoundaryConditions(;
                    no_slip = (left = true, right = true, top = true, bot = true),
                    free_slip = (left = false, right = false, top = false, bot = false),
                )
                flow_bcs!(bcs2, Ux, Uy)
                @test sum(!iszero(Ux[1, i]) for i in axes(Ux, 2)) == 0
                @test sum(!iszero(Ux[end, i]) for i in axes(Ux, 2)) == 0
                @test sum(!iszero(Uy[i, 1]) for i in axes(Uy, 1)) == 0
                @test sum(!iszero(Uy[i, 1]) for i in axes(Uy, 1)) == 0
                @test @views Uy[1, :] == -Uy[2, :]
                @test @views Uy[end, :] == -Uy[end - 1, :]
                @test @views Ux[:, 1] == -Ux[:, 2]
                @test @views Ux[:, end] == -Ux[:, end - 1]

                # test with StokesArrays
                ni = 5, 5
                stokes = StokesArrays(backend, ni)
                stokes.U.Ux .= PTArray(backend)(rand(n + 1, n + 2))
                stokes.U.Uy .= PTArray(backend)(rand(n + 2, n + 1))
                # free-slip
                flow_bcs = DisplacementBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = false),
                    free_slip = (left = true, right = true, top = true, bot = true),
                )
                flow_bcs!(stokes, flow_bcs)

                @test @views stokes.U.Ux[:, 1] == stokes.U.Ux[:, 2]
                @test @views stokes.U.Ux[:, end] == stokes.U.Ux[:, end - 1]
                @test @views stokes.U.Uy[1, :] == stokes.U.Uy[2, :]
                @test @views stokes.U.Uy[end, :] == stokes.U.Uy[end - 1, :]
                # no-slip
                flow_bcs = DisplacementBoundaryConditions(;
                    no_slip = (left = true, right = true, top = true, bot = true),
                    free_slip = (left = false, right = false, top = false, bot = false),
                )
                flow_bcs!(stokes, flow_bcs)

                @test sum(!iszero(stokes.U.Ux[1, i]) for i in axes(Ux, 2)) == 0
                @test sum(!iszero(stokes.U.Ux[end, i]) for i in axes(Ux, 2)) == 0
                @test sum(!iszero(stokes.U.Uy[i, 1]) for i in axes(Uy, 1)) == 0
                @test sum(!iszero(stokes.U.Uy[i, 1]) for i in axes(Uy, 1)) == 0
                @test @views stokes.U.Uy[1, :] == -stokes.U.Uy[2, :]
                @test @views stokes.U.Uy[end, :] == -stokes.U.Uy[end - 1, :]
                @test @views stokes.U.Ux[:, 1] == -stokes.U.Ux[:, 2]
                @test @views stokes.U.Ux[:, end] == -stokes.U.Ux[:, end - 1]
            else
                @test true === true
            end
        end

        @testset "no-slip acts only on the selected faces" begin
            if backend === CPUBackend
                n = 5
                inactive = (left = false, right = false, top = false, bot = false)

                Vx, Vy = PTArray(backend)(rand(n + 1, n + 2)), PTArray(backend)(rand(n + 2, n + 1))
                Vx0, Vy0 = copy(Vx), copy(Vy)
                flow_bcs!(
                    VelocityBoundaryConditions(;
                        no_slip = (left = true, right = false, top = false, bot = false),
                        free_slip = inactive,
                    ), Vx, Vy,
                )
                @test all(iszero, Vx[1, :])
                @test @views Vy[1, :] == -Vy0[2, :]
                @test @views Vx[2:end, :] == Vx0[2:end, :]
                @test @views Vy[2:end, :] == Vy0[2:end, :]

                Vx, Vy = PTArray(backend)(rand(n + 1, n + 2)), PTArray(backend)(rand(n + 2, n + 1))
                Vx0, Vy0 = copy(Vx), copy(Vy)
                flow_bcs!(
                    VelocityBoundaryConditions(;
                        no_slip = (left = false, right = true, top = false, bot = false),
                        free_slip = inactive,
                    ), Vx, Vy,
                )
                @test all(iszero, Vx[end, :])
                @test @views Vy[end, :] == -Vy0[end - 1, :]
                @test @views Vx[1:(end - 1), :] == Vx0[1:(end - 1), :]
                @test @views Vy[1:(end - 1), :] == Vy0[1:(end - 1), :]

                Vx, Vy = PTArray(backend)(rand(n + 1, n + 2)), PTArray(backend)(rand(n + 2, n + 1))
                Vx0, Vy0 = copy(Vx), copy(Vy)
                flow_bcs!(
                    VelocityBoundaryConditions(;
                        no_slip = (left = false, right = false, top = false, bot = true),
                        free_slip = inactive,
                    ), Vx, Vy,
                )
                @test all(iszero, Vy[:, 1])
                @test @views Vx[:, 1] == -Vx0[:, 2]
                @test @views Vx[:, 2:end] == Vx0[:, 2:end]
                @test @views Vy[:, 2:end] == Vy0[:, 2:end]

                Vx, Vy = PTArray(backend)(rand(n + 1, n + 2)), PTArray(backend)(rand(n + 2, n + 1))
                Vx0, Vy0 = copy(Vx), copy(Vy)
                flow_bcs!(
                    VelocityBoundaryConditions(;
                        no_slip = (left = false, right = false, top = true, bot = false),
                        free_slip = inactive,
                    ), Vx, Vy,
                )
                @test all(iszero, Vy[:, end])
                @test @views Vx[:, end] == -Vx0[:, end - 1]
                @test @views Vx[:, 1:(end - 1)] == Vx0[:, 1:(end - 1)]
                @test @views Vy[:, 1:(end - 1)] == Vy0[:, 1:(end - 1)]
            else
                @test true === true
            end
        end

        @testset "free-slip acts only on the selected faces" begin
            if backend === CPUBackend
                n = 5
                inactive = (left = false, right = false, top = false, bot = false)

                Vx, Vy = PTArray(backend)(rand(n + 1, n + 2)), PTArray(backend)(rand(n + 2, n + 1))
                Vx0, Vy0 = copy(Vx), copy(Vy)
                flow_bcs!(
                    VelocityBoundaryConditions(;
                        no_slip = inactive,
                        free_slip = (left = true, right = false, top = false, bot = false),
                    ), Vx, Vy,
                )
                @test @views Vy[1, :] == Vy0[2, :]
                @test @views Vy[2:end, :] == Vy0[2:end, :]
                @test Vx == Vx0

                Vx, Vy = PTArray(backend)(rand(n + 1, n + 2)), PTArray(backend)(rand(n + 2, n + 1))
                Vx0, Vy0 = copy(Vx), copy(Vy)
                flow_bcs!(
                    VelocityBoundaryConditions(;
                        no_slip = inactive,
                        free_slip = (left = false, right = false, top = false, bot = true),
                    ), Vx, Vy,
                )
                @test @views Vx[:, 1] == Vx0[:, 2]
                @test @views Vx[:, 2:end] == Vx0[:, 2:end]
                @test Vy == Vy0

                Vx, Vy = PTArray(backend)(rand(n + 1, n + 2)), PTArray(backend)(rand(n + 2, n + 1))
                Vx0, Vy0 = copy(Vx), copy(Vy)
                flow_bcs!(
                    VelocityBoundaryConditions(;
                        no_slip = inactive,
                        free_slip = (left = false, right = false, top = true, bot = false),
                    ), Vx, Vy,
                )
                @test @views Vx[:, end] == Vx0[:, end - 1]
                @test @views Vx[:, 1:(end - 1)] == Vx0[:, 1:(end - 1)]
                @test Vy == Vy0
            else
                @test true === true
            end
        end

        @testset "flow boundary conditions on a non-square grid" begin
            if backend === CPUBackend
                nx, ny = 4, 9
                all_on = (left = true, right = true, top = true, bot = true)
                inactive = (left = false, right = false, top = false, bot = false)

                Vx, Vy = PTArray(backend)(rand(nx + 1, ny + 2)), PTArray(backend)(rand(nx + 2, ny + 1))
                Vx0, Vy0 = copy(Vx), copy(Vy)
                flow_bcs!(
                    VelocityBoundaryConditions(; no_slip = all_on, free_slip = inactive),
                    Vx, Vy,
                )
                @test all(iszero, Vx[1, :])
                @test all(iszero, Vx[end, :])
                @test all(iszero, Vy[:, 1])
                @test all(iszero, Vy[:, end])
                # corners are written twice, so compare away from them
                @test @views Vy[1, 2:(end - 1)] == -Vy0[2, 2:(end - 1)]
                @test @views Vy[end, 2:(end - 1)] == -Vy0[end - 1, 2:(end - 1)]
                @test @views Vx[2:(end - 1), 1] == -Vx0[2:(end - 1), 2]
                @test @views Vx[2:(end - 1), end] == -Vx0[2:(end - 1), end - 1]

                Vx, Vy = PTArray(backend)(rand(nx + 1, ny + 2)), PTArray(backend)(rand(nx + 2, ny + 1))
                Vx0, Vy0 = copy(Vx), copy(Vy)
                flow_bcs!(
                    VelocityBoundaryConditions(; no_slip = inactive, free_slip = all_on),
                    Vx, Vy,
                )
                @test @views Vx[:, 1] == Vx0[:, 2]
                @test @views Vx[:, end] == Vx0[:, end - 1]
                @test @views Vy[1, :] == Vy0[2, :]
                @test @views Vy[end, :] == Vy0[end - 1, :]
            else
                @test true === true
            end
        end

        @testset "mixed no-slip and free-slip faces" begin
            if backend === CPUBackend
                n = 5
                Vx, Vy = PTArray(backend)(rand(n + 1, n + 2)), PTArray(backend)(rand(n + 2, n + 1))
                Vx0, Vy0 = copy(Vx), copy(Vy)
                flow_bcs!(
                    VelocityBoundaryConditions(;
                        no_slip = (left = true, right = false, top = false, bot = false),
                        free_slip = (left = false, right = true, top = true, bot = true),
                    ), Vx, Vy,
                )
                # no-slip is applied before free-slip, so the left face survives
                @test all(iszero, Vx[1, :])
                @test @views Vy[1, :] == -Vy0[2, :]
                @test @views Vy[end, :] == Vy0[end - 1, :]
                @test @views Vx[:, 1] == Vx[:, 2]
                @test @views Vx[:, end] == Vx[:, end - 1]
            else
                @test true === true
            end
        end

        @testset "inactive faces leave the arrays untouched" begin
            if backend === CPUBackend
                n = 5
                inactive = (left = false, right = false, top = false, bot = false)
                Vx, Vy = PTArray(backend)(rand(n + 1, n + 2)), PTArray(backend)(rand(n + 2, n + 1))
                Vx0, Vy0 = copy(Vx), copy(Vy)
                flow_bcs!(
                    VelocityBoundaryConditions(; no_slip = inactive, free_slip = inactive),
                    Vx, Vy,
                )
                @test Vx == Vx0
                @test Vy == Vy0
            else
                @test true === true
            end
        end

        @testset "DirichletBoundaryCondition" begin
            ni = 10, 10
            A = rand(ni...)
            value = zeros(ni...)
            value[4:7, 4:7] .= 5

            bc = JustRelax.DirichletBoundaryCondition(value)

            @test all(JustRelax.apply_dirichlet(A, bc)[4:7, 4:7] .== 5)

            A = rand(ni...)
            @test JustRelax.apply_dirichlet(A, bc, 1, 1) == A[1, 1]
            @test JustRelax.apply_dirichlet(A, bc, 5, 5) == 5
            @test JustRelax.isdirichlet(bc, 1, 1) === false
            @test JustRelax.isdirichlet(bc, 5, 5) === true

            bc2 = JustRelax.DirichletBoundaryCondition()

            @test all(JustRelax.apply_dirichlet(A, bc2) == A)

        end

        @testset "ConstantDirichletBoundaryCondition" begin
            ni = 10, 10
            A = rand(ni...)
            value = 5.0e0
            mask = JustRelax.Mask(ni..., 4:7, 4:7)

            bc = JustRelax.ConstantDirichletBoundaryCondition(value, mask)

            @test JustRelax.apply_dirichlet(A, bc, 1, 1) == A[1, 1]
            @test JustRelax.apply_dirichlet(A, bc, 5, 5) == 5
            @test JustRelax.isdirichlet(bc, 1, 1) === false
            @test JustRelax.isdirichlet(bc, 5, 5) === true

            bc2 = JustRelax.ConstantDirichletBoundaryCondition()

            @test all(JustRelax.apply_dirichlet(A, bc2) == A)

            @test JustRelax.apply_dirichlet(A, bc2, 1, 1) == A[1, 1]
        end

        @testset "Dirichlet factory + ConstantArray" begin
            ni = 10, 10

            # 4 dispatch paths of `Dirichlet(constant, mask)`
            bc_empty = JustRelax.Dirichlet(nothing, nothing)
            @test bc_empty isa JustRelax.DirichletBoundaryCondition{Nothing, Nothing}

            mask_arr = zeros(ni...); mask_arr[3:5, 3:5] .= 7
            bc_mask = JustRelax.Dirichlet(nothing, mask_arr)
            @test bc_mask isa JustRelax.DirichletBoundaryCondition

            bc_const = JustRelax.Dirichlet(3.0, mask_arr)
            @test bc_const isa JustRelax.ConstantDirichletBoundaryCondition

            # kwarg form
            bc_kw = JustRelax.Dirichlet(; constant = nothing, mask = nothing)
            @test bc_kw isa JustRelax.DirichletBoundaryCondition{Nothing, Nothing}

            bc_nt = JustRelax.Dirichlet((; constant = 2.5, mask = mask_arr))
            @test bc_nt isa JustRelax.ConstantDirichletBoundaryCondition

            # ConstantArray
            ca = JustRelax.ConstantArray(4.2)
            @test ca[1, 1] === 4.2
            @test ca[7, 9, 3] === 4.2

            JustRelax.ConstantArray(4.2)[1, 1] = 0.0
            io = IOBuffer()
            show(io, MIME"text/plain"(), ca)
            @test occursin("ConstantArray", String(take!(io)))
            io = IOBuffer()
            show(io, ca)
            @test occursin("ConstantArray", String(take!(io)))
        end

        @testset "apply_dirichlet!/isdirichlet Nothing branches" begin
            ni = 6, 6
            A = rand(ni...)
            A_copy = copy(A)
            bc_empty = JustRelax.DirichletBoundaryCondition()

            # mutating: no-op (array unchanged)
            JustRelax.apply_dirichlet!(A, bc_empty)
            @test A == A_copy
            JustRelax.apply_dirichlet!(A, bc_empty, 2, 2)
            @test A == A_copy

            # non-mutating: returns A or A[inds...]
            @test JustRelax.apply_dirichlet(A, bc_empty) === A
            @test JustRelax.apply_dirichlet(A, bc_empty, 3, 3) == A[3, 3]

            # isdirichlet false for the all-nothing BC
            @test JustRelax.isdirichlet(bc_empty, 1, 1) === false

            # The mutating apply_dirichlet! with a real (array-backed) BC
            value = zeros(ni...); value[2:4, 2:4] .= 9
            bc = JustRelax.DirichletBoundaryCondition(value)
            B = zeros(ni...)
            JustRelax.apply_dirichlet!(B, bc)
            @test all(B[2:4, 2:4] .== 9)
            # indexed apply_dirichlet! variant
            B2 = zeros(ni...)
            JustRelax.apply_dirichlet!(B2, bc, 3, 3)
            @test B2[3, 3] == 9
        end

        @testset "pure shear boundary condition" begin
            stokes = StokesArrays(backend, (3, 4))
            xci = (collect(1.0:3.0), collect(1.0:4.0))
            xvi = (collect(1.0:4.0), collect(1.0:5.0))
            pureshear_bc!(stokes, xci, xvi, 2.0)

            @test Array(@view stokes.V.Vx[:, 2:(end - 1)]) == [2.0 * x for x in xvi[1], _ in xci[2]]
            @test Array(@view stokes.V.Vy[2:(end - 1), :]) == [-2.0 * y for _ in xci[1], y in xvi[2]]
        end

        @testset "simple shear boundary condition" begin
            stokes = StokesArrays(backend, (3, 4))
            xci = (collect(1.0:3.0), collect(1.0:4.0))
            xvi = (collect(1.0:4.0), collect(1.0:5.0))
            simpleshear_bc!(stokes, xci, xvi, 2.0)

            @test Array(@view stokes.V.Vx[:, 2:(end - 1)]) == [2.0 * y for _ in xvi[1], y in xci[2]]
            @test all(iszero, Array(@view stokes.V.Vy[2:(end - 1), :]))
        end

        @testset "background shear fields leave ghost layers untouched" begin
            xci = (LinRange(0.5, 2.5, 3), LinRange(0.5, 3.5, 4))
            xvi = (LinRange(0.0, 3.0, 4), LinRange(0.0, 4.0, 5))

            for bc! in (pureshear_bc!, simpleshear_bc!)
                stokes = StokesArrays(backend, (3, 4))
                stokes.V.Vx .= NaN
                stokes.V.Vy .= NaN
                bc!(stokes, xci, xvi, 2.0)

                Vx, Vy = Array(stokes.V.Vx), Array(stokes.V.Vy)
                @test all(isnan, @view Vx[:, 1])
                @test all(isnan, @view Vx[:, end])
                @test all(isnan, @view Vy[1, :])
                @test all(isnan, @view Vy[end, :])
                @test !any(isnan, @view Vx[:, 2:(end - 1)])
                @test !any(isnan, @view Vy[2:(end - 1), :])
            end
        end

        @testset "background shear fields accept the legacy backend argument" begin
            xci = (LinRange(0.5, 2.5, 3), LinRange(0.5, 3.5, 4))
            xvi = (LinRange(0.0, 3.0, 4), LinRange(0.0, 4.0, 5))

            for bc! in (pureshear_bc!, simpleshear_bc!)
                stokes4 = StokesArrays(backend, (3, 4))
                stokes5 = StokesArrays(backend, (3, 4))
                bc!(stokes4, xci, xvi, 2.0)
                bc!(stokes5, xci, xvi, 2.0, backend)

                @test Array(stokes4.V.Vx) == Array(stokes5.V.Vx)
                @test Array(stokes4.V.Vy) == Array(stokes5.V.Vy)
            end
        end

        @testset "background shear fields scale linearly with the imposed rate" begin
            xci = (LinRange(0.5, 2.5, 3), LinRange(0.5, 3.5, 4))
            xvi = (LinRange(0.0, 3.0, 4), LinRange(0.0, 4.0, 5))

            for bc! in (pureshear_bc!, simpleshear_bc!)
                unit = StokesArrays(backend, (3, 4))
                scaled = StokesArrays(backend, (3, 4))
                bc!(unit, xci, xvi, 1.0)
                bc!(scaled, xci, xvi, 3.0)

                @test @views Array(scaled.V.Vx)[:, 2:(end - 1)] ≈
                    3 .* Array(unit.V.Vx)[:, 2:(end - 1)]
                @test @views Array(scaled.V.Vy)[2:(end - 1), :] ≈
                    3 .* Array(unit.V.Vy)[2:(end - 1), :]
            end
        end
    end
end
