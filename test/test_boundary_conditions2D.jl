@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D
using Test, Suppressor

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    CPUBackend
end

@testset "Boundary Conditions 2D" begin
    @suppress begin
        @testset "VelocityBoundaryConditions" begin
            if backend === CPUBackend
                # test incompatible boundary conditions
                @test_throws ErrorException VelocityBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = false),
                    free_slip = (left = false, right = true, top = true, bot = true),
                )
                @test_throws ErrorException VelocityBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = false),
                    free_slip = (left = true, right = true, top = true, bot = false),
                )

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
            else
                @test true === true
            end
        end

        @testset "DisplacementBoundaryConditions" begin
            if backend === CPUBackend
                # test incompatible boundary conditions
                @test_throws ErrorException DisplacementBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = false),
                    free_slip = (left = false, right = true, top = true, bot = true),
                )
                @test_throws ErrorException DisplacementBoundaryConditions(;
                    no_slip = (left = false, right = false, top = false, bot = false),
                    free_slip = (left = true, right = true, top = true, bot = false),
                )
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

            bc = JustRelax.DirichletBoundaryCondition()

            @test all(JustRelax.apply_dirichlet(A, bc) == A)

        end

        @testset "ConstantDirichletBoundaryCondition" begin
            ni = 10, 10
            A = rand(ni...)
            value = 5.0e0
            mask = JustRelax.Mask(ni..., 4:7, 4:7)

            bc = JustRelax.ConstantDirichletBoundaryCondition(value, mask)

            @test JustRelax.apply_dirichlet(A, bc, 1, 1) == A[1, 1]
            @test JustRelax.apply_dirichlet(A, bc, 5, 5) == 5

            bc = JustRelax.ConstantDirichletBoundaryCondition()

            @test all(JustRelax.apply_dirichlet(A, bc) == A)

            @test JustRelax.apply_dirichlet(A, bc, 1, 1) == A[1, 1]
            @test JustRelax.apply_dirichlet(A, bc, 5, 5) == A[5, 5]
        end
    end
end
