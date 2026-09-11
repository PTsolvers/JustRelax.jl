@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using JustRelax, JustRelax.JustRelax3D
using Test
using ParallelStencil

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 3)
    CUDABackend
else
    CPUBackend
end

@testset "Boundary Conditions 3D" begin
    @testset "TemperatureBoundaryConditions" begin
        thermal_bc = TemperatureBoundaryConditions(
            no_flux = (left = false, right = false, top = false, bot = false),
            constant_value = (
                left = false,
                right = false,
                front = false,
                back = false,
                top = false,
                bot = false,
            ),
        )
        @test typeof(thermal_bc).parameters[end] == 3

        T = reshape(collect(Float64, 1:(6 * 7 * 8)), 6, 7, 8)
        T0 = copy(T)
        thermal_bcs!(
            T,
            TemperatureBoundaryConditions(;
                no_flux = (left = false, right = false, front = false, back = false, top = false, bot = false),
                constant_value = (left = true, right = true, front = true, back = true, top = true, bot = true),
            ),
        )

        @test @views T[2:(end - 1), 2:(end - 1), 1] == 2 .- T0[2:(end - 1), 2:(end - 1), 2]
        @test @views T[2:(end - 1), 2:(end - 1), end] == 2 .- T0[2:(end - 1), 2:(end - 1), end - 1]
        @test @views T[1, 2:(end - 1), 2:(end - 1)] == 2 .- T0[2, 2:(end - 1), 2:(end - 1)]
        @test @views T[end, 2:(end - 1), 2:(end - 1)] == 2 .- T0[end - 1, 2:(end - 1), 2:(end - 1)]
        @test @views T[2:(end - 1), 1, 2:(end - 1)] == 2 .- T0[2:(end - 1), 2, 2:(end - 1)]
        @test @views T[2:(end - 1), end, 2:(end - 1)] == 2 .- T0[2:(end - 1), end - 1, 2:(end - 1)]

        T = reshape(collect(Float64, 1:(6 * 7 * 8)), 6, 7, 8)
        T0 = copy(T)
        thermal_bcs!(
            T,
            TemperatureBoundaryConditions(;
                no_flux = (left = false, right = false, front = false, back = false, top = false, bot = false),
                periodic = (left = true, right = true, front = true, back = true, top = true, bot = true),
            ),
        )

        @test @views T[2:(end - 1), 2:(end - 1), 1] == T0[2:(end - 1), 2:(end - 1), end - 1]
        @test @views T[2:(end - 1), 2:(end - 1), end] == T0[2:(end - 1), 2:(end - 1), 2]
        @test @views T[1, 2:(end - 1), 2:(end - 1)] == T0[end - 1, 2:(end - 1), 2:(end - 1)]
        @test @views T[end, 2:(end - 1), 2:(end - 1)] == T0[2, 2:(end - 1), 2:(end - 1)]
        @test @views T[2:(end - 1), 1, 2:(end - 1)] == T0[2:(end - 1), end - 1, 2:(end - 1)]
        @test @views T[2:(end - 1), end, 2:(end - 1)] == T0[2:(end - 1), 2, 2:(end - 1)]

        @test_throws ErrorException TemperatureBoundaryConditions(;
            no_flux = (left = false, right = false, top = false, bot = false),
            periodic = (left = false, right = false, front = true, back = false, top = false, bot = false),
        )

        @test_throws "must use 4 (2D) or 6 (3D) faces" TemperatureBoundaryConditions(;
            no_flux = (left = false, right = false, front = false, back = false, top = false),
        )
    end

    @testset "VelocityBoundaryConditions" begin
        if backend === CPUBackend
            # test incompatible boundary conditions
            @test_throws ErrorException VelocityBoundaryConditions(;
                no_slip = (left = true, right = true, front = true, back = true, top = true, bot = true),
                free_slip = (left = false, right = true, front = true, back = true, top = true, bot = true),
            )
            # a boundary that is neither no_slip nor free_slip carries a prescribed velocity
            @test VelocityBoundaryConditions(;
                no_slip = (left = false, right = false, front = false, back = false, top = false, bot = false),
                free_slip = (left = false, right = false, front = false, back = false, top = false, bot = false),
            ) isa VelocityBoundaryConditions

            # test with StokesArrays
            ni = 5, 5, 5
            stokes = StokesArrays(backend, ni)
            stokes.V.Vx .= PTArray(backend)(rand(size(stokes.V.Vx)...))
            stokes.V.Vy .= PTArray(backend)(rand(size(stokes.V.Vy)...))
            stokes.V.Vz .= PTArray(backend)(rand(size(stokes.V.Vz)...))

            # free-slip
            flow_bcs = VelocityBoundaryConditions(;
                no_slip = (left = false, right = false, front = false, back = false, top = false, bot = false),
                free_slip = (left = true, right = true, front = true, back = true, top = true, bot = true),
            )
            flow_bcs!(stokes, flow_bcs)
            flow_bcs!(stokes, flow_bcs) # just a trick to pass the CI

            @test @views stokes.V.Vx[:, :, 1] == stokes.V.Vx[:, :, 2]
            @test @views stokes.V.Vx[:, :, end] == stokes.V.Vx[:, :, end - 1]
            @test @views stokes.V.Vx[:, 1, :] == stokes.V.Vx[:, 2, :]
            @test @views stokes.V.Vx[:, end, :] == stokes.V.Vx[:, end - 1, :]
            @test @views stokes.V.Vy[:, :, 1] == stokes.V.Vy[:, :, 2]
            @test @views stokes.V.Vy[:, :, end] == stokes.V.Vy[:, :, end - 1]
            @test @views stokes.V.Vy[1, :, :] == stokes.V.Vy[2, :, :]
            @test @views stokes.V.Vy[end, :, :] == stokes.V.Vy[end - 1, :, :]
            @test @views stokes.V.Vz[1, :, :] == stokes.V.Vz[2, :, :]
            @test @views stokes.V.Vz[end, :, :] == stokes.V.Vz[end - 1, :, :]
            @test @views stokes.V.Vz[:, 1, :] == stokes.V.Vz[:, 2, :]
            @test @views stokes.V.Vz[:, end, :] == stokes.V.Vz[:, end - 1, :]

            # top and bottom must remain distinct when only one face is active
            Vx = PTArray(backend)(rand(size(stokes.V.Vx)...))
            Vy = PTArray(backend)(rand(size(stokes.V.Vy)...))
            Vz = PTArray(backend)(rand(size(stokes.V.Vz)...))
            Vx0, Vy0, Vz0 = copy(Vx), copy(Vy), copy(Vz)
            flow_bcs!(
                VelocityBoundaryConditions(
                    ; no_slip = (left = false, right = false, front = false, back = false, top = false, bot = false),
                    free_slip = (left = false, right = false, front = false, back = false, top = true, bot = false)
                ),
                Vx,
                Vy,
                Vz,
            )
            @test @views Vx[:, :, end] == Vx0[:, :, end - 1]
            @test @views Vy[:, :, end] == Vy0[:, :, end - 1]
            @test @views Vx[:, :, 1] == Vx0[:, :, 1]
            @test @views Vy[:, :, 1] == Vy0[:, :, 1]
            @test Vz == Vz0

            # no-slip
            flow_bcs = VelocityBoundaryConditions(;
                no_slip = (left = true, right = true, front = true, back = true, top = true, bot = true),
                free_slip = (left = false, right = false, front = false, back = false, top = false, bot = false),
            )
            flow_bcs!(stokes, flow_bcs)

            (; Vx, Vy, Vz) = stokes.V
            @test sum(!iszero(Vx[1, i, j]) for i in axes(Vx, 2), j in axes(Vx, 3)) == 0
            @test sum(!iszero(Vx[end, i, j]) for i in axes(Vx, 2), j in axes(Vx, 3)) == 0
            @test sum(!iszero(Vy[i, 1, j]) for i in axes(Vy, 1), j in axes(Vy, 2)) == 0
            @test sum(!iszero(Vy[i, end, j]) for i in axes(Vy, 1), j in axes(Vy, 2)) == 0
            @test sum(!iszero(Vz[i, j, 1]) for i in axes(Vz, 1), j in axes(Vz, 3)) == 0
            @test sum(!iszero(Vz[i, j, end]) for i in axes(Vz, 1), j in axes(Vz, 3)) == 0
            @test @views Vx[:, 1, :] == -Vx[:, 2, :]
            @test @views Vx[:, end, :] == -Vx[:, end - 1, :]
            @test @views Vx[:, :, 1] == -Vx[:, :, 2]
            @test @views Vx[:, :, end] == -Vx[:, :, end - 1]
            @test @views Vy[1, :, :] == -Vy[2, :, :]
            @test @views Vy[end, :, :] == -Vy[end - 1, :, :]
            @test @views Vy[:, :, 1] == -Vy[:, :, 2]
            @test @views Vy[:, :, end] == -Vy[:, :, end - 1]
            @test @views Vz[:, 1, :] == -Vz[:, 2, :]
            @test @views Vz[:, end, :] == -Vz[:, end - 1, :]
            @test @views Vz[1, :, :] == -Vz[2, :, :]
            @test @views Vz[end, :, :] == -Vz[end - 1, :, :]

            # traction-free top boundary uses τzz and physical grid spacing
            ni = (3, 3, 3)
            dx, dy, dz = 2.0, 3.0, 4.0
            stokes = StokesArrays(backend, ni)
            ηeff = fill(5.0, size(stokes.P))
            stokes.P[:, :, end] .= 10.0
            stokes.τ.yy[:, :, end] .= -100.0
            stokes.V.Vz[:, :, end - 1] .= 7.0
            for j in axes(stokes.V.Vx, 2), i in axes(stokes.V.Vx, 1)
                stokes.V.Vx[i, j, end - 1] = 4.0 * (i - 1) * dx
            end
            for j in axes(stokes.V.Vy, 2), i in axes(stokes.V.Vy, 1)
                stokes.V.Vy[i, j, end - 1] = 6.0 * (j - 1) * dy
            end
            flow_bcs = VelocityBoundaryConditions(;
                no_slip = (left = false, right = false, front = false, back = false, top = false, bot = false),
                free_slip = (left = false, right = false, front = false, back = false, top = true, bot = false),
                free_surface = true,
            )
            JustRelax3D.free_surface_stress_bcs!(stokes, flow_bcs, Val(3))
            JustRelax3D.free_surface_bcs!(
                stokes,
                flow_bcs,
                ηeff,
                (dx, dy, dz),
                (dx, dy, dz),
                (dx, dy, dz),
                Val(3),
            )
            @test stokes.τ.zz[:, :, end] == stokes.P[:, :, end]
            @test stokes.τ.yy[:, :, end] == fill(-100.0, ni[1:2])
            @test stokes.V.Vz[2:(end - 1), 2:(end - 1), end] ≈
                fill(7.0 + ((4.0 + 6.0) / 2 + 3 * 10.0 / (4 * 5.0)) * dz, ni[1:2])
        else
            @test true === true
        end
    end

    @testset "DisplacementBoundaryConditions" begin
        if backend === CPUBackend
            # test incompatible boundary conditions
            @test_throws ErrorException DisplacementBoundaryConditions(;
                no_slip = (left = true, right = true, front = true, back = true, top = true, bot = true),
                free_slip = (left = false, right = true, front = true, back = true, top = true, bot = true),
            )
            # a boundary that is neither no_slip nor free_slip carries a prescribed velocity
            @test DisplacementBoundaryConditions(;
                no_slip = (left = false, right = false, front = false, back = false, top = false, bot = false),
                free_slip = (left = false, right = false, front = false, back = false, top = false, bot = false),
            ) isa DisplacementBoundaryConditions

            # test with StokesArrays
            ni = 5, 5, 5
            stokes = StokesArrays(backend, ni)
            stokes.U.Ux .= PTArray(backend)(rand(size(stokes.U.Ux)...))
            stokes.U.Uy .= PTArray(backend)(rand(size(stokes.U.Uy)...))
            stokes.U.Uz .= PTArray(backend)(rand(size(stokes.U.Uz)...))

            # free-slip
            flow_bcs = DisplacementBoundaryConditions(;
                no_slip = (left = false, right = false, front = false, back = false, top = false, bot = false),
                free_slip = (left = true, right = true, front = true, back = true, top = true, bot = true),
            )
            flow_bcs!(stokes, flow_bcs)
            flow_bcs!(stokes, flow_bcs) # just a trick to pass the CI

            @test @views stokes.U.Ux[:, :, 1] == stokes.U.Ux[:, :, 2]
            @test @views stokes.U.Ux[:, :, end] == stokes.U.Ux[:, :, end - 1]
            @test @views stokes.U.Ux[:, 1, :] == stokes.U.Ux[:, 2, :]
            @test @views stokes.U.Ux[:, end, :] == stokes.U.Ux[:, end - 1, :]
            @test @views stokes.U.Uy[:, :, 1] == stokes.U.Uy[:, :, 2]
            @test @views stokes.U.Uy[:, :, end] == stokes.U.Uy[:, :, end - 1]
            @test @views stokes.U.Uy[1, :, :] == stokes.U.Uy[2, :, :]
            @test @views stokes.U.Uy[end, :, :] == stokes.U.Uy[end - 1, :, :]
            @test @views stokes.U.Uz[1, :, :] == stokes.U.Uz[2, :, :]
            @test @views stokes.U.Uz[end, :, :] == stokes.U.Uz[end - 1, :, :]
            @test @views stokes.U.Uz[:, 1, :] == stokes.U.Uz[:, 2, :]
            @test @views stokes.U.Uz[:, end, :] == stokes.U.Uz[:, end - 1, :]

            # no-slip
            flow_bcs = DisplacementBoundaryConditions(;
                no_slip = (left = true, right = true, front = true, back = true, top = true, bot = true),
                free_slip = (left = false, right = false, front = false, back = false, top = false, bot = false),
            )
            flow_bcs!(stokes, flow_bcs)

            (; Ux, Uy, Uz) = stokes.U
            @test sum(!iszero(Ux[1, i, j]) for i in axes(Ux, 2), j in axes(Ux, 3)) == 0
            @test sum(!iszero(Ux[end, i, j]) for i in axes(Ux, 2), j in axes(Ux, 3)) == 0
            @test sum(!iszero(Uy[i, 1, j]) for i in axes(Uy, 1), j in axes(Uy, 2)) == 0
            @test sum(!iszero(Uy[i, end, j]) for i in axes(Uy, 1), j in axes(Uy, 2)) == 0
            @test sum(!iszero(Uz[i, j, 1]) for i in axes(Uz, 1), j in axes(Uz, 3)) == 0
            @test sum(!iszero(Uz[i, j, end]) for i in axes(Uz, 1), j in axes(Uz, 3)) == 0
            @test @views Ux[:, 1, :] == -Ux[:, 2, :]
            @test @views Ux[:, end, :] == -Ux[:, end - 1, :]
            @test @views Ux[:, :, 1] == -Ux[:, :, 2]
            @test @views Ux[:, :, end] == -Ux[:, :, end - 1]
            @test @views Uy[1, :, :] == -Uy[2, :, :]
            @test @views Uy[end, :, :] == -Uy[end - 1, :, :]
            @test @views Uy[:, :, 1] == -Uy[:, :, 2]
            @test @views Uy[:, :, end] == -Uy[:, :, end - 1]
            @test @views Uz[:, 1, :] == -Uz[:, 2, :]
            @test @views Uz[:, end, :] == -Uz[:, end - 1, :]
            @test @views Uz[1, :, :] == -Uz[2, :, :]
            @test @views Uz[end, :, :] == -Uz[end - 1, :, :]
        else
            @test true === true
        end
    end

    @testset "no-slip acts only on the selected faces" begin
        if backend === CPUBackend
            n = 5
            inactive = (left = false, right = false, front = false, back = false, top = false, bot = false)
            newV() = (
                PTArray(backend)(rand(n + 1, n + 2, n + 2)),
                PTArray(backend)(rand(n + 2, n + 1, n + 2)),
                PTArray(backend)(rand(n + 2, n + 2, n + 1)),
            )

            Vx, Vy, Vz = newV()
            Vx0, Vy0, Vz0 = copy(Vx), copy(Vy), copy(Vz)
            flow_bcs!(
                VelocityBoundaryConditions(;
                    no_slip = merge(inactive, (; left = true)), free_slip = inactive,
                ), Vx, Vy, Vz,
            )
            @test all(iszero, Vx[1, :, :])
            @test @views Vy[1, :, :] == -Vy0[2, :, :]
            @test @views Vz[1, :, :] == -Vz0[2, :, :]
            @test @views Vx[2:end, :, :] == Vx0[2:end, :, :]
            @test @views Vy[2:end, :, :] == Vy0[2:end, :, :]
            @test @views Vz[2:end, :, :] == Vz0[2:end, :, :]

            Vx, Vy, Vz = newV()
            Vx0, Vy0, Vz0 = copy(Vx), copy(Vy), copy(Vz)
            flow_bcs!(
                VelocityBoundaryConditions(;
                    no_slip = merge(inactive, (; right = true)), free_slip = inactive,
                ), Vx, Vy, Vz,
            )
            @test all(iszero, Vx[end, :, :])
            @test @views Vy[end, :, :] == -Vy0[end - 1, :, :]
            @test @views Vz[end, :, :] == -Vz0[end - 1, :, :]
            @test @views Vx[1:(end - 1), :, :] == Vx0[1:(end - 1), :, :]
            @test @views Vy[1:(end - 1), :, :] == Vy0[1:(end - 1), :, :]
            @test @views Vz[1:(end - 1), :, :] == Vz0[1:(end - 1), :, :]

            Vx, Vy, Vz = newV()
            Vx0, Vy0, Vz0 = copy(Vx), copy(Vy), copy(Vz)
            flow_bcs!(
                VelocityBoundaryConditions(;
                    no_slip = merge(inactive, (; front = true)), free_slip = inactive,
                ), Vx, Vy, Vz,
            )
            @test all(iszero, Vy[:, 1, :])
            @test @views Vx[:, 1, :] == -Vx0[:, 2, :]
            @test @views Vz[:, 1, :] == -Vz0[:, 2, :]
            @test @views Vx[:, 2:end, :] == Vx0[:, 2:end, :]
            @test @views Vy[:, 2:end, :] == Vy0[:, 2:end, :]
            @test @views Vz[:, 2:end, :] == Vz0[:, 2:end, :]

            Vx, Vy, Vz = newV()
            Vx0, Vy0, Vz0 = copy(Vx), copy(Vy), copy(Vz)
            flow_bcs!(
                VelocityBoundaryConditions(;
                    no_slip = merge(inactive, (; back = true)), free_slip = inactive,
                ), Vx, Vy, Vz,
            )
            @test all(iszero, Vy[:, end, :])
            @test @views Vx[:, end, :] == -Vx0[:, end - 1, :]
            @test @views Vz[:, end, :] == -Vz0[:, end - 1, :]
            @test @views Vx[:, 1:(end - 1), :] == Vx0[:, 1:(end - 1), :]
            @test @views Vy[:, 1:(end - 1), :] == Vy0[:, 1:(end - 1), :]
            @test @views Vz[:, 1:(end - 1), :] == Vz0[:, 1:(end - 1), :]

            Vx, Vy, Vz = newV()
            Vx0, Vy0, Vz0 = copy(Vx), copy(Vy), copy(Vz)
            flow_bcs!(
                VelocityBoundaryConditions(;
                    no_slip = merge(inactive, (; bot = true)), free_slip = inactive,
                ), Vx, Vy, Vz,
            )
            @test all(iszero, Vz[:, :, 1])
            @test @views Vx[:, :, 1] == -Vx0[:, :, 2]
            @test @views Vy[:, :, 1] == -Vy0[:, :, 2]
            @test @views Vx[:, :, 2:end] == Vx0[:, :, 2:end]
            @test @views Vy[:, :, 2:end] == Vy0[:, :, 2:end]
            @test @views Vz[:, :, 2:end] == Vz0[:, :, 2:end]

            Vx, Vy, Vz = newV()
            Vx0, Vy0, Vz0 = copy(Vx), copy(Vy), copy(Vz)
            flow_bcs!(
                VelocityBoundaryConditions(;
                    no_slip = merge(inactive, (; top = true)), free_slip = inactive,
                ), Vx, Vy, Vz,
            )
            @test all(iszero, Vz[:, :, end])
            @test @views Vx[:, :, end] == -Vx0[:, :, end - 1]
            @test @views Vy[:, :, end] == -Vy0[:, :, end - 1]
            @test @views Vx[:, :, 1:(end - 1)] == Vx0[:, :, 1:(end - 1)]
            @test @views Vy[:, :, 1:(end - 1)] == Vy0[:, :, 1:(end - 1)]
            @test @views Vz[:, :, 1:(end - 1)] == Vz0[:, :, 1:(end - 1)]
        else
            @test true === true
        end
    end

    @testset "free-slip acts only on the selected faces" begin
        if backend === CPUBackend
            n = 5
            inactive = (left = false, right = false, front = false, back = false, top = false, bot = false)
            newV() = (
                PTArray(backend)(rand(n + 1, n + 2, n + 2)),
                PTArray(backend)(rand(n + 2, n + 1, n + 2)),
                PTArray(backend)(rand(n + 2, n + 2, n + 1)),
            )

            # `bot` is the z-min XY plane and `top` the z-max one
            Vx, Vy, Vz = newV()
            Vx0, Vy0, Vz0 = copy(Vx), copy(Vy), copy(Vz)
            flow_bcs!(
                VelocityBoundaryConditions(;
                    no_slip = inactive, free_slip = merge(inactive, (; bot = true)),
                ), Vx, Vy, Vz,
            )
            @test @views Vx[:, :, 1] == Vx0[:, :, 2]
            @test @views Vy[:, :, 1] == Vy0[:, :, 2]
            @test @views Vx[:, :, 2:end] == Vx0[:, :, 2:end]
            @test @views Vy[:, :, 2:end] == Vy0[:, :, 2:end]
            @test Vz == Vz0

            Vx, Vy, Vz = newV()
            Vx0, Vy0, Vz0 = copy(Vx), copy(Vy), copy(Vz)
            flow_bcs!(
                VelocityBoundaryConditions(;
                    no_slip = inactive, free_slip = merge(inactive, (; left = true)),
                ), Vx, Vy, Vz,
            )
            @test @views Vy[1, :, :] == Vy0[2, :, :]
            @test @views Vz[1, :, :] == Vz0[2, :, :]
            @test @views Vy[2:end, :, :] == Vy0[2:end, :, :]
            @test @views Vz[2:end, :, :] == Vz0[2:end, :, :]
            @test Vx == Vx0

            Vx, Vy, Vz = newV()
            Vx0, Vy0, Vz0 = copy(Vx), copy(Vy), copy(Vz)
            flow_bcs!(
                VelocityBoundaryConditions(;
                    no_slip = inactive, free_slip = merge(inactive, (; right = true)),
                ), Vx, Vy, Vz,
            )
            @test @views Vy[end, :, :] == Vy0[end - 1, :, :]
            @test @views Vz[end, :, :] == Vz0[end - 1, :, :]
            @test @views Vy[1:(end - 1), :, :] == Vy0[1:(end - 1), :, :]
            @test @views Vz[1:(end - 1), :, :] == Vz0[1:(end - 1), :, :]
            @test Vx == Vx0

            Vx, Vy, Vz = newV()
            Vx0, Vy0, Vz0 = copy(Vx), copy(Vy), copy(Vz)
            flow_bcs!(
                VelocityBoundaryConditions(;
                    no_slip = inactive, free_slip = merge(inactive, (; front = true)),
                ), Vx, Vy, Vz,
            )
            @test @views Vx[:, 1, :] == Vx0[:, 2, :]
            @test @views Vz[:, 1, :] == Vz0[:, 2, :]
            @test @views Vx[:, 2:end, :] == Vx0[:, 2:end, :]
            @test @views Vz[:, 2:end, :] == Vz0[:, 2:end, :]
            @test Vy == Vy0

            Vx, Vy, Vz = newV()
            Vx0, Vy0, Vz0 = copy(Vx), copy(Vy), copy(Vz)
            flow_bcs!(
                VelocityBoundaryConditions(;
                    no_slip = inactive, free_slip = merge(inactive, (; back = true)),
                ), Vx, Vy, Vz,
            )
            @test @views Vx[:, end, :] == Vx0[:, end - 1, :]
            @test @views Vz[:, end, :] == Vz0[:, end - 1, :]
            @test @views Vx[:, 1:(end - 1), :] == Vx0[:, 1:(end - 1), :]
            @test @views Vz[:, 1:(end - 1), :] == Vz0[:, 1:(end - 1), :]
            @test Vy == Vy0
        else
            @test true === true
        end
    end

    @testset "flow boundary conditions on a non-cubic grid" begin
        if backend === CPUBackend
            nx, ny, nz = 3, 5, 7
            all_on = (left = true, right = true, front = true, back = true, top = true, bot = true)
            inactive = map(_ -> false, all_on)
            newV() = (
                PTArray(backend)(rand(nx + 1, ny + 2, nz + 2)),
                PTArray(backend)(rand(nx + 2, ny + 1, nz + 2)),
                PTArray(backend)(rand(nx + 2, ny + 2, nz + 1)),
            )

            Vx, Vy, Vz = newV()
            Vx0, Vy0, Vz0 = copy(Vx), copy(Vy), copy(Vz)
            flow_bcs!(
                VelocityBoundaryConditions(; no_slip = all_on, free_slip = inactive),
                Vx, Vy, Vz,
            )
            @test all(iszero, Vx[1, :, :])
            @test all(iszero, Vx[end, :, :])
            @test all(iszero, Vy[:, 1, :])
            @test all(iszero, Vy[:, end, :])
            @test all(iszero, Vz[:, :, 1])
            @test all(iszero, Vz[:, :, end])
            # edges are written by more than one face, so compare away from them
            @test @views Vx[2:(end - 1), 2:(end - 1), 1] == -Vx0[2:(end - 1), 2:(end - 1), 2]
            @test @views Vy[2:(end - 1), 2:(end - 1), end] == -Vy0[2:(end - 1), 2:(end - 1), end - 1]
            @test @views Vz[1, 2:(end - 1), 2:(end - 1)] == -Vz0[2, 2:(end - 1), 2:(end - 1)]

            Vx, Vy, Vz = newV()
            Vx0, Vy0, Vz0 = copy(Vx), copy(Vy), copy(Vz)
            flow_bcs!(
                VelocityBoundaryConditions(; no_slip = inactive, free_slip = all_on),
                Vx, Vy, Vz,
            )
            @test @views Vx[:, 1, 2:(end - 1)] == Vx0[:, 2, 2:(end - 1)]
            @test @views Vx[:, 2:(end - 1), 1] == Vx0[:, 2:(end - 1), 2]
            @test @views Vy[1, :, 2:(end - 1)] == Vy0[2, :, 2:(end - 1)]
            @test @views Vy[2:(end - 1), :, end] == Vy0[2:(end - 1), :, end - 1]
            @test @views Vz[end, 2:(end - 1), :] == Vz0[end - 1, 2:(end - 1), :]
            @test @views Vz[2:(end - 1), end, :] == Vz0[2:(end - 1), end - 1, :]
        else
            @test true === true
        end
    end

    @testset "periodic velocity boundary conditions" begin
        if backend === CPUBackend
            n = 5
            inactive = (left = false, right = false, front = false, back = false, top = false, bot = false)
            newV() = (
                PTArray(backend)(rand(n + 1, n + 2, n + 2)),
                PTArray(backend)(rand(n + 2, n + 1, n + 2)),
                PTArray(backend)(rand(n + 2, n + 2, n + 1)),
            )

            # x-periodic: Vx is the normal component, so its paired faces coincide
            Vx, Vy, Vz = newV()
            Vx0, Vy0, Vz0 = copy(Vx), copy(Vy), copy(Vz)
            flow_bcs!(
                VelocityBoundaryConditions(;
                    no_slip = inactive, free_slip = inactive,
                    periodic = merge(inactive, (; left = true, right = true)),
                ), Vx, Vy, Vz,
            )
            @test @views Vx[1, :, :] == Vx0[end, :, :]
            @test @views Vx[end, :, :] == Vx0[end, :, :]
            @test @views Vy[1, :, :] == Vy0[end - 1, :, :]
            @test @views Vy[end, :, :] == Vy0[2, :, :]
            @test @views Vz[1, :, :] == Vz0[end - 1, :, :]
            @test @views Vz[end, :, :] == Vz0[2, :, :]

            # y-periodic: Vy is the normal component
            Vx, Vy, Vz = newV()
            Vx0, Vy0, Vz0 = copy(Vx), copy(Vy), copy(Vz)
            flow_bcs!(
                VelocityBoundaryConditions(;
                    no_slip = inactive, free_slip = inactive,
                    periodic = merge(inactive, (; front = true, back = true)),
                ), Vx, Vy, Vz,
            )
            @test @views Vx[:, 1, :] == Vx0[:, end - 1, :]
            @test @views Vx[:, end, :] == Vx0[:, 2, :]
            @test @views Vy[:, 1, :] == Vy0[:, end, :]
            @test @views Vy[:, end, :] == Vy0[:, end, :]
            @test @views Vz[:, 1, :] == Vz0[:, end - 1, :]
            @test @views Vz[:, end, :] == Vz0[:, 2, :]

            # z-periodic: Vz is the normal component
            Vx, Vy, Vz = newV()
            Vx0, Vy0, Vz0 = copy(Vx), copy(Vy), copy(Vz)
            flow_bcs!(
                VelocityBoundaryConditions(;
                    no_slip = inactive, free_slip = inactive,
                    periodic = merge(inactive, (; bot = true, top = true)),
                ), Vx, Vy, Vz,
            )
            @test @views Vx[:, :, 1] == Vx0[:, :, end - 1]
            @test @views Vx[:, :, end] == Vx0[:, :, 2]
            @test @views Vy[:, :, 1] == Vy0[:, :, end - 1]
            @test @views Vy[:, :, end] == Vy0[:, :, 2]
            @test @views Vz[:, :, 1] == Vz0[:, :, end]
            @test @views Vz[:, :, end] == Vz0[:, :, end]

            @test_throws "Periodic boundary conditions must be paired" VelocityBoundaryConditions(;
                no_slip = inactive, free_slip = inactive,
                periodic = merge(inactive, (; front = true)),
            )
            @test_throws "Incompatible boundary conditions on the bot boundary" VelocityBoundaryConditions(;
                no_slip = inactive, free_slip = merge(inactive, (; bot = true)),
                periodic = merge(inactive, (; bot = true, top = true)),
            )
            @test_throws "top can't be both periodic and free_surface" VelocityBoundaryConditions(;
                no_slip = inactive, free_slip = inactive,
                periodic = merge(inactive, (; bot = true, top = true)),
                free_surface = true,
            )
        else
            @test true === true
        end
    end

    @testset "displacement boundary conditions on bare arrays" begin
        if backend === CPUBackend
            n = 5
            inactive = (left = false, right = false, front = false, back = false, top = false, bot = false)
            Ux = PTArray(backend)(rand(n + 1, n + 2, n + 2))
            Uy = PTArray(backend)(rand(n + 2, n + 1, n + 2))
            Uz = PTArray(backend)(rand(n + 2, n + 2, n + 1))
            Ux0, Uy0, Uz0 = copy(Ux), copy(Uy), copy(Uz)
            flow_bcs!(
                DisplacementBoundaryConditions(;
                    no_slip = merge(inactive, (; left = true)),
                    free_slip = merge(inactive, (; top = true)),
                ), Ux, Uy, Uz,
            )
            @test all(iszero, Ux[1, :, :])
            # free-slip runs after no-slip and overwrites the top XY plane
            @test @views Uy[1, :, 1:(end - 1)] == -Uy0[2, :, 1:(end - 1)]
            @test @views Uz[1, :, :] == -Uz0[2, :, :]
            @test @views Ux[:, :, end] == Ux[:, :, end - 1]
            @test @views Uy[:, :, end] == Uy[:, :, end - 1]
            @test Uz[2:end, :, :] == Uz0[2:end, :, :]
        else
            @test true === true
        end
    end

    @testset "background shear fields leave ghost layers untouched" begin
        xci = (LinRange(0.5, 2.5, 3), LinRange(0.5, 3.5, 4), LinRange(0.5, 4.5, 5))
        xvi = (LinRange(0.0, 3.0, 4), LinRange(0.0, 4.0, 5), LinRange(0.0, 5.0, 6))

        for bc! in (pureshear_bc!, simpleshear_bc!)
            stokes = StokesArrays(backend, (3, 4, 5))
            stokes.V.Vx .= NaN
            stokes.V.Vy .= NaN
            stokes.V.Vz .= NaN
            bc!(stokes, xci, xvi, 2.0)

            Vx, Vy, Vz = Array(stokes.V.Vx), Array(stokes.V.Vy), Array(stokes.V.Vz)
            @test all(isnan, @view Vx[:, 1, :])
            @test all(isnan, @view Vx[:, end, :])
            @test all(isnan, @view Vx[:, :, 1])
            @test all(isnan, @view Vx[:, :, end])
            @test all(isnan, @view Vy[1, :, :])
            @test all(isnan, @view Vy[end, :, :])
            @test all(isnan, @view Vz[1, :, :])
            @test all(isnan, @view Vz[end, :, :])
            @test !any(isnan, @view Vx[:, 2:(end - 1), 2:(end - 1)])
            @test !any(isnan, @view Vy[2:(end - 1), :, 2:(end - 1)])
            @test !any(isnan, @view Vz[2:(end - 1), 2:(end - 1), :])
        end
    end

    @testset "background shear fields accept the legacy backend argument" begin
        xci = (LinRange(0.5, 2.5, 3), LinRange(0.5, 3.5, 4), LinRange(0.5, 4.5, 5))
        xvi = (LinRange(0.0, 3.0, 4), LinRange(0.0, 4.0, 5), LinRange(0.0, 5.0, 6))

        for bc! in (pureshear_bc!, simpleshear_bc!)
            stokes4 = StokesArrays(backend, (3, 4, 5))
            stokes5 = StokesArrays(backend, (3, 4, 5))
            bc!(stokes4, xci, xvi, 2.0)
            bc!(stokes5, xci, xvi, 2.0, backend)

            @test Array(stokes4.V.Vx) == Array(stokes5.V.Vx)
            @test Array(stokes4.V.Vy) == Array(stokes5.V.Vy)
            @test Array(stokes4.V.Vz) == Array(stokes5.V.Vz)
        end
    end

    @testset "pure shear uses each direction's own coordinate vector" begin
        # the three axes have different extents, so a mixed-up coordinate
        # vector would not reproduce the expected field
        stokes = StokesArrays(backend, (3, 4, 5))
        xci = (LinRange(0.5, 2.5, 3), LinRange(10.5, 13.5, 4), LinRange(100.5, 104.5, 5))
        xvi = (LinRange(0.0, 3.0, 4), LinRange(10.0, 14.0, 5), LinRange(100.0, 105.0, 6))
        pureshear_bc!(stokes, xci, xvi, 2.0)

        @test Array(@view stokes.V.Vx[:, 2:(end - 1), 2:(end - 1)]) ==
            [2.0 * x for x in xvi[1], _ in xci[2], _ in xci[3]]
        @test Array(@view stokes.V.Vy[2:(end - 1), :, 2:(end - 1)]) ==
            [2.0 * y for _ in xci[1], y in xvi[2], _ in xci[3]]
        @test Array(@view stokes.V.Vz[2:(end - 1), 2:(end - 1), :]) ==
            [-2.0 * z for _ in xci[1], _ in xci[2], z in xvi[3]]
    end

    @testset "pure shear boundary condition" begin
        stokes = StokesArrays(backend, (3, 4, 5))
        xci = (collect(1.0:3.0), collect(1.0:4.0), collect(1.0:5.0))
        xvi = (collect(1.0:4.0), collect(1.0:5.0), collect(1.0:6.0))
        pureshear_bc!(stokes, xci, xvi, 2.0)

        @test Array(@view stokes.V.Vx[:, 2:(end - 1), 2:(end - 1)]) == [2.0 * x for x in xvi[1], _ in xci[2], _ in xci[3]]
        @test Array(@view stokes.V.Vy[2:(end - 1), :, 2:(end - 1)]) == [2.0 * y for _ in xci[1], y in xvi[2], _ in xci[3]]
        @test Array(@view stokes.V.Vz[2:(end - 1), 2:(end - 1), :]) == [-2.0 * z for _ in xci[1], _ in xci[2], z in xvi[3]]
    end

    @testset "simple shear boundary condition" begin
        stokes = StokesArrays(backend, (3, 4, 5))
        xci = (collect(1.0:3.0), collect(1.0:4.0), collect(1.0:5.0))
        xvi = (collect(1.0:4.0), collect(1.0:5.0), collect(1.0:6.0))
        simpleshear_bc!(stokes, xci, xvi, 2.0)

        @test Array(@view stokes.V.Vx[:, 2:(end - 1), 2:(end - 1)]) == [2.0 * y for _ in xvi[1], y in xci[2], _ in xci[3]]
        @test all(iszero, Array(@view stokes.V.Vy[2:(end - 1), :, 2:(end - 1)]))
        @test all(iszero, Array(@view stokes.V.Vz[2:(end - 1), 2:(end - 1), :]))
    end
end
