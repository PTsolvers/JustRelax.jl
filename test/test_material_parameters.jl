push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil, ParallelStencil.FiniteDifferences2D

const backend_JR = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    CPUBackend
end

function test_material()
    elasticity = ConstantElasticity(; G = 0.1, Kb = 0.5)
    return SetMaterialParams(;
        Phase = 1,
        Density = PT_Density(; ρ0 = 1.0, α = 0.1, β = 0.05, T0 = 0.0, P0 = 0.0),
        CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), elasticity)),
        Elasticity = elasticity,
        Gravity = ConstantGravity(; g = 1.0),
    )
end

@testset "Material parameter discovery" begin
    material = test_material()

    @test JustRelax2D.material_parameter_is_used(material, :ρ0)
    @test JustRelax2D.material_parameter_is_used(material, :η)
    @test JustRelax2D._density_parameter_paths(material, :α) == [(:Density, 1, :α)]
    @test JustRelax2D._linear_viscosity_parameter_paths(material, :η) ==
        [(:CompositeRheology, 1, :elements, 1, :η)]
    @test JustRelax2D.material_parameter_is_used(material, :G)
    # ConstantDensity has no thermal expansivity, so :α is absent for that phase.
    constant = SetMaterialParams(; Phase = 1, Density = ConstantDensity(; ρ = 1.0))
    @test !JustRelax2D.material_parameter_is_used(constant, :α)
end

@testset "material_controls allocation" begin
    ni = (6, 5)
    nphases = 2

    parameters = (:G, :ρ0, :α, :η, :C)
    gradients = material_controls(backend_JR, ni, parameters; nphases)
    @test keys(gradients) == (:G, :ρ0, :α, :η, :C)

    @test size(gradients.G.center) == (nphases, ni...)
    @test size(gradients.G.vertex) == (nphases, (ni .+ 1)...)
    @test size(gradients.η.center) == (nphases, ni...)
    @test size(gradients.η.vertex) == (nphases, (ni .+ 1)...)
    @test size(gradients.C.center) == (nphases, ni...)
    @test size(gradients.C.vertex) == (nphases, (ni .+ 1)...)
    @test size(gradients.ρ0.center) == (nphases, ni...)
    @test size(gradients.ρ0.vertex) == (nphases, (ni .+ 1)...)

    gradients = material_controls(
        backend_JR, ni, (:C,); nphases
    )
    @test size(gradients.C.center) == (nphases, ni...)
    @test size(gradients.C.vertex) == (nphases, (ni .+ 1)...)
    @test all(iszero, gradients.C.center)

    gradients = material_controls(backend_JR, ni, ())
    @test isempty(gradients)

    @test_throws ArgumentError material_controls(
        backend_JR, ni, (:G, :G);
        nphases,
    )
end

@testset "Periodic adjoint allocation" begin
    ni = (4, 3)
    periodic = (true, false)
    adjoint = AdjointStokesArrays(CPUBackend, ni, periodic)
    @test size(adjoint.R.Rx) == ni
    @test size(adjoint.R.Ry) == (ni[1], ni[2] - 1)
end

# `combine_center_vertex_gradient!` has to be the exact transpose of `center2vertex!`,
# otherwise the elastic gradient is wrong wherever the two grids disagree -- which is
# every boundary cell. The adjoint identity <center2vertex!(c), v̄> == <c, pullback(v̄)>
# pins that down without reference to the implementation.
@testset "center2vertex pullback" begin
    for (nx, ny) in ((2, 2), (3, 4), (5, 5), (8, 3), (16, 9))
        c = rand(nx, ny)
        v̄ = rand(nx + 1, ny + 1)

        cin = @zeros(nx, ny)
        copyto!(cin, c)
        v = @zeros(nx + 1, ny + 1)
        center2vertex!(v, cin)

        # the pullback works on the phase-resolved buffers, so use a single-phase slice
        c̄ = @zeros(1, nx, ny)
        seed = @zeros(1, nx + 1, ny + 1)
        copyto!(seed, reshape(v̄, 1, nx + 1, ny + 1))
        JustRelax2D.combine_center_vertex_gradient!(c̄, seed, 1)

        lhs = sum(Array(v) .* v̄)
        rhs = sum(c .* Array(c̄)[1, :, :])
        @test lhs ≈ rhs
    end

    # It accumulates into the center buffer rather than overwriting it: each phase adds
    # its vertex contribution on top of the center contribution already stored there.
    c̄ = @ones(1, 3, 3)
    JustRelax2D.combine_center_vertex_gradient!(c̄, @zeros(1, 4, 4), 1)
    @test all(isone, Array(c̄))

    @test_throws DimensionMismatch JustRelax2D.combine_center_vertex_gradient!(
        @zeros(1, 3, 3), @zeros(1, 3, 3), 1
    )

    for periodic in ((true, false), (false, true), (true, true))
        nx, ny = 4, 3
        c = rand(nx, ny)
        v̄ = rand(nx + 1, ny + 1)
        v = zeros(nx + 1, ny + 1)
        for iv in axes(v, 1), jv in axes(v, 2)
            iv_eff = periodic[1] ? iv : clamp(iv, 2, nx)
            jv_eff = periodic[2] ? jv : clamp(jv, 2, ny)
            i0 = periodic[1] ? mod1(iv_eff - 1, nx) : iv_eff - 1
            ic = periodic[1] ? mod1(iv_eff, nx) : iv_eff
            j0 = periodic[2] ? mod1(jv_eff - 1, ny) : jv_eff - 1
            jc = periodic[2] ? mod1(jv_eff, ny) : jv_eff
            v[iv, jv] = 0.25 * (c[i0, j0] + c[ic, jc] + c[i0, jc] + c[ic, j0])
        end

        c̄ = @zeros(1, nx, ny)
        seed = @zeros(1, nx + 1, ny + 1)
        copyto!(seed, reshape(v̄, 1, nx + 1, ny + 1))
        JustRelax2D.combine_center_vertex_gradient!(c̄, seed, 1, periodic)
        @test sum(v .* v̄) ≈ sum(c .* Array(c̄)[1, :, :])
    end
end
