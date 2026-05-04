import Enzyme
using Enzyme: Const, Duplicated, DuplicatedNoNeed, Reverse
using ParallelStencil, Test, Random

using JustRelax
using JustRelax.JustRelax2D

const AD     = ParallelStencil.AD
const DupNN  = DuplicatedNoNeed
const Dup    = Duplicated
const JR     = JustRelax

@init_parallel_stencil(Threads, Float64, 2)

function expected_gradients(ni, grid, ∇V̄, εxx̄, εyȳ, εxȳ)
    Vx̄    = zeros(Float64, ni[1] + 1, ni[2] + 2)
    Vȳ    = zeros(Float64, ni[1] + 2, ni[2] + 1)
    third = 1.0 / 3.0

    for i in 1:(ni[1] + 1), j in 1:(ni[2] + 1)
        dy_vx   = JR.get_dy(grid._di.velocity[1], j)
        dx_vy   = JR.get_dx(grid._di.velocity[2], i)
        εxy_bar = εxȳ[i, j]

        Vx̄[i, j]     -= 0.5 * dy_vx * εxy_bar
        Vx̄[i, j + 1] += 0.5 * dy_vx * εxy_bar
        Vȳ[i, j]     -= 0.5 * dx_vy * εxy_bar
        Vȳ[i + 1, j] += 0.5 * dx_vy * εxy_bar

        if i <= ni[1] && j <= ni[2]
            dx, dy     = JR.get_dxi(grid._di.vertex, i, j)
            dVx_dx_bar = (2.0 * third) * εxx̄[i, j] - third * εyȳ[i, j]
            dVy_dy_bar = (2.0 * third) * εyȳ[i, j] - third * εxx̄[i, j]

            Vx̄[i, j + 1]     -= dx * dVx_dx_bar
            Vx̄[i + 1, j + 1] += dx * dVx_dx_bar
            Vȳ[i + 1, j]     -= dy * dVy_dy_bar
            Vȳ[i + 1, j + 1] += dy * dVy_dy_bar
        end
    end

    return Vx̄, Vȳ
end

@testset "ParallelStencil Enzyme Test" begin
    ni   = (3, 2)
    li   = (3.0, 2.0)
    grid = Geometry(ni, li)

    Vx  = randn(Float64, ni[1] + 1, ni[2] + 2)
    Vy  = randn(Float64, ni[1] + 2, ni[2] + 1)
    ∇V  = zeros(Float64, ni...)
    εxx = zeros(Float64, ni...)
    εyy = zeros(Float64, ni...)
    εxy = zeros(Float64, ni[1] + 1, ni[2] + 1)

    Vx̄  = zeros(Float64, size(Vx)...)
    Vȳ  = zeros(Float64, size(Vy)...)
    ∇V̄  = zeros(Float64, ni...)
    εxx̄ = ones(Float64, ni...)
    εyȳ = ones(Float64, ni...)
    εxȳ = ones(Float64, ni[1] + 1, ni[2] + 1)

    ranges = (1:(ni[1] + 1), 1:(ni[2] + 1))

    @parallel ranges configcall=JustRelax2D.compute_∇V_strain_rate!(
        ∇V,
        εxx,
        εyy,
        εxy,
        Vx,
        Vy,
        grid._di.vertex,
        grid._di.velocity[1],
        grid._di.velocity[2],
    ) AD.autodiff_deferred!(
        Reverse,
        JustRelax2D.compute_∇V_strain_rate!,
        DupNN(∇V, ∇V̄),
        DupNN(εxx, copy(εxx̄)),
        DupNN(εyy, copy(εyȳ)),
        DupNN(εxy, copy(εxȳ)),
        DupNN(Vx, Vx̄),
        DupNN(Vy, Vȳ),
        Const(grid._di.vertex),
        Const(grid._di.velocity[1]),
        Const(grid._di.velocity[2]),
    )

    Vx̄_ref, Vȳ_ref = expected_gradients(ni, grid, ∇V̄, εxx̄, εyȳ, εxȳ)

    @test Vx̄ ≈ Vx̄_ref rtol=5e-6 atol=5e-6
    @test Vȳ ≈ Vȳ_ref rtol=5e-6 atol=5e-6
end
