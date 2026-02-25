using Test
using JustRelax, JustRelax.JustRelax2D

@testset "MiniKernels" begin
    A2 = reshape(collect(1.0:16.0), 4, 4)
    A3 = reshape(collect(1.0:64.0), 4, 4, 4)
    i = 2
    j = 2
    k = 2

    @test JustRelax2D.center(A2, i, j) == 6.0
    @test JustRelax2D.next(A2, i, j) == 11.0
    @test JustRelax2D.left(A2, i, j) == 5.0
    @test JustRelax2D.right(A2, i, j) == 7.0
    @test JustRelax2D.back(A2, i, j) == 2.0
    @test JustRelax2D.front(A2, i, j) == 10.0

    @test JustRelax2D.left(A3, i, j, k) == 21.0
    @test JustRelax2D.right(A3, i, j, k) == 23.0
    @test JustRelax2D.back(A3, i, j, k) == 18.0
    @test JustRelax2D.front(A3, i, j, k) == 26.0
    @test JustRelax2D.bot(A3, i, j, k) == 6.0
    @test JustRelax2D.top(A3, i, j, k) == 38.0

    dx = 2.0
    dy = 3.0
    dz = 4.0

    @test JustRelax2D._d_xa(A2, dx, i, j) == 2.0
    @test JustRelax2D._d_ya(A2, dy, i, j) == 12.0
    @test JustRelax2D._d_za(A3, dz, i, j, k) == 64.0

    @test JustRelax2D._d_xi(A2, dx, i, j) == 2.0
    @test JustRelax2D._d_yi(A2, dy, i, j) == 12.0

    @test JustRelax2D._d_xi(A3, dx, i, j, k) == 2.0
    @test JustRelax2D._d_yi(A3, dy, i, j, k) == 12.0
    @test JustRelax2D._d_zi(A3, dz, i, j, k) == 64.0

    Ax = A2
    Ay = A2 .* 2.0
    @test JustRelax2D.div(Ax, Ay, dx, dy, i, j) == 26.0

    Ax3 = A3
    Ay3 = A3 .* 2.0
    Az3 = A3 .* 3.0
    @test JustRelax2D.div(Ax3, Ay3, Az3, dx, dy, dz, i, j, k) == 218.0

    @test JustRelax2D._av(A2, i, j) == 13.5
    @test JustRelax2D._av_a(A2, i, j) == 8.5
    @test JustRelax2D._av_xa(A2, i, j) == 6.5
    @test JustRelax2D._av_ya(A2, i, j) == 8.0
    @test JustRelax2D._av_xi(A2, i, j) == 10.5
    @test JustRelax2D._av_yi(A2, i, j) == 9.0

    @test JustRelax2D._harm(A2, i, j) == 1.2136363636363636
    @test JustRelax2D._harm_a(A2, i, j) == 2.001731601731602
    @test JustRelax2D._harm_xa(A2, i, j) == 0.6190476190476191
    @test JustRelax2D._harm_ya(A2, i, j) == 0.5333333333333333

    @test JustRelax2D._gather(A2, i, j) == (6.0, 7.0, 10.0, 11.0)

    @test JustRelax2D._av(A3, i, j, k) == 32.5
    @test JustRelax2D._av_x(A3, i, j, k) == 22.5
    @test JustRelax2D._av_y(A3, i, j, k) == 24.0
    @test JustRelax2D._av_z(A3, i, j, k) == 30.0
    @test JustRelax2D._av_xy(A3, i, j, k) == 24.5
    @test JustRelax2D._av_xz(A3, i, j, k) == 30.5
    @test JustRelax2D._av_yz(A3, i, j, k) == 32.0
    @test JustRelax2D._av_xyi(A3, i, j, k) == 19.5
    @test JustRelax2D._av_xzi(A3, i, j, k) == 13.5
    @test JustRelax2D._av_yzi(A3, i, j, k) == 12.0

    @test JustRelax2D._harm_x(A3, i, j, k) == 22.488888888888887
    @test JustRelax2D._harm_y(A3, i, j, k) == 23.833333333333332
    @test JustRelax2D._harm_z(A3, i, j, k) == 27.866666666666667
    @test JustRelax2D._harm_xy(A3, i, j, k) == 0.04081632653061224
    @test JustRelax2D._harm_xz(A3, i, j, k) == 0.03278688524590164
    @test JustRelax2D._harm_yz(A3, i, j, k) == 0.03125
    @test JustRelax2D._harm_xyi(A3, i, j, k) == 0.05128205128205128
    @test JustRelax2D._harm_xzi(A3, i, j, k) == 0.07407407407407407
    @test JustRelax2D._harm_yzi(A3, i, j, k) == 0.08333333333333333

    @test JustRelax2D._gather_yz(A3, i, j, k) == (22.0, 26.0, 38.0, 42.0)
    @test JustRelax2D._gather_xz(A3, i, j, k) == (22.0, 23.0, 38.0, 39.0)
    @test JustRelax2D._gather_xy(A3, i, j, k) == (22.0, 23.0, 26.0, 27.0)

    @test JustRelax2D._current(A3, i, j, k) == 22.0

    v = collect(1.0:5.0)
    @test JustRelax2D.mysum(v, 2:4) == 9.0
    @test JustRelax2D.mysum(inv, v, 2:4) == 1.0833333333333333
    @test JustRelax2D.mysum(A2, 2:3, 2:3) == 34.0
    @test JustRelax2D.mysum(inv, A2, 2:3, 2:3) == 0.5004329004329005
    @test JustRelax2D.mysum(A3, 2:3, 2:3, 2:3) == 260.0
    @test JustRelax2D.mysum(inv, A3, 2:3, 2:3, 2:3) == 0.2634535347004082
end
