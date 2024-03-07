using Test
using JustRelax, JustRelax.DataIO
using WriteVTK, LightXML, Printf

@testset "Paraview collection" begin
    x, y, z = 0:10, 1:6, 2:0.1:3
    times = range(0, 1; step=1)

    #generate `*.vti` files
    for (n, time) in enumerate(times)
        vtk_grid("./test_vti_$n", x, y, z) do vtk
            vtk["Pressure"] = rand(length(x), length(y), length(z))
        end
    end

    nx_c, ny_c, nz_c = length(x), length(y), length(z)
    nx_v, ny_v, nz_v = nx_c + 1, ny_c + 1, nz_c + 1
    xci = 0.5 * (x[1:(end - 1)] + x[2:end]),
    0.5 * (y[1:(end - 1)] + y[2:end]),
    0.5 * (z[1:(end - 1)] + z[2:end])
    xvi = 0.5 * (x[1:(end - 1)] + x[2:end]),
    0.5 * (y[1:(end - 1)] + y[2:end]),
    0.5 * (z[1:(end - 1)] + z[2:end])
    data_c = (;
        Temperature=Array(rand(nx_c, ny_c, nz_c)), Testing=Array(rand(nx_c, ny_c, nz_c))
    )
    data_v = (;
        Pressure=Array(rand(nx_v, ny_v, nz_v)), Testing=Array(rand(nx_v, ny_v, nz_v))
    )
    save_vtk(joinpath("./", "vtk_" * lpad("1", 6, "0")), xvi, xci, data_v, data_c)

    make_paraview_collection(; dir="./", pvd_name="test", file_extension=".vti")
    @test isfile("test.pvd")
    @test filesize("test.pvd") == 413

    make_paraview_collection(; dir="./", file_extension=".vti")
    @test isfile("full_simulation.pvd")
    @test filesize("full_simulation.pvd") == 413

    make_paraview_collection(;)
    @test isfile("full_simulation.pvd")
    @test filesize("full_simulation.pvd") == 285

    files = ["test_files/test_vti_1.vti", "test_files/test_vti_2.vti"]
    time = ["1.0", "2.0"]
    make_paraview_collection("test2", files, time)
    @test isfile("test2.pvd")
    @test filesize("test2.pvd") == 317

    make_paraview_collection(; pvd_name="test3", files=files, time=time)
    @test isfile("test3.pvd")
    @test filesize("test3.pvd") == 317

    rm("test.pvd")
    rm("full_simulation.pvd")
    rm("vtk_000001.vtm")
    rm("vtk_000001_1.vti")
    rm("vtk_000001_2.vti")
    rm("test_vti_1.vti")
    rm("test_vti_2.vti")
    rm("test2.pvd")
    rm("test3.pvd")
end
