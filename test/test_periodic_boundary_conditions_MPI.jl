pushfirst!(LOAD_PATH, dirname(@__DIR__))

using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
using Test

@init_parallel_stencil(Threads, Float64, 2)

nx, ny = 8, 6
igg = IGG(
    init_global_grid(
        nx, ny, 1;
        dimx = 2,
        dimy = 1,
        periodx = true,
        init_MPI = true,
        select_device = false,
    )...,
)

thermal = ThermalArrays(CPUBackend, (nx, ny))
thermal.T .= igg.coords[1] + 1
thermal_bc = TemperatureBoundaryConditions(;
    no_flux = (left = false, right = false, top = true, bot = true),
    periodic = (left = true, right = true, top = false, bot = false),
)

thermal_bcs!(thermal, thermal_bc)
update_halo!(thermal.T)

other_rank_value = 2 - igg.coords[1]
@test all(@view(thermal.T[1, 2:(end - 1)]) .== other_rank_value)
@test all(@view(thermal.T[end, 2:(end - 1)]) .== other_rank_value)

stokes = StokesArrays(CPUBackend, (nx, ny))
stokes.V.Vx .= igg.coords[1] + 1
stokes.V.Vy .= igg.coords[1] + 1
flow_bc = VelocityBoundaryConditions(;
    no_slip = (left = false, right = false, top = false, bot = false),
    free_slip = (left = false, right = false, top = true, bot = true),
    periodic = (left = true, right = true, top = false, bot = false),
)

flow_bcs!(stokes, flow_bc)
update_halo!(@velocity(stokes)...)

@test all(@view(stokes.V.Vy[1, :]) .== other_rank_value)
@test all(@view(stokes.V.Vy[end, :]) .== other_rank_value)

finalize_global_grid(; finalize_MPI = true)
