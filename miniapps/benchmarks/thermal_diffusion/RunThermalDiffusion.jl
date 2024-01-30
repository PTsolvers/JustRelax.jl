# using CairoMakie
using JustRelax, GeoParams
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)

# setup ParallelStencil.jl environment
dimension = 3 # 2 | 3
device = :cpu # :cpu | :CUDA | :AMDGPU
precision = Float64
model = PS_Setup(device, precision, dimension)
environment!(model)

# Model size
L = 100e3 # [m]
if dimension === 3
    # include model setup
    include("diffusion/diffusion3D.jl")

    # model resolution (number of gridpoints)
    nx, ny, nz = 32, 32, 32

    # start model
    geometry, thermal = diffusion_3D(;
        nx=nx,
        ny=ny,
        nz=nz,
        lx=L,
        ly=L,
        lz=L,
        ρ0=3.3e3,
        Cp0=1.2e3,
        K0=3.0,
        init_MPI=JustRelax.MPI.Initialized() ? false : true,
        finalize_MPI=true,
    )

elseif dimension == 2
    # include model setup
    include("diffusion/diffusion2D.jl")

    # model resolution (number of gridpoints)
    nx, ny = 64, 64

    # start model
    geometry, thermal = diffusion_2D(;
        nx=nx, ny=ny, lx=L, ly=L, ρ0=3.3e3, Cp0=1.2e3, K0=3.0
    )

end
