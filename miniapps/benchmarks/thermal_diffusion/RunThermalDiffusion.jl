# using CairoMakie
using JustRelax

# setup ParallelStencil.jl environment
dimension = 3 # 1 | 2 | 3 
device = :cpu # :cpu | :gpu
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
    geometry, thermal, iters = diffusion_3D(;
        nx=nx,
        ny=ny,
        nz=nz,
        lx=L,
        ly=L,
        lz=L,
        ρ=3.3e3,
        Cp=1.2e3,
        K=3.0,
        init_MPI=MPI.Initialized() ? false : true,
        finalize_MPI=false,
    )

elseif dimension == 2
    # include model setup
    include("diffusion/diffusion2D.jl")

    # model resolution (number of gridpoints)
    nx, ny = 64, 64

    # start model
    geometry, thermal, iters = diffusion_2D(;
        nx=nx, ny=ny, lx=L, ly=L, ρ=3.3e3, Cp=1.2e3, K=3.0
    )

elseif dimension == 1
    # include model setup
    include("diffusion/diffusion1D.jl")

    # model resolution (number of gridpoints)
    nx = 256

    # start model
    geometry, thermal, iters = diffusion_1D(; nx=nx, lx=L, ρ=3.3e3, Cp=1.2e3, K=3.0)
end