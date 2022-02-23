# TODO: ADD REFERENCE TO TARAS' BOOK
function elastic_buildup(; nx=256-1, ny=256-1, lx=1e3, ly=1e3)
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny) # number of nodes in x- and y-
    li = (lx, ly)  # domain length in x- and y-
    di = @. li/(ni-1) # grid step in x- and -y
    max_li = max(li...)
    nDim = length(ni) # domain dimension
    xci = Tuple([di[i]/2:di[i]:(li[i]-di[i]/2) for i in 1:nDim]) # nodes at the center of the cells
    xvi = Tuple([0:di[i]:li[i] for i in 1:nDim]) # nodes at the vertices of the cells

    ## (Physical) Time domain and discretization
    yr = 365.25*3600*24
    kyr, Myr = 1e6*yr, 1e3*yr
    ttot = 1*Myr # total simulation time
    Δt = 100*kyr   # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni, ViscoElastic)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(ni, di)

    ## Setup-specific parameters and fields
    η0 = 1e22   # viscosity
    εbg = 1e-15 # background strain-rate
    η = fill(1e22, ni...) # viscosity field
    G = 10e9 # elastic modulus
    g = 0.0 # gravity

    Δt = η0/(G)

    ## Boundary conditions
    pureshear_bc!(stokes, di, li, εbg) 
    freeslip = ( freeslip_x = true, freeslip_y = true)

    # Physical time loop
    t = 0.0
    ρ = @zeros(size(stokes.P))
    while t < ttot
        solve2!(stokes, pt_stokes, di, li, max_li, freeslip, ρ.*g, η, G, Δt; iterMax = 10e3)
        copy_τ!(stokes)
        t += Δt
    end

    return  (ni=ni, xci=xci, xvi=xvi, li=li), stokes

end

## VISCO-ELASTIC STOKES SOLVER 

function solve2!(
    stokes::StokesArrays{ViscoElastic, A, B, C, D, 2}, 
    pt_stokes::PTStokesCoeffs, di::NTuple{2,T}, 
    li::NTuple{2,T}, 
    max_li, 
    freeslip,
    ρg, 
    η,
    G,
    dt;
    iterMax = 10e3, 
    nout = 500
) where {A, B, C, D, T}
    
    # unpack
    dx, dy = di 
    lx, ly = li 
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    dVx, dVy = stokes.dV.Vx, stokes.dV.Vy
    τ, τ_o = stress(stokes)
    τxx, τyy, τxy = τ
    τxx_o, τyy_o, τxy_o = τ_o
    P, ∇V = stokes.P, stokes.∇V
    Rx, Ry = stokes.R.Rx, stokes.R.Ry
    Gdτ, dτ_Rho, ϵ, Re, r, Vpdτ = pt_stokes.Gdτ, pt_stokes.dτ_Rho, pt_stokes.ϵ,  pt_stokes.Re,  pt_stokes.r,  pt_stokes.Vpdτ
    freeslip_x, freeslip_y = freeslip

    # ~preconditioner
    ητ = deepcopy(η)
    @parallel compute_maxloc!(ητ, η)
    # PT numerical coefficients
    @parallel elastic_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, G, dt, Re, r, max_li)

    # errors
    err=2*ϵ; iter=0; err_evo1=Float64[]; err_evo2=Float64[]; err_rms = Float64[]
    
    # solver loop
    while err > ϵ && iter <= iterMax
        if (iter==11)  global wtime0 = Base.time()  end
        @parallel compute_P!(∇V, P, Vx, Vy, Gdτ, r, dx, dy)
        @parallel compute_τ!(τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, Gdτ, Vx, Vy, η, G, dt, dx, dy)
        @parallel compute_dV!(dVx, dVy, P, τxx, τyy, τxy, dτ_Rho, dx, dy)
        @parallel compute_V!(Vx, Vy, dVx, dVy)

        # free slip boundary conditions
        if (freeslip_x) @parallel (1:size(Vx,1)) free_slip_y!(Vx) end
        if (freeslip_y) @parallel (1:size(Vy,2)) free_slip_x!(Vy) end

        iter += 1
        if iter % nout == 0
            @parallel compute_Res!(Rx, Ry, P, τxx, τyy, τxy, dx, dy)
            Vmin, Vmax = minimum(Vx), maximum(Vx)
            Pmin, Pmax = minimum(P), maximum(P)
            norm_Rx    = norm(Rx)/(Pmax-Pmin)*lx/sqrt(length(Rx))
            norm_Ry    = norm(Ry)/(Pmax-Pmin)*lx/sqrt(length(Ry))
            norm_∇V    = norm(∇V)/(Vmax-Vmin)*lx/sqrt(length(∇V))
            err = maximum([norm_Rx, norm_Ry, norm_∇V])
            push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter)
            @printf("Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n", iter, err, norm_Rx, norm_Ry, norm_∇V)
        end
    end

    return (iter= iter, err_evo1=err_evo1, err_evo2=err_evo2)
end