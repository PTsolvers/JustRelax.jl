const isCUDA = true

@static if isCUDA
    using CUDA
    CUDA.allowscalar(true)
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
import JustRelax.@cell

const backend_JR = @static if isCUDA
    CUDABackend          # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = @static if isCUDA
    CUDABackend        # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using Printf, Statistics, LinearAlgebra, GeoParams, GLMakie
import GeoParams.Dislocation
using StaticArrays, GeophysicalModelGenerator, WriteVTK, Interpolations, JLD2

# -----------------------------------------------------
include("CalderaModelSetup.jl")
include("CalderaRheology.jl")
# -----------------------------------------------------
## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    esc(:($A[$idx_j]))
end

function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    @parallel f_x(A, B)
end


function BC_velo!(Vx,Vy, εbg, xvi, lx,ly)
    xv, yv = xvi

    # @parallel_indices (i, j) function pure_shear_x!(Vx)
    #     xi = xv[i]
    #     yi = min(yv[j],0.0)
    #     Vx[i, j + 1] = yi < 0 ? (εbg * (xi - lx * 0.5) * (lx)/2) : 0.0
    #     return nothing
    # end
    @parallel_indices (i, j) function pure_shear_x!(Vx)
        xi = xv[i]
        Vx[i, j + 1] = εbg * (xi - lx * 0.5) * lx / 2
        return nothing
    end

    @parallel_indices (i, j) function pure_shear_y!(Vy)
        yi = min(yv[j],0.0)
        Vy[i + 1, j] = (abs(yi) * εbg / ly) / 2
        return nothing
    end

    nx, ny = size(Vx)
    @parallel (1:nx, 1:(ny - 2)) pure_shear_x!(Vx)
    nx, ny = size(Vy)
    @parallel (1:(nx - 2), 1:ny) pure_shear_y!(Vy)

    return nothing
end

function BC_displ!(Ux,Uy, εbg, xvi, lx,ly, dt)
    xv, yv = xvi


    @parallel_indices (i, j) function pure_shear_x!(Ux)
        xi = xv[i]
        yi = min(yv[j],0.0)
        Ux[i, j + 1] = εbg * (xi - lx * 0.5) * lx * dt / 2
        return nothing
    end

    @parallel_indices (i, j) function pure_shear_y!(Uy)
        yi = min(yv[j],0.0)
        Uy[i + 1, j] = (abs(yi) * εbg / ly) *dt /2
        return nothing
    end

    nx, ny = size(Ux)
    @parallel (1:nx, 1:(ny - 2)) pure_shear_x!(Ux)
    nx, ny = size(Uy)
    @parallel (1:(nx - 2), 1:ny) pure_shear_y!(Uy)

    return nothing
end

@parallel_indices (i, j) function init_P!(P, ρg, z, phases,sticky_air)
    # if phases[i, j] == 4.0
    #     @all(P) = 0.0
    # else
        @all(P) = abs(@all(ρg) * (@all_j(z))) #* <((@all_j(z)), 0.0)
        # @all(P) = @all(ρg)
    # end
    return nothing
end

@parallel_indices (I...) function compute_melt_fraction!(ϕ, phase_ratios, rheology, args)
    ϕ[I...] = compute_melt_frac(rheology, (;T=args.T[I...]), phase_ratios[I...])
    return nothing
end

@inline function compute_melt_frac(rheology, args, phase_ratios)
    return GeoParams.compute_meltfraction_ratio(phase_ratios, rheology, args)
end

function phase_change!(phases, particles)
    ni = size(phases)
    @parallel_indices (I...) function _phase_change!(phases, px, py, index)

        @inbounds for ip in JustRelax.cellaxes(phases)
            #quick escape
            JustRelax.@cell(index[ip, I...]) == 0 && continue

            x = JustRelax.@cell px[ip,I...]
            y = (JustRelax.@cell py[ip,I...])
            phase_ij = @cell phases[ip, I...]
            if y > 0.0 && (phase_ij  == 2.0 || phase_ij  == 3.0)
                @cell phases[ip, I...] = 4.0
            end
        end
        return nothing
    end

    @parallel (@idx ni) _phase_change!( phases, particles.coords..., particles.index)
end

function phase_change!(phases, EII_pl, threshold, particles)
    ni = size(phases)
    @parallel_indices (I...) function _phase_change!(phases, EII_pl, threshold, px, py, index)

        @inbounds for ip in JustRelax.cellaxes(phases)
            #quick escape
            JustRelax.@cell(index[ip, I...]) == 0 && continue

            x = JustRelax.@cell px[ip,I...]
            y = (JustRelax.@cell py[ip,I...])
            phase_ij = @cell phases[ip, I...]
            EII_pl_ij = @cell EII_pl[ip, I...]
            if EII_pl_ij > threshold && (phase_ij < 4.0)
                @cell phases[ip, I...] = 2.0
            end
        end
        return nothing
    end

    @parallel (@idx ni) _phase_change!(phases, EII_pl, threshold, particles.coords..., particles.index)
end

function phase_change!(phases, melt_fraction, threshold, sticky_air_phase, particles)
    ni = size(phases)
    @parallel_indices (I...) function _phase_change!(phases, melt_fraction, threshold, sticky_air_phase, px, py, index)

        @inbounds for ip in JustRelax.cellaxes(phases)
            #quick escape
            JustRelax.@cell(index[ip, I...]) == 0 && continue

            x = JustRelax.@cell px[ip,I...]
            y = (JustRelax.@cell py[ip,I...])
            phase_ij = @cell phases[ip, I...]
            melt_fraction_ij = @cell melt_fraction[ip, I...]
            if melt_fraction_ij < threshold && (phase_ij < sticky_air_phase)
                @cell phases[ip, I...] = 1.0
            end
        end
        return nothing
    end

    @parallel (@idx ni) _phase_change!(phases, melt_fraction, threshold, sticky_air_phase, particles.coords..., particles.index)
end

function circular_perturbation!(T, δT, max_temperature, xc_anomaly, yc_anomaly, r_anomaly, xvi)

    @parallel_indices (i, j) function _circular_perturbation!(T, δT, max_temperature, xc_anomaly, yc_anomaly, r_anomaly, x, y)
        @inbounds if  ((x[i] - xc_anomaly)^2 + (y[j] - yc_anomaly)^2 ≤ r_anomaly^2)
            new_temperature = T[i+1, j] * (δT / 100 + 1)
            T[i+1, j] = new_temperature > max_temperature ? max_temperature : new_temperature
        end
        return nothing
    end

    nx, ny = size(T)

    @parallel (1:nx-2, 1:ny) _circular_perturbation!(T, δT, max_temperature, xc_anomaly, yc_anomaly, r_anomaly, xvi...)
end

function new_thermal_anomaly!(phases, particles, xc_anomaly, yc_anomaly, r_anomaly)
    ni = size(phases)

    @parallel_indices (I...) function new_anomlay_particles(phases, px, py, index, xc_anomaly, yc_anomaly, r_anomaly)
        @inbounds for ip in JustRelax.cellaxes(phases)
            @cell(index[ip, I...]) == 0 && continue

            x = JustRelax.@cell px[ip, I...]
            y = JustRelax.@cell py[ip, I...]

            # thermal anomaly - circular
            if ((x - xc_anomaly)^2 + (y - yc_anomaly)^2 ≤ r_anomaly^2)
                JustRelax.@cell phases[ip, I...] = 3.0
            end
        end
        return nothing
    end
    @parallel (@idx ni) new_anomlay_particles(phases, particles.coords..., particles.index, xc_anomaly, yc_anomaly, r_anomaly)
end


## quick and dirty function
function add_thermal_anomaly!(
    pPhases, particles, interval, lx, CharDim, thermal, T_buffer, Told_buffer, Tsurf, xvi, phase_ratios, grid, pT
)
    new_thermal_anomaly!(pPhases, particles, lx*0.5, nondimensionalize(-5km, CharDim), nondimensionalize(0.5km, CharDim))
    circular_perturbation!(thermal.T, 30.0, nondimensionalize(1250C, CharDim), lx*0.5, nondimensionalize(-5km, CharDim), nondimensionalize(0.5km, CharDim), xvi)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    @views T_buffer[:,end] .= Tsurf
    @views thermal.T[2:end-1, :] .= T_buffer
    temperature2center!(thermal)
    grid2particle_flip!(pT, xvi, T_buffer, Told_buffer, particles)
    phase_ratios_center!(phase_ratios, particles, grid, pPhases)
    interval += 1.0
end

function plot_particles(particles, pPhases)
    p = particles.coords
    # pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
    ppx, ppy = p
    # pxv = ustrip.(dimensionalize(ppx.data[:], km, CharDim))
    # pyv = ustrip.(dimensionalize(ppy.data[:], km, CharDim))
    pxv = ppx.data[:]
    pyv = ppy.data[:]
    clr = pPhases.data[:]
    # clr = pϕ.data[:]
    idxv = particles.index.data[:]
    f,ax,h=scatter(Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma, markersize=1)
    Colorbar(f[1,2], h)
    f
end


# [...]


@views function Caldera_2D(igg; figname=figname, nx=64, ny=64, nz=64, do_vtk=false)

    #-----------------------------------------------------
    # USER INPUTS
    #-----------------------------------------------------
    # Characteristic lengths for nondimensionalisation
    CharDim         = GEO_units(; length=40km, viscosity=1e20Pa * s, temperature=1000C)
    #-----------------------------------------------------
    # Define model to be run
    nt              = 500                       # number of timesteps
    DisplacementFormulation = false              #specify if you want to use the displacement formulation
    Topography      = false;                    #specify if you want topography plotted in the figures
    Freesurface     = true                      #specify if you want to use freesurface
        sticky_air  = 5                         #specify the thickness of the sticky air layer in km
    toy = true                                  #specify if you want to use the toy model or the Toba model

    shear = true                                #specify if you want to use pure shear boundary conditions
    εbg_dim   = 1e-14 / s * shear                                 #specify the background strain rate

    # IO --------------------------------------------------
    # if it does not exist, make folder where figures are stored
    figdir = "./fig2D/$figname/"
    checkpoint = joinpath(figdir, "checkpoint")
    if do_vtk
        vtk_dir      = joinpath(figdir,"vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------
    # Set up the grid
    # ----------------------------------------------------
    if Topography == true
        li_GMG, origin_GMG, phases_GMG, T_GMG = Toba_setup2D(nx+1,ny+1,nz+1; sticky_air=sticky_air)
    elseif Freesurface == true
        li_GMG, origin_GMG, phases_GMG, T_GMG, Grid = volcano_setup2D(nx+1,ny+1,nz+1; sticky_air=sticky_air)
    else
        li_GMG, origin_GMG, phases_GMG, T_GMG = simple_setup_no_FS2D(nx+1,ny+1,nz+1)
    end
    # -----------------------------------------------------
    # Set up the JustRelax model
    # -----------------------------------------------------
    sticky_air      = nondimensionalize(sticky_air*km, CharDim)             # nondimensionalize sticky air
    lx              = nondimensionalize(li_GMG[1]*km, CharDim)              # nondimensionalize domain length in x-direction
    lz              = nondimensionalize(li_GMG[end]*km, CharDim)            # nondimensionalize domain length in y-direction
    li              = (lx, lz)                                              # domain length in x- and y-direction
    ni              = (nx, nz)                                              # number of grid points in x- and y-direction
    di              = @. li / ni                                            # grid spacing in x- and y-direction
    origin          = ntuple(Val(2)) do i
        nondimensionalize(origin_GMG[i] * km,CharDim)                       # origin coordinates of the domain
    end
    grid         = Geometry(ni, li; origin=origin)
    (; xci, xvi) = grid                                                     # nodes at the center and vertices of the cells

    εbg          = nondimensionalize(εbg_dim, CharDim)                      # background strain rate
    perturbation_C = @rand(ni...);                                          # perturbation of the cohesion

    # Physical Parameters
    rheology     = init_rheology(CharDim; is_compressible=true, linear = false)
    cutoff_visc  = nondimensionalize((1e17Pa*s, 1e23Pa*s),CharDim)
    κ            = (4 / (rheology[1].HeatCapacity[1].Cp.val * rheology[1].Density[1].ρ0.val))                                 # thermal diffusivity
    # κ            = (4 / (rheology[2].HeatCapacity[1].Cp.Cp.val * rheology[2].Density[1].ρ0.val))                                 # thermal diffusivity
    dt           = dt_diff = 0.5 * min(di...)^2 / κ / 2.01

    # Initalize particles ----------------------------------
    nxcell           = 30
    max_xcell        = 40
    min_xcell        = 20
    particles        = init_particles(backend, nxcell, max_xcell, min_xcell, xvi...);

    subgrid_arrays   = SubgridDiffusionCellArrays(particles);
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di);
    # temperature
    pT, pT0, pPhases, pη_vep, pEII, pϕ    = init_cell_arrays(particles, Val(6));
    particle_args       = (pT, pT0, pPhases, pη_vep, pEII, pϕ);

    # Assign material phases --------------------------
    phases_dev   = PTArray(backend_JR)(phases_GMG)
    phase_ratios = PhaseRatio(backend_JR, ni, length(rheology));
    init_phases2D!(pPhases, phases_dev, particles, xvi)
    phase_ratios_center!(phase_ratios, particles, grid, pPhases)

    thermal         = ThermalArrays(backend_JR, ni)
    @views thermal.T[2:end-1, :] .= PTArray(backend_JR)(nondimensionalize(T_GMG.*C, CharDim))
    thermal_bc      = TemperatureBoundaryConditions(;
        no_flux     = (left = true, right = true, top = false, bot = false),
    )
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)


    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes          = StokesArrays(backend_JR, ni)
    pt_stokes       = PTStokesCoeffs(
        li, di; 
        ϵ  = 1e-4, CFL=1 / √2.1,
        Re = 3π,
        r  = 0.7
    ) #ϵ=1e-4,  CFL=1 / √2.1 CFL=0.27 / √2.1
    # -----------------------------------------------------
    args = (; T=thermal.Tc, P=stokes.P, dt=dt)#,  ΔTc=thermal.ΔTc, perturbation_C = perturbation_C)

    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=0.98 / √2.1
    )
    # Boundary conditions of the flow
    if shear == true && DisplacementFormulation == true
        BC_displ!(@displacement(stokes)..., εbg, xvi,lx,lz,dt)
        flow_bcs = DisplacementBoundaryConditions(;
            free_slip=(left=true, right=true, top=true, bot=true),
            free_surface =false,
        )
        flow_bcs!(stokes, flow_bcs) # apply boundary conditions
        displacement2velocity!(stokes, dt) # convert displacement to velocity
        update_halo!(@velocity(stokes)...) # update halo cells
    elseif shear == true && DisplacementFormulation == false
        BC_velo!(@velocity(stokes)..., εbg, xvi,lx,lz)
        flow_bcs = VelocityBoundaryConditions(;
            free_slip=(left=true, right=true, top=true, bot=true),
            free_surface =false,
        )
        flow_bcs!(stokes, flow_bcs) # apply boundary conditions
        update_halo!(@velocity(stokes)...) # update halo cells
    else
        flow_bcs = VelocityBoundaryConditions(;
            free_slip    = (left=true, right=true, top=true, bot=true),
            free_surface =false,
        )
        flow_bcs!(stokes, flow_bcs) # apply boundary conditions
        update_halo!(@velocity(stokes)...) # update halo cells
    end

    # Melt Fraction
    ϕ = @zeros(ni...)

    # Buoyancy force
    ρg = @zeros(ni...), @zeros(ni...)                      # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction

    # Preparation for Visualisation
    ni_v_viz  = nx_v_viz, ny_v_viz = (ni[1] - 1) * igg.dims[1], (ni[2] - 1) * igg.dims[2]      # size of the visualisation grid on the vertices according to MPI dims
    ni_viz    = nx_viz, ny_viz = (ni[1] - 2) * igg.dims[1], (ni[2] - 2) * igg.dims[2]            # size of the visualisation grid on the vertices according to MPI dims
    Vx_vertex = PTArray(backend_JR)(ones(ni .+ 1...))                                                  # initialise velocity for the vertices in x direction
    Vy_vertex = PTArray(backend_JR)(ones(ni .+ 1...))                                                  # initialise velocity for the vertices in y direction

    global_grid         = Geometry(ni_viz, li; origin = origin)
    (global_xci, global_xvi) = (global_grid.xci, global_grid.xvi) # nodes at the center and vertices of the cells
    # Arrays for visualisation
    Tc_viz    = Array{Float64}(undef,ni_viz...)                                   # Temp center with ni
    Vx_viz    = Array{Float64}(undef,ni_v_viz...)                                 # Velocity in x direction with ni_viz .-1
    Vy_viz    = Array{Float64}(undef,ni_v_viz...)                                 # Velocity in y direction with ni_viz .-1
    ∇V_viz    = Array{Float64}(undef,ni_viz...)                                   # Velocity in y direction with ni_viz .-1
    P_viz     = Array{Float64}(undef,ni_viz...)                                   # Pressure with ni_viz .-2
    τxy_viz   = Array{Float64}(undef,ni_v_viz...)                                 # Shear stress with ni_viz .-1
    τII_viz   = Array{Float64}(undef,ni_viz...)                                   # 2nd invariant of the stress tensor with ni_viz .-2
    εII_viz   = Array{Float64}(undef,ni_viz...)                                   # 2nd invariant of the strain tensor with ni_viz .-2
    EII_pl_viz= Array{Float64}(undef,ni_viz...)                                   # 2nd invariant of the strain tensor with ni_viz .-2
    εxy_viz   = Array{Float64}(undef,ni_v_viz...)                                 # Shear strain with ni_viz .-1
    η_viz     = Array{Float64}(undef,ni_viz...)                                   # Viscosity with ni_viz .-2
    η_vep_viz = Array{Float64}(undef,ni_viz...)                                   # Viscosity for the VEP with ni_viz .-2
    ϕ_viz     = Array{Float64}(undef,ni_viz...)                                   # Melt fraction with ni_viz .-2
    ρg_viz    = Array{Float64}(undef,ni_viz...)                                   # Buoyancy force with ni_viz .-2

    args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt, ΔTc=thermal.ΔTc, perturbation_C = perturbation_C)

    for _ in 1:5
        compute_ρg!(ρg[end], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
        @parallel (@idx ni) init_P!(stokes.P, ρg[2], xci[2],phases_dev, sticky_air)
    end

    @parallel (@idx ni) compute_melt_fraction!(
        ϕ, phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P)
    )
    compute_viscosity!(stokes, phase_ratios, args, rheology, cutoff_visc)

    @copy stokes.P0 stokes.P
    @copy thermal.Told thermal.T

    t, it      = 0.0, 0
    interval   = 1.0
    dt_new     = dt *0.1
    iterMax_stokes = 250e3
    iterMax_thermal = 10e3
    local Vx_v, Vy_v
    if do_vtk
        Vx_v = @zeros(ni.+1...)
        Vy_v = @zeros(ni.+1...)
    end

    T_buffer    = @zeros(ni.+1)
    Told_buffer = similar(T_buffer)
    Tsurf  = nondimensionalize(0C,CharDim)
    Tbot   = nondimensionalize(575C,CharDim)
    dt₀         = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)
    centroid2particle!(pη_vep, xci, stokes.viscosity.η_vep, particles)
    centroid2particle!(pEII, xci, stokes.EII_pl, particles)
    centroid2particle!(pϕ, xci, ϕ, particles)
    pT0.data    .= pT.data

    ## Plot initial T and P profile
    let
        Yv = [y for x in xvi[1], y in xvi[2]][:]
        Y = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(; size=(1200, 900))
        ax1 = Axis(fig[1, 1]; aspect=2 / 3, title="T")
        ax2 = Axis(fig[1, 2]; aspect=2 / 3, title="Pressure")
        scatter!(
            ax1,
            Array(ustrip.(dimensionalize(thermal.T[2:(end - 1), :][:], C, CharDim))),
            ustrip.(dimensionalize(Yv, km, CharDim)),
        )
        lines!(
            ax2,
            Array(ustrip.(dimensionalize(stokes.P[:], MPa, CharDim))),
            ustrip.(dimensionalize(Y, km, CharDim)),
        )
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    # for it in 1:2500 #nt
    while it < 2500 #nt

        dt = dt_new # update dt
        if DisplacementFormulation == true
            BC_displ!(@displacement(stokes)..., εbg, xvi,lx,lz,dt)
            flow_bcs!(stokes, flow_bcs) # apply boundary conditions
            # BC_velo!(@velocity(stokes)..., εbg, xvi,lx,lz)
        end

        # if it > 1 && ustrip(dimensionalize(t,yr,CharDim)) >= (ustrip.(1.5e3yr)*interval)
        #     add_thermal_anomaly!(
        #         pPhases, particles, interval, lx, CharDim, thermal, T_buffer, Told_buffer, Tsurf, xvi, phase_ratios, grid, pT
        #     )
        # end

        args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt, ΔTc=thermal.ΔTc, perturbation_C = perturbation_C)
        ## Stokes solver -----------------------------------
        stokes.viscosity.η .= mean(stokes.viscosity.η
        )
        iter, err_evo1 =
        solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            rheology,
            args,
            dt,
            igg;
            kwargs = (;
                iterMax          = 150e3,#250e3,
                free_surface     = false,
                nout             = 2e3,#5e3,
                viscosity_relaxation=1e-3,
                viscosity_cutoff = cutoff_visc,
            )
        )
        dt_new = compute_dt(stokes, di, dt_diff, igg) #/ 9.81
        dt     = dt_new
        
        tensor_invariant!(stokes.ε)

        ## Save the checkpoint file before a possible thermal solver blow up
        checkpointing_jld2(joinpath(checkpoint, "thermal"), stokes, thermal, t, dt, igg)

        # ------------------------------
        compute_shear_heating!(
            thermal,
            stokes,
            phase_ratios,
            rheology, # needs to be a tuple
            dt,
        )
        # Thermal solver ---------------
        iter_count, norm_ResT =
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di;
            kwargs =(;
                igg     = igg,
                phase   = phase_ratios,
                iterMax = 50e3,
                nout    = 100,
                verbose = true,
            )
        )

        for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
            copyinn_x!(dst, src)
        end
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, T_buffer, thermal.ΔT[2:end-1, :], subgrid_arrays, particles, xvi,  di, dt
        )
        @parallel (@idx ni) compute_melt_fraction!(
            ϕ, phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P)
        )
        # ------------------------------
        # Update the particles Arguments
        centroid2particle!(pη_vep, xci, stokes.viscosity.η_vep, particles)
        centroid2particle!(pEII, xci, stokes.EII_pl, particles)
        centroid2particle!(pϕ, xci, ϕ, particles)

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advection_LinP!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advection_MQS!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)

        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer, ), xvi)

        # phase change for particles
        # phase_change!(pPhases, pϕ, 0.05, 4.0, particles)
        # phase_change!(pPhases, pEII, 1e-2, particles)
        # phase_change!(pPhases, particles)

        # update phase ratios
        phase_ratios_center!(phase_ratios, particles, grid, pPhases)

        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end] .= Tsurf;
        @views T_buffer[:, 1] .= Tbot;
        @views thermal.T[2:end - 1, :] .= T_buffer;
        thermal_bcs!(thermal.T, thermal_bc)
        temperature2center!(thermal)
        vertex2center!(thermal.ΔTc, thermal.ΔT)

        # dt_new =  compute_dt(stokes, di, dt_diff, igg) #/ 9.81

        @show it += 1
        t += dt

        ## Plotting -------------------------------------------------------
        if it == 1 || rem(it, 1) == 0
            if igg.me == 0 && it == 1
                metadata(pwd(), checkpoint, basename(@__FILE__), "CalderaModelSetup.jl", "CalderaRheology.jl")
            end
            checkpointing_jld2(checkpoint, stokes, thermal, t, dt, igg)
            ## Somehow fails to open with load("particles.jld2")
            mktempdir() do tmpdir
                # Save the checkpoint file in the temporary directory
                tmpfname = joinpath(tmpdir, basename(joinpath(checkpoint, "particles.jld2")))
                jldsave(
                    tmpfname;
                    particles=Array(particles),
                    pPhases=Array(pPhases),
                    time=t,
                    timestep=dt,
                )
                # Move the checkpoint file from the temporary directory to the destination directory
                mv(tmpfname, joinpath(checkpoint, "particles.jld2"); force=true)
            end


            velocity2vertex!(Vx_vertex, Vy_vertex, stokes.V.Vx, stokes.V.Vy)

            x_v = ustrip.(dimensionalize(global_xvi[1], km, CharDim))  #not sure about this with MPI and the size (intuition says should be fine)
            y_v = ustrip.(dimensionalize(global_xvi[2], km, CharDim))
            x_c = ustrip.(dimensionalize(global_xci[1], km, CharDim))
            y_c = ustrip.(dimensionalize(global_xci[2], km, CharDim))

            T_inn = Array(thermal.Tc[2:(end - 1), 2:(end - 1)])
            Vx_inn = Array(Vx_vertex[2:(end - 1), 2:(end - 1)])
            Vy_inn = Array(Vy_vertex[2:(end - 1), 2:(end - 1)])
            ∇V_inn = Array(stokes.∇V[2:(end - 1), 2:(end - 1)])
            P_inn = Array(stokes.P[2:(end - 1), 2:(end - 1)])
            τII_inn = Array(stokes.τ.II[2:(end - 1), 2:(end - 1)])
            τxy_inn = Array(stokes.τ.xy[2:(end - 1), 2:(end - 1)])
            EII_pl_inn = Array(stokes.EII_pl[2:(end - 1), 2:(end - 1)])
            εII_inn = Array(stokes.ε.II[2:(end - 1), 2:(end - 1)])
            εxy_inn = Array(stokes.ε.xy[2:(end - 1), 2:(end - 1)])
            η_inn = Array(stokes.viscosity.η[2:(end - 1), 2:(end - 1)])
            η_vep_inn = Array(stokes.viscosity.η_vep[2:(end - 1), 2:(end - 1)])
            ϕ_inn = Array(ϕ[2:(end - 1), 2:(end - 1)])
            ρg_inn = Array(ρg[2][2:(end - 1), 2:(end - 1)])

            gather!(T_inn, Tc_viz)
            gather!(Vx_inn, Vx_viz)
            gather!(Vy_inn, Vy_viz)
            gather!(∇V_inn, ∇V_viz)
            gather!(P_inn, P_viz)
            gather!(τII_inn, τII_viz)
            gather!(τxy_inn, τxy_viz)
            gather!(EII_pl_inn, EII_pl_viz)
            gather!(εII_inn, εII_viz)
            gather!(εxy_inn, εxy_viz)
            gather!(η_inn, η_viz)
            gather!(η_vep_inn, η_vep_viz)
            gather!(ϕ_inn, ϕ_viz)
            gather!(ρg_inn, ρg_viz)

            T_d = ustrip.(dimensionalize(Array(Tc_viz), C, CharDim))
            η_d = ustrip.(dimensionalize(Array(η_viz), Pas, CharDim))
            η_vep_d = ustrip.(dimensionalize(Array(η_vep_viz), Pas, CharDim))
            Vy_d = ustrip.(dimensionalize(Array(Vy_viz), cm / yr, CharDim))
            Vx_d = ustrip.(dimensionalize(Array(Vx_viz), cm / yr, CharDim))
            ∇V_d = ustrip.(dimensionalize(Array(∇V_viz), cm / yr, CharDim))
            P_d = ustrip.(dimensionalize(Array(P_viz), MPa, CharDim))
            ρg_d = ustrip.(dimensionalize(Array(ρg_viz), kg / m^3 * m / s^2, CharDim))
            ρ_d = ρg_d / 10
            ϕ_d = Array(ϕ_viz)
            τII_d = ustrip.(dimensionalize(Array(τII_viz), MPa, CharDim))
            τxy_d = ustrip.(dimensionalize(Array(τxy_viz), MPa, CharDim))
            EII_pl_d = Array(EII_pl_viz)
            εII_d = ustrip.(dimensionalize(Array(εII_viz), s^-1, CharDim))
            εxy_d = ustrip.(dimensionalize(Array(εxy_viz), s^-1, CharDim))
            t_yrs = dimensionalize(t, yr, CharDim)
            t_Kyrs = t_yrs / 1e3
            t_Myrs = t_Kyrs / 1e3

            p = particles.coords
            # pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
            ppx, ppy = p
            pxv = ustrip.(dimensionalize(ppx.data[:], km, CharDim))
            pyv = ustrip.(dimensionalize(ppy.data[:], km, CharDim))
            clr = pPhases.data[:]
            clrT = pT.data[:]
            idxv = particles.index.data[:]

            if do_vtk
                # velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    T   = Array(T_d),
                    τxy = Array(τxy_d),
                    εxy = Array(εxy_d),
                    Vx  = Array(Vx_d),
                    Vy  = Array(Vy_d),
                )
                data_c = (;
                    P   = Array(P_d),
                    τII = Array(τII_d),
                    η   = Array(η_d),
                    ϕ   = Array(ϕ_d),
                    ρ  = Array(ρ_d),
                )
                velocity_v = (
                    Array(Vx_d),
                    Array(Vy_d),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    (x_v,y_v),
                    (x_c,y_c),
                    data_v,
                    data_c,
                    velocity_v
                )
            end

            if igg.me == 0
                fig = Figure(; size=(2000, 1800), createmissing=true)
                ar = li[1] / li[2]
                # ar = DataAspect()

                ax0 = Axis(
                    fig[1, 1:2];
                    aspect=ar,
                    title="t = $(round.(ustrip.(t_Kyrs); digits=3)) Kyrs",
                    titlesize=50,
                    height=0.0,
                )
                ax0.ylabelvisible = false
                ax0.xlabelvisible = false
                ax0.xgridvisible = false
                ax0.ygridvisible = false
                ax0.xticksvisible = false
                ax0.yticksvisible = false
                ax0.yminorticksvisible = false
                ax0.xminorticksvisible = false
                ax0.xgridcolor = :white
                ax0.ygridcolor = :white
                ax0.ytickcolor = :white
                ax0.xtickcolor = :white
                ax0.yticklabelcolor = :white
                ax0.xticklabelcolor = :white
                ax0.yticklabelsize = 0
                ax0.xticklabelsize = 0
                ax0.xlabelcolor = :white
                ax0.ylabelcolor = :white

                ax1 = Axis(
                    fig[2, 1][1, 1];
                    aspect=ar,
                    title=L"T [\mathrm{C}]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax2 = Axis(
                    fig[2, 2][1, 1];
                    aspect=ar,
                    title=L"\log_{10}(\eta_{vep}) [\mathrm{Pas}]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax3 = Axis(
                    fig[3, 1][1, 1];
                    aspect=ar,
                    title=L"Vy [\mathrm{cm/yr}]",
                    # title=L"\tau_{\textrm{II}} [MPa]",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax4 = Axis(
                    fig[3, 2][1, 1];
                    aspect=ar,
                    title=L"\log_{10}(\dot{\varepsilon}_{\textrm{II}}) [\mathrm{s}^{-1}]",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax5 = Axis(
                    fig[4, 1][1, 1];
                    aspect=ar,
                    title=L"Plastic Strain",
                    # title=L"Phases",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax6 = Axis(
                    fig[4, 2][1, 1];
                    aspect=ar,
                    title=L"\tau_{\textrm{II}} [MPa]",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )

                linkyaxes!(ax1, ax2)
                linkyaxes!(ax3, ax4)
                hidexdecorations!(ax1; grid=false)
                hideydecorations!(ax2; grid=false)
                hidexdecorations!(ax3; grid=false)
                hidexdecorations!(ax2; grid=false)
                # hidexdecorations!(ax4; grid=false)
                # pp = [argmax(p) for p in phase_ratios.center];
                # @views pp = pp[2:end-1,2:end-1]
                # @views T_d[pp.==4.0] .= NaN
                # @views η_vep_d[pp.==4.0] .= NaN
                # @views τII_d[pp.==4.0] .= NaN
                # @views εII_d[pp.==4.0] .= NaN
                # @views EII_pl_d[pp.==4.0] .= NaN
                # @views ϕ_d[pp.==4.0] .= NaN
                # @views Vy_d[1:end-1, 1:end-1][pp.==4.0] .=NaN

                p1 = heatmap!(ax1, x_c, y_c, T_d; colormap=:batlow, colorrange=(000, 1200))
                contour!(ax1, x_c, y_c, T_d, ; color=:white, levels=600:200:1200)
                p2 = heatmap!(ax2, x_c, y_c, log10.(η_vep_d); colormap=:glasgow)#, colorrange= (log10(1e16), log10(1e22)))
                contour!(ax2, x_c, y_c, T_d, ; color=:white, levels=600:200:1200, labels = true)
                p3 = heatmap!(ax3, x_v, y_v, Vy_d; colormap=:vik)
                p4 = heatmap!(ax4, x_c, y_c, log10.(εII_d); colormap=:glasgow, colorrange= (log10(5e-15), log10(5e-12)))
                p5 = heatmap!(ax5, x_c, y_c, EII_pl_d; colormap=:glasgow)
                contour!(ax5, x_c, y_c, T_d, ; color=:white, levels=600:200:1200, labels = true)
                p6 = heatmap!(ax6, x_v, y_v, τII_d; colormap=:batlow)

                Colorbar(
                    fig[2, 1][1, 2], p1; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[2, 2][1, 2], p2; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[3, 1][1, 2], p3; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[3, 2][1, 2], p4; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[4, 1][1, 2], p5; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[4, 2][1, 2], p6; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                rowgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                fig
                figsave = joinpath(figdir, @sprintf("%06d.png", it))
                save(figsave, fig)

                let
                    Yv = [y for x in xvi[1], y in xvi[2]][:]
                    Y = [y for x in xci[1], y in xci[2]][:]
                    fig = Figure(; size=(1200, 900))
                    ax1 = Axis(fig[1, 1]; aspect=2 / 3, title="T")
                    ax2 = Axis(fig[1, 2]; aspect=2 / 3, title="Pressure")

                    scatter!(
                        ax1,
                        Array(ustrip.(dimensionalize(thermal.T[2:(end - 1), :][:], C, CharDim))),
                        ustrip.(dimensionalize(Yv, km, CharDim)),
                    )
                    lines!(
                        ax2,
                        Array(ustrip.(dimensionalize(stokes.P[:], MPa, CharDim))),
                        ustrip.(dimensionalize(Y, km, CharDim)),
                    )

                    hideydecorations!(ax2)
                    save(joinpath(figdir, "pressure_profile_$it.png"), fig)
                    fig
                end

                let
                    p = particles.coords
                    # pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
                    ppx, ppy = p
                    # pxv = ustrip.(dimensionalize(ppx.data[:], km, CharDim))
                    # pyv = ustrip.(dimensionalize(ppy.data[:], km, CharDim))
                    pxv = ppx.data[:]
                    pyv = ppy.data[:]
                    clr = pPhases.data[:]
                    # clrT = pT.data[:]
                    idxv = particles.index.data[:]
                    f,ax,h=scatter(Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma, markersize=1)
                    Colorbar(f[1,2], h)
                    save(joinpath(figdir, "particles_$it.png"), f)
                    f
                end
            end
        end
    end
end

figname = "debug_viscosity"
# mkdir(figname)
do_vtk = true
ar = 2 # aspect ratio
n = 64
nx = n * ar
ny = n
nz = n
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI=true)...)
else
    igg
end

Caldera_2D(igg; figname=figname, nx=nx, ny=ny, nz=nz, do_vtk=do_vtk)
