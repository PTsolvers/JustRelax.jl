# Benchmark of Duretz et al. 2014
# http://dx.doi.org/10.1002/2014GL060438
using JustRelax, JustRelax.DataIO, JustPIC
import JustRelax.@cell

## NOTE: need to run one of the lines below if one wishes to switch from one backend to another
# set_backend("Threads_Float64_2D")
# set_backend("CUDA_Float64_2D")

# setup ParallelStencil.jl environment
model = PS_Setup(:Threads, Float64, 2)
environment!(model)

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, CellArrays
using GLMakie

# Load file with all the rheology configurations
include("Shearheating_rheology.jl")


## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------
@inline init_particle_fields(particles) = @zeros(size(particles.coords[1])...)
@inline init_particle_fields(particles, nfields) = tuple([zeros(particles.coords[1]) for i in 1:nfields]...)
@inline init_particle_fields(particles, ::Val{N}) where N = ntuple(_ -> @zeros(size(particles.coords[1])...) , Val(N))
@inline init_particle_fields_cellarrays(particles, ::Val{N}) where N = ntuple(_ -> @fill(0.0, size(particles.coords[1])..., celldims=(cellsize(particles.index))), Val(N))

function init_particles_cellarrays(nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny)
    ni     = nx, ny
    ncells = nx * ny
    np     = max_xcell * ncells
    px, py = ntuple(_ -> @fill(NaN, ni..., celldims=(max_xcell,)) , Val(2))
    inject = @fill(false, nx, ny, eltype=Bool)
    index  = @fill(false, ni..., celldims=(max_xcell,), eltype=Bool)

    @parallel_indices (i, j) function fill_coords_index(px, py, index)
        # lower-left corner of the cell
        x0, y0 = x[i], y[j]
        # fill index array
        for l in 1:nxcell
            JustRelax.@cell px[l, i, j]    = x0 + dx * rand(0.05:1e-5:0.95)
            JustRelax.@cell py[l, i, j]    = y0 + dy * rand(0.05:1e-5:0.95)
            JustRelax.@cell index[l, i, j] = true
        end
        return nothing
    end

    @parallel (1:nx, 1:ny) fill_coords_index(px, py, index)

    return Particles(
        (px, py), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny)
    )
end

# Velocity helper grids for the particle advection
function velocity_grids(xci, xvi, di)
    dx, dy  = di
    yVx     = LinRange(xci[2][1] - dy, xci[2][end] + dy, length(xci[2])+2)
    xVy     = LinRange(xci[1][1] - dx, xci[1][end] + dx, length(xci[1])+2)
    grid_vx = xvi[1], yVx
    grid_vy = xVy, xvi[2]

    return grid_vx, grid_vy
end

function copyinn_x!(A, B)

    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    @parallel f_x(A, B)
end

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    esc(:($A[$idx_j]))
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_j(z)) * <(@all_j(z), 0.0)
    return nothing
end


## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main2D(igg; ar=8, ny=16, nx=ny*8, figdir="figs2D", save_vtk =false)

    # Physical domain ------------------------------------
    ly           = 40e3              # domain length in y
    lx           = 70e3           # domain length in x
    ni           = nx, ny            # number of cells
    li           = lx, ly            # domain length in x- and y-
    di           = @. li / ni        # grid step in x- and -y
    origin       = 0.0, -ly          # origin coordinates (15km f sticky air layer)
    xci, xvi     = lazy_grid(di, li, ni; origin=origin) # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = init_rheologies(; is_TP_Conductivity=false)
    κ            = (4 / (rheology[1].HeatCapacity[1].cp * rheology[1].Density[1].ρ))
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 40, 1
    particles = init_particles_cellarrays(
        nxcell, max_xcell, min_xcell, xvi[1], xvi[2], di[1], di[2], nx, ny
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases      = init_particle_fields_cellarrays(particles, Val(3))
    particle_args    = (pT, pPhases)

    # Elliptical temperature anomaly
    xc_anomaly       = lx/2    # origin of thermal anomaly
    yc_anomaly       = 40e3  # origin of thermal anomaly
    # yc_anomaly       = 39e3  # origin of thermal anomaly
    r_anomaly        = 3e3    # radius of perturbation
    init_phases!(pPhases, particles, lx/2, yc_anomaly, r_anomaly)
    phase_ratios     = PhaseRatio(ni, length(rheology))
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(ni, ViscoElastic)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.9 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(ni)
    thermal_bc       = TemperatureBoundaryConditions(;
        no_flux      = (left = true, right = true, top = false, bot = false),
        periodicity  = (left = false, right = false, top = false, bot = false),
    )

    # Initialize constant temperature
    @views thermal.T .= 273.0 + 400
    thermal_bcs!(thermal.T, thermal_bc)

    @parallel (JustRelax.@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg               = @zeros(ni...), @zeros(ni...)

    @parallel (JustRelax.@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P))
    @parallel init_P!(stokes.P, ρg[2], xci[2])

    # Rheology
    η                = @zeros(ni...)
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (-Inf, Inf)
    )
    η_vep            = copy(η)

    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL= 1e-3 / √2.1
    )

    # Boundary conditions
    flow_bcs         = FlowBoundaryConditions(;
        free_slip    = (left = true, right=true, top=true, bot=true),
        periodicity  = (left = false, right = false, top = false, bot = false),
    )
    ## Compression and not extension - fix this
    εbg              = 5e-14
    stokes.V.Vx .= PTArray([ -(x - lx/2) * εbg for x in xvi[1], _ in 1:ny+2])
    stokes.V.Vy .= PTArray([ (ly - abs(y)) * εbg for _ in 1:nx+2, y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions

    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    if save_vtk
        vtk_dir      = figdir*"\\vtk"
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    # Plot initial T and η profiles
    let
        Yv  = [y for x in xvi[1], y in xvi[2]][:]
        Y   = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
        scatter!(ax1, Array(thermal.T[2:end-1,:][:]), Yv./1e3)
        scatter!(ax2, Array(log10.(η[:])), Y./1e3)
        ylims!(ax1, minimum(xvi[2])./1e3, 0)
        ylims!(ax2, minimum(xvi[2])./1e3, 0)
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    T_buffer    = @zeros(ni.+1)
    Told_buffer = similar(T_buffer)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles.coords)

    local Vx_v, Vy_v
    if save_vtk
        Vx_v = @zeros(ni.+1...)
        Vy_v = @zeros(ni.+1...)
    end
    # Time loop
    t, it = 0.0, 0
    while it < 100
          # Update buoyancy and viscosity -
          args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)
          @parallel (@idx ni) compute_viscosity!(
              η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (-Inf, Inf)
          )
          @parallel (JustRelax.@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, args)
          # ------------------------------

          # Stokes solver ----------------
          solve!(
              stokes,
              pt_stokes,
              di,
              flow_bcs,
              ρg,
              η,
              η_vep,
              phase_ratios,
              rheology,
              args,
              Inf,
              igg;
              iterMax = 75e3,
              nout=1e3,
              viscosity_cutoff=(-Inf, Inf)
          )
          @parallel (JustRelax.@idx ni) tensor_invariant!(stokes.ε.II, @strain(stokes)...)
          dt   = compute_dt(stokes, di, dt_diff)
          # ------------------------------

          # interpolate fields from particle to grid vertices
          particle2grid!(T_buffer, pT, xvi, particles.coords)
          @views T_buffer[:, end]      .= 273.0 + 400
          @views thermal.T[2:end-1, :] .= T_buffer
          temperature2center!(thermal)

          @parallel (@idx ni) compute_shear_heating(
            thermal.shear_heating,
            @tensor_center(stokes.τ),
            @tensor_center(stokes.τ_o),
            @strain(stokes),
            phase_ratios.center,
            rheology, # needs to be a tuple
            dt,
          )

          # Thermal solver ---------------
          heatdiffusion_PT!(
              thermal,
              pt_thermal,
              thermal_bc,
              rheology,
              args,
              dt,
              di;
              igg     = igg,
              phase   = phase_ratios,
              iterMax = 10e3,
              nout    = 1e2,
              verbose = true,
          )
          # ------------------------------

          # Advection --------------------
          # advect particles in space
          advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 2 / 3)
          # advect particles in memory
          shuffle_particles!(particles, xvi, particle_args)
          # interpolate fields from grid vertices to particles
          for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
              copyinn_x!(dst, src)
          end
          grid2particle_flip!(pT, xvi, T_buffer, Told_buffer, particles.coords)
          # check if we need to inject particles
          inject = check_injection(particles)
          inject && inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer,), xvi)
          # update phase ratios
          @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)

          @show it += 1
          t        += dt

          # Data I/O and plotting ---------------------
          if it == 1 || rem(it, 10) == 0
              checkpointing(figdir, stokes, thermal.T, η, t)

              if save_vtk
                  JustRelax.velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                  data_v = (;
                      T   = Array(thermal.T[2:end-1, :]),
                      τxy = Array(stokes.τ.xy),
                      εxy = Array(stokes.ε.xy),
                      Vx  = Array(Vx_v),
                      Vy  = Array(Vy_v),
                  )
                  data_c = (;
                      P   = Array(stokes.P),
                      τxx = Array(stokes.τ.xx),
                      τyy = Array(stokes.τ.yy),
                      εxx = Array(stokes.ε.xx),
                      εyy = Array(stokes.ε.yy),
                      η   = Array(η),
                  )
                  save_vtk(
                      joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                      xvi,
                      xci,
                      data_v,
                      data_c
                  )
              end

              # Make particles plottable
              p        = particles.coords
              ppx, ppy = p
              pxv      = ppx.data[:]./1e3
              pyv      = ppy.data[:]./1e3
              clr      = pPhases.data[:]
              idxv     = particles.index.data[:];

              # Make Makie figure
              fig = Figure(size = (900, 900), title = "t = $t")
              ax1 = Axis(fig[1,1], aspect = ar, title = "T [C]  (t=$(t/(1e6 * 3600 * 24 *365.25)) Myrs)")
              ax2 = Axis(fig[2,1], aspect = ar, title = "Vy [m/s]")
              ax3 = Axis(fig[1,3], aspect = ar, title = "log10(εII)")
              ax4 = Axis(fig[2,3], aspect = ar, title = "log10(η)")
              # Plot temperature
              h1  = heatmap!(ax1, xvi[1].*1e-3, xvi[2].*1e-3, Array(thermal.T[2:end-1,:].-273.0) , colormap=:batlow)
              # Plot particles phase
              h2  = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]))
              # Plot 2nd invariant of strain rate
              h3  = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε.II)) , colormap=:batlow)
              # Plot effective viscosity
              h4  = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(η_vep)) , colormap=:batlow)
              hidexdecorations!(ax1)
              hidexdecorations!(ax2)
              hidexdecorations!(ax3)
              Colorbar(fig[1,2], h1)
              Colorbar(fig[2,2], h2)
              Colorbar(fig[1,4], h3)
              Colorbar(fig[2,4], h4)
              linkaxes!(ax1, ax2, ax3, ax4)
              save(joinpath(figdir, "$(it).png"), fig)
              fig
          end
          # ------------------------------

      end

      return nothing
  end

figdir   = "Benchmark_Duretz_etal_2014"
save_vtk = false # set to true to generate VTK files for ParaView
ar       = 1 # aspect ratio
n        = 128
nx       = n*ar - 2
ny       = n - 2
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end

main2D(igg; ar=ar, ny=ny, nx=nx, figdir=figdir, save_vtk=save_vtk)
