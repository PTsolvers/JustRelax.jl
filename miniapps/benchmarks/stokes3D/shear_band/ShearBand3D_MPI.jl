using JustRelax, JustRelax.JustRelax3D, JustRelax.DataIO

const backend_JR = CPUBackend

using GeoParams, CairoMakie

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)

using JustPIC, JustPIC._3D

const backend = JustPIC.CPUBackend

# HELPER FUNCTIONS ---------------------------------------------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

# Initialize phases on the particles
function init_phases!(phase_ratios, xci, xvi, radius)
    ni      = size(phase_ratios.center)
    origin  = 0.5, 0.5, 0.5

    @parallel_indices (i, j, k) function init_phases!(phases, xc, yc, zc, o_x, o_y, o_z)
        x, y, z = xc[i], yc[j], zc[k]
        if ((x-o_x)^2 + (y-o_y)^2 + (z-o_z)^2) > radius^2
            @index phases[1, i, j, k] = 1.0
            @index phases[2, i, j, k] = 0.0

        else
            @index phases[1, i, j, k] = 0.0
            @index phases[2, i, j, k] = 1.0
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phase_ratios.center, xci..., origin...)
    @parallel (@idx ni .+ 1) init_phases!(phase_ratios.vertex, xvi..., origin...)
    return nothing
end

# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx=64, ny=64, nz=64, figdir="model_figs", do_vtk=false)

    # Physical domain ------------------------------------
    lx = ly = lz = 1e0             # domain length in y
    ni           = nx, ny, nz      # number of cells
    li           = lx, ly, lz      # domain length in x- and y-
    di           = @. li / (nx_g(), ny_g(), nz_g()) # grid step in x- and -y and z-
    origin       = 0.0, 0.0, 0.0   # origin coordinates
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells    dt          = Inf

    # Physical properties using GeoParams ----------------
    τ_y         = 1.6           # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ           = 30            # friction angle
    C           = τ_y           # Cohesion
    η0          = 1.0           # viscosity
    G0          = 1.0           # elastic shear modulus
    Gi          = G0/(6.0-4.0)  # elastic shear modulus perturbation
    εbg         = 1.0           # background strain-rate
    # η_reg       = 8e-3          # regularisation "viscosity"
    η_reg       = 1.25e-2       # regularisation "viscosity"
    dt          = η0/G0/4.0     # assumes Maxwell time of 4
    el_bg       = ConstantElasticity(; G=G0, ν=0.5)
    el_inc      = ConstantElasticity(; G=Gi, ν=0.5)
    visc        = LinearViscous(; η=η0)
    pl          = DruckerPrager_regularised(;  # non-regularized plasticity
        C    = C,
        ϕ    = ϕ,
        η_vp = η_reg,
        Ψ    = 0.0,
    )
    rheology    = (
        # Low density phase
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            # CompositeRheology = CompositeRheology((visc, el_bg, )),
            CompositeRheology = CompositeRheology((visc, el_bg, pl)),
            Elasticity        = el_bg,

        ),
        # High density phase
        SetMaterialParams(;
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            # CompositeRheology = CompositeRheology((visc, el_inc, )),
            CompositeRheology = CompositeRheology((visc, el_inc, pl)),
            Elasticity        = el_inc,
        ),
    )

    # Initialize phase ratios -------------------------------
    radius       = 0.1
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    init_phases!(phase_ratios, xci, xvi, radius)

     # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes       = StokesArrays(backend_JR, ni)
    pt_stokes    = PTStokesCoeffs(li, di; ϵ = 1e-6,  CFL = 0.5 / √3.1)
    # Buoyancy forces
    ρg           = @zeros(ni...), @zeros(ni...), @zeros(ni...)
    args         = (; T = @zeros(ni...), P = stokes.P, dt = Inf)
    # Rheology
    cutoff_visc = -Inf, Inf
    compute_viscosity!(stokes, phase_ratios, args, rheology, cutoff_visc)

    # Boundary conditions
    flow_bcs     = VelocityBoundaryConditions(;
        free_slip   = (left = true , right = true , top = true , bot = true , back = true , front = true),
        no_slip     = (left = false, right = false, top = false, bot = false, back = false, front = false),
    )

    Vx = PTArray(backend_JR)([ x*εbg for x in xvi[1], _ in 1:ny+2, _ in 1:nz+2])
    Vz = PTArray(backend_JR)([-z*εbg for _ in 1:nx+2, _ in 1:ny+2, z in xvi[3]])
    stokes.V.Vx .= Vx
    stokes.V.Vz .= Vz

    # println("Rank $(igg.me):
    #     extrema Vx = $(extrema(Vx))
    #     extrema Vz = $(extrema(Vz))
    # ")

    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir      = joinpath(figdir, "vtk")
        take(vtk_dir)
    end
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    # global array
    nx_v         = (nx - 2) * igg.dims[1]
    ny_v         = (ny - 2) * igg.dims[2]
    nz_v         = (nz - 2) * igg.dims[3]
    τII_v        = zeros(nx_v, ny_v, nz_v)
    η_vep_v      = zeros(nx_v, ny_v, nz_v)
    εII_v        = zeros(nx_v, ny_v, nz_v)
    phases_c_v   = zeros(nx_v, ny_v, nz_v)
    τII_nohalo   = zeros(nx-2, ny-2, nz-2)
    η_vep_nohalo = zeros(nx-2, ny-2, nz-2)
    εII_nohalo   = zeros(nx-2, ny-2, nz-2)
    phases_c_nohalo = zeros(nx-2, ny-2, nz-2)
    #vertex
    Vxv_v         = zeros(nx_v+1, ny_v+1, nz_v+1)
    Vyv_v         = zeros(nx_v+1, ny_v+1, nz_v+1)
    Vzv_v         = zeros(nx_v+1, ny_v+1, nz_v+1)
    phases_v_v    = zeros(nx_v+1, ny_v+1, nz_v+1)
    Vx_nohalo    = zeros(nx-1, ny-1, nz-1)
    Vy_nohalo    = zeros(nx-1, ny-1, nz-1)
    Vz_nohalo    = zeros(nx-1, ny-1, nz-1)
    phases_v_nohalo = zeros(nx-1, ny-1, nz-1)

    xci_v        = LinRange(0, 1, nx_v), LinRange(0, 1, ny_v), LinRange(0, 1, nz_v)
    xvi_v        = LinRange(0, 1, nx_v+1), LinRange(0, 1, ny_v+1), LinRange(0, 1, nz_v+1)

    local Vx_v, Vy_v, Vz_v
    if do_vtk
        Vx_v = @zeros(ni.+1...)
        Vy_v = @zeros(ni.+1...)
        Vz_v = @zeros(ni.+1...)
    end

    # Time loop
    t, it = 0.0, 0
    tmax  = 4
    τII   = Float64[]
    sol   = Float64[]
    ttot  = Float64[]

    while t < tmax
       # Stokes solver ----------------
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
                iterMax          = 150e3,
                nout             = 1e3,
                viscosity_cutoff = (-Inf, Inf)
            )
        )

        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        push!(τII, maximum(stokes.τ.xx))

        it += 1
        t  += dt

        push!(sol, solution(εbg, t, G0, η0))
        push!(ttot, t)

        igg.me == 0 && println("it = $it; t = $t \n")

        if do_vtk
            velocity2vertex!(Vx_v, Vy_v, Vz_v, @velocity(stokes)...)
            @views Vx_nohalo .= Array(Vx_v[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
            @views Vy_nohalo .= Array(Vy_v[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
            @views Vz_nohalo .= Array(Vz_v[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
            # gather!(Vx_nohalo, Vxv_v)
            # gather!(Vy_nohalo, Vyv_v)
            # gather!(Vz_nohalo, Vzv_v)
        end

        # MPI
        phase_vertex = [argmax(p) for p in Array(phase_ratios.vertex)]
        phase_center = [argmax(p) for p in Array(phase_ratios.center)]

        @views τII_nohalo   .= Array(stokes.τ.II[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        @views η_vep_nohalo .= Array(stokes.viscosity.η_vep[2:end-1, 2:end-1, 2:end-1])       # Copy data to CPU removing the halo
        @views εII_nohalo   .= Array(stokes.ε.II[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        gather!(τII_nohalo, τII_v)
        gather!(η_vep_nohalo, η_vep_v)
        gather!(εII_nohalo, εII_v)

        @views phases_c_nohalo .= Array(phase_center[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        gather!(phases_c_nohalo, phases_c_v)
        # @views phases_v_nohalo .= Array(phase_vertex[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        # gather!(phases_v_nohalo, phases_v_v)

        if igg.me == 0

            if do_vtk == true

                data_v = (;
                    # T = T_v,
                    # phases_v = phases_v_v,
                )
                data_c = (;
                    τII = τII_v,
                    εII = εII_v,
                    η   = η_vep_v,
                    phases = phases_c_v


                )
                velocity_v = (
                    Array(Vxv_v),
                    Array(Vyv_v),
                    Array(Vzv_v),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$(it)_$(igg.me)", 6, "0")),
                    xvi_v,
                    xci_v,
                    data_v,
                    data_c,
                    velocity_v
                )
            end

            # visualisation
            th      = 0:pi/50:3*pi;
            xunit   = @. radius * cos(th) + 0.5;
            yunit   = @. radius * sin(th) + 0.5;
            slice_j = ny_v >>> 1
            fig     = Figure(size = (1600, 1600), title = "t = $t")
            ax1     = Axis(fig[1,1], aspect = 1, title = "τII")
            ax2     = Axis(fig[2,1], aspect = 1, title = "η_vep")
            ax3     = Axis(fig[1,2], aspect = 1, title = "log10(εII)")
            ax4     = Axis(fig[2,2], aspect = 1)
            heatmap!(ax1, xci_v[1], xci_v[3], Array(τII_v[:, slice_j, :]) , colormap=:batlow)
            heatmap!(ax2, xci_v[1], xci_v[3], Array(log10.(η_vep_v)[:, slice_j, :]) , colormap=:batlow)
            heatmap!(ax3, xci_v[1], xci_v[3], Array(log10.(εII_v)[:, slice_j, :]) , colormap=:batlow)
            lines!(ax2, xunit, yunit, color = :black, linewidth = 5)
            lines!(ax4, ttot, τII, color = :black)
            lines!(ax4, ttot, sol, color = :red)
            hidexdecorations!(ax1)
            hidexdecorations!(ax3)
            save(joinpath(figdir, "MPI_3D_$(it).png"), fig)
        end
    end

    return nothing
end

do_vtk = true
n      = 32
nx     = n # ÷ 2
ny     = n # ÷ 2
nz     = n # if only 2 CPU/GPU are used nx = 17 - 2 with N =32
figdir = "ShearBand3D_MPI"
igg    = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, nz; init_MPI = true, select_device=false)...)
else
    igg
end
main(igg; figdir = figdir, nx = nx, ny = ny, nz = nz, do_vtk = do_vtk);
