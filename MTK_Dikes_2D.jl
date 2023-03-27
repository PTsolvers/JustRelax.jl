using Pkg
Pkg.activate(".")
using JustRelax
# using MagmaThermoKinematics
ENV["PS_PACKAGE"] = :Threads     # if GPU use :CUDA

const USE_GPU=false;

if USE_GPU  model = PS_Setup(:gpu, Int64, 2)            # initialize parallel stencil in 2D
            environment!(model)      
else        model = PS_Setup(:cpu, Float64, 2)            # initialize parallel stencil in 2D
            environment!(model)                        
end

using Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions
using ParallelStencil.FiniteDifferences2D   #specify Dimension
using GeophysicalModelGenerator, StencilInterpolations, StaticArrays
using Plots, WriteVTK
using JustPIC                          

# -----------------------------------------------------------------------------------------
# Viscosity calculation functions stored in this script
include("./src/Helperfunc.jl");

## Workaround of MTK/JR environment! conflicts
include("./adm-MTK/Utils.jl");
include("./adm-MTK/Advection.jl");
include("./adm-MTK/Dikes.jl");
println("Loaded helper functions")
#------------------------------------------------------------------------------------------
include("./src/LoadModel.jl");

LonLat                      = load("./Data/ExportToba_2.jld2", "TobaTopography_LATLON")#, "TobaTopography_Cart");
        
proj                        = ProjectionPoint(; Lat=2.19, Lon=98.91);
Topo                        = Convert2CartData(LonLat, proj);

println("Done loading Model... starting Dike Injection 2D routine")
#------------------------------------------------------------------------------------------
@views function DikeInjection_2D();
    figdir = "figs2D"
  # Standard MTK Toba routine--------------------------------------------------------------

   # nondimensionalize with CharDim 
    CharDim    =    GEO_units(length=100km, viscosity=1e20Pa*s)
    
    # dimensions should be multiplication of 32 to be scalable @ GPU
    Nx,Ny,Nz                = 100,100,100
    # Nx +1 bc of conflicts&errors in functions due to different definition of center and verex grid cells
    Grid                    = CreateCartGrid(size=(Nx+1,Ny+1,Nz+1),x=((Topo.x.val[1,1,1])km,(Topo.x.val[end,1,1])km), y=((Topo.y.val[1,1,1])km,(Topo.y.val[1,end,1])km),z=(-40km,4km))
    X,Y,Z                   = XYZGrid(Grid.coord1D...);
    DataTest                = CartData(X,Y,Z,(Depthdata=Z,));

    Lon,Lat,Depth           = (Topo.x.val.*km), (Topo.y.val.*km), ((Topo.z.val./1e3).*km);
    Topo_Cart                = CartData(Lon,Lat,Depth,(Depth=Depth,));

    ind                     = AboveSurface(DataTest,Topo_Cart);
    Phase                   = ones(size(X));
    Phase[ind]             .= 3;

    DataPara                = CartData(X,Y,Z,(Phase=Phase,));
    Phase                   = Int64.(round.(DataPara.fields.Phase));             

# ----------CrossSections----------------------------------

    Data_Cross              = CrossSection(DataPara, dims=(101,101), Interpolate=true,Start=(ustrip(Grid.min[1]),ustrip(Grid.max[2])), End=(ustrip(Grid.max[1]), ustrip(Grid.min[2])))
    x_new                   = FlattenCrossSection(Data_Cross);
    Data_Cross              = AddField(Data_Cross,"FlatCrossSection", x_new);

    #Seismo Model 
    Model3D_Cross           = CrossSection(Model3D_cart,dims=(101,101), Interpolate=true, Start=(ustrip(Grid.min[1]),ustrip(Grid.max[2])), End=(ustrip(Grid.max[1]), ustrip(Grid.min[2])));
    Model3D_new             = FlattenCrossSection(Model3D_Cross);
    Model3D_Cross           = AddField(Model3D_Cross,"Model3D_Cross",Model3D_new)
    
    #New 2D Grid
    Grid2D                  = CreateCartGrid(size=(Nx+1,Ny+1), x=(extrema(Data_Cross.fields.FlatCrossSection).*km),z=(extrema(Data_Cross.z.val).*km), CharDim=CharDim); #create new 2D grid for Injection routine
    Phase                   = dropdims(Data_Cross.fields.Phase,dims=3); #Dropped the 3rd dim of the Phases to be consistent with new grid of the CrossSection
    Phi_melt_data           = dropdims(Model3D_Cross.fields.Phi_melt, dims=3);
    Phase                   = Int64.(round.(Phase));
    #Dike location initiation                                                              
    ind_melt                =  findall(Phi_melt_data.>0.12); # only Meltfraction of 12% used for dike injectio
    # x, z are redefined in ellipse equation of AdM right now... to be updated
    x, z                    = dropdims(Data_Cross.fields.FlatCrossSection,dims=3), dropdims(Data_Cross.z.val,dims=3); # Define x,z coord for injection
    x, z                    = x.*km, z.*km; # add dimension to FlatCrossSection to non-dim x,z for consistency
    x1, z1                    = nondimensionalize(x,CharDim), nondimensionalize(z,CharDim);

#----------Paraview Files--------------------------------------------------------------------
## Save Paraview Data to visualise 
# Write_Paraview(Topo_new, "Topography")
# Write_Paraview(DataPara,"Phase")
# Write_Paraview(Data_Cross,"Cross")
# Write_Paraview(Model3D_Cross,"Model3D_Cross")

#-------JustRelax parameters----------------------------------------------------------------
    η_uppercrust = 1e22
    η_magma = 1e14
    creep_rock = LinearViscous(; η=η_uppercrust*Pa * s)
    creep_magma = LinearViscous(; η=η_magma*Pa * s)
    creep = LinearViscous(;η=1e20*Pa*s)

    # Domain setup for JustRelax 
    ni                      = (Nx,Nz);   #grid spacing for JustRelax calculation 
    lx, lz                  = (Grid2D.L[1]), (Grid2D.L[2]) # nondim if CharDim=CharDim
    li                      = lx, lz
    di                      = Grid2D.Δ  
    xci, xvi                = Grid2D.coord1D_cen, Grid2D.coord1D    #non-dim nodes at the center and the vertices of the cell (staggered grid)

    #---------------------------------------------------------------------------------------
    # Set material parameters                                       
    MatParam                =   (
        SetMaterialParams(Name="UpperCrust", Phase=1, 
                 Density    = ConstantDensity(ρ=2800kg/m^3),      
               HeatCapacity = ConstantHeatCapacity(cp=1050J/kg/K),
               Conductivity = ConstantConductivity(k=1.5Watt/K/m),       
                 LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
                #    CreepLaws= LinearViscous(η=η_uppercrust*Pa * s),
          CompositeRheology = CompositeRheology((creep,)),
                    Melting = MeltingParam_Caricchi(),
                 Elasticity = ConstantElasticity(; G=Inf*Pa, Kb=Inf*Pa),
                   CharDim  = CharDim,), 

        SetMaterialParams(Name="Magma", Phase=2, 
                 Density    = ConstantDensity(ρ=2800kg/m^3),               
               HeatCapacity = ConstantHeatCapacity(cp=1050J/kg/K),
               Conductivity = ConstantConductivity(k=1.5Watt/K/m),       
                 LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
                  #   CreepLaws = LinearViscous(η= η_magma*Pa*s),
          CompositeRheology = CompositeRheology((creep,)),
                    Melting = MeltingParam_Caricchi(),
                   Elasticity = ConstantElasticity(; G=Inf*Pa, Kb=Inf*Pa),
                   CharDim  = CharDim,),                               
                                        

        # SetMaterialParams(Name="Air", Phase=3, 
        #          Density    = ConstantDensity(ρ=2800kg/m^3),               
        #        HeatCapacity = ConstantHeatCapacity(cp=1050J/kg/K),
        #        Conductivity = ConstantConductivity(k=15Watt/K/m),       
        #          LatentHeat = ConstantLatentHeat(Q_L=0.0J/kg),
        #                # Melting = MeltingParam_Caricchi()
        #            CharDim  = CharDim),
                                    )  

                        

    # Physical Parameters 

    ## Multiple Phases defined !!!  ATTENTION MatParam[2] vs MatParam[3]
    ΔT                      =   nondimensionalize(800C, CharDim)
    GeoT                    =   -(ΔT - nondimensionalize(0C, CharDim)) / li[2]
    η                       =   MatParam[2].CompositeRheology[1][1].η.val
    cp                      =   MatParam[2].HeatCapacity[1].cp.val  # heat capacity    
    ρ0                      =   MatParam[2].Density[1].ρ.val        # reference Density
    k0                      =   MatParam[2].Conductivity[1].k.val   # Conductivity of "Air" due to stability reasons
    κ                       =   k0/(ρ0 * cp);                       # thermal diffusivity
    g                       =   MatParam[2].Gravity[1].g.val        # Gravity
    α                       =   0.03
    # α                       =   MatParam[1].Density[1].α.val                     # thermal expansion coefficient for PT Density
    Ra                      =   ρ0 * g * α * ΔT * lz^3 / (η * κ)
    dt                      =   dt_diff = 0.5 / 6.1 * min(di...)^3 / κ # diffusive CFL timestep limiter (AdM 6.1 and ^3)

    # --- AdM ellipse code------------------------------------------------------------------------------------------
    # Set the Phases distribution ------------------------
    # this is not yet adapted to the Toba setup... prerequisit for the Stokes solver is phase_c (center) and phase_v (vertex)

    phase_v   = ones(Int64, ni.+1...)         # constant for now
    phase_c   = ones(Int64, ni...)         # constant for now
    # a_ellipse = ustrip(nondimensionalize(35km, CharDim))
    # b_ellipse = ustrip(nondimensionalize(15km, CharDim))
    # z_c       = ustrip(nondimensionalize(-20km, CharDim))
    for i in CartesianIndices(phase_v)
        x, z = xvi[1][i[1]], xvi[2][i[2]]
        if Phase[i] == 2
            phase_v[i] = 2
        end
    end
    for i in CartesianIndices(phase_c)
        x, z = xci[1][i[1]], xci[2][i[2]]
        if Phase[i] == 2
            phase_c[i] = 2
        end
    end

    #----- thermal Array setup ----------------------------------------------------------------------
    thermal                 = ThermalArrays(ni)
    thermal_bc              = TemperatureBoundaryConditions(; 
                no_flux     = (left = true , right = true , top = false, bot = false), 
                periodicity = (left = false, right = false, top = false, bot = false),
    )
    
    w = 1e-2 * li[2] 
    thermal.T .= PTArray([
      0.5 * ΔT * exp(-(xvi[1][ix] / w)^2 - ((xvi[2][iy] + 0.5 * li[2]) / w)^2) +
      xvi[2][iy] * GeoT +
      nondimensionalize(0C, CharDim) for ix in 1:ni[1]+1, iy in 1:ni[2]+1
      ])
    Tnew_cpu = Matrix{Float64}(undef, (ni.+1)...)

    @views thermal.T[:, 1] .= ΔT;
    ind                     = findall(thermal.T.<=nondimensionalize(0C,CharDim));
    @views thermal.T[ind]  .= nondimensionalize(0C,CharDim);
    # ind                     = findall(Phase.==3);
    # @views thermal.T[ind]  .= nondimensionalize(20C, CharDim);
    @views thermal.T[:, end] .= nondimensionalize(0C, CharDim)
    @copy  thermal.Told thermal.T
    @copy  Tnew_cpu Array(thermal.T)
   
    #------------------------------------------------------------------------------------------------------------
     # Dike parameters 
    W_in, H_in              = nondimensionalize(5km, CharDim),nondimensionalize(0.5km, CharDim); # Width and thickness of dike
    T_in                    = nondimensionalize(900C, CharDim);  
    H_ran, W_ran            = length(z1) ,length(x1) ;# Size of domain in which we randomly place dikes and range of angles
     # H_ran, W_ran            =   length(Model3D_Cross.fields.Model3D_Cross) ,length(Model3D_Cross.z.val);# Size of domain in which we randomly place dikes and range of angles   
    center                  = @. getindex.(xvi,1) + li* 0.5       #to be upscaled to Phi_melt
    dike                    = Dike(; Center=center, W=W_in, H=H_in, DikeType=:ElasticDike, T=T_in)
    nTr_dike                =   300; 

    #Workaround for center of dike according to phi_melt (to be implented in loop)

    # --------------------------------------------------------------------------------
    # Allocate it to GPU
    if USE_GPU; #GPU_Phase   =   CuArray{Int64}(undef,(ni.+1...));
                #Phases      =   GPU_Phase;
                Phases      =   PTArray(Phase)
    else        #CPU_Phase   =   Array{Int64}(undef,(ni.+1...));
                #CPU_Phase  .=   Phase;
                #Phases      =   CPU_Phase;   
                Phases      =   PTArray(Phase)
                   end                        

 # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes          = StokesArrays(ni, ViscoElastic)
    pt_stokes       = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL=1 / √2.1)
    # Rheology
    η               = @ones(ni...)
    args_η          = (; T=thermal.T)
    @parallel (@idx ni) initViscosity!(η, phase_c, MatParam) # init viscosity field
    η_vep           = deepcopy(η)
    dt_elasticity   = Inf

    # to be added in GP...
    ϕ       = similar(η) # melt fraction
    S, mfac = 1.0, -2.8 # factors for hexagons
    # η_f     = MatParam[1].CompositeRheology[1][1].η.val    # melt viscosity
    # η_s     = MatParam[1].CompositeRheology[1][1].η.val    # solid viscosity
    #Multiple Phases defined
    η_f     = MatParam[2].CompositeRheology[1][1].η.val    # melt viscosity
    η_s     = MatParam[1].CompositeRheology[1][1].η.val    # solid viscosity
    args_η  = (; ϕ = ϕ) 
    # args_ϕ  = (; ϕ = ϕ) #copy paste error???
    # Buoyancy forces
    ρg              = @zeros(ni...), @zeros(ni...)
    # Boundary conditions
    flow_bcs = FlowBoundaryConditions(; 
        free_slip   = (left=true, right=true, top=true, bot=true), 
    )
    # ----------------------------------------------------

    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    anim = Animation(figdir, String[])
    # println("Animation directory: $(anim.dir)")
    # ----------------------------------------------------   
    
    parts_semilagrange = SemiLagrangianParticles(xvi)
    Tracers =  Tracer2(2) # Initialize tracers   
    # Tracers, Tnew_cpu, Vol = InjectDike(Tracers, parts_semilagrange, Tnew_cpu, xvi, dike, nTr_dike)   # Add dike, move hostrocks
    
     # Time loop
     t, it = 0.0, 0
     nt    = 20
     local iters                   
 
     while it < nt
 
         # Update buoyancy and viscosity -
         @copy thermal.Told thermal.T
         @parallel (@idx ni) compute_melt_fraction!(ϕ, MatParam, phase_c,  (T=thermal.T,))
         fill!(phase_c, 1)
         fill!(phase_v, 1)
         @parallel computeViscosity!(η, ϕ, S, mfac, η_f, η_s)
         @copy η_vep η
         @parallel (@idx ni) compute_ρg!(ρg[2], MatParam, phase_c, (T=thermal.T, P=stokes.P))
         # ------------------------------
 
         # Stokes solver ----------------
         iters = solve!(
             stokes,
             thermal,
             pt_stokes,
             di,
             flow_bcs,
             ρg,
             η,
             η_vep,
             phase_v,
             phase_c,
             args_η, 
             MatParam, # do a few initial time-steps without plasticity to improve convergence
             dt,
             iterMax=100e3,
             nout=1e3,
         )
         dt = compute_dt(stokes, di, dt_diff)
         # ------------------------------
 
# Inject dike every x timesteps 
        # if floor(time/InjectionInterval)> dike_inj       # Add new dike every X years
        if mod(it, 2) == 0
          #         dike_inj  =     floor(time/InjectionInterval)                                               # Keeps track on what was injected already
          ID        =     rand(ind_melt)
          cen       =     [x1[ID],z1[ID]]  # Randomly vary center of dike 
          if cen[end] < ustrip(nondimensionalize(-25km, CharDim))
              Angle_rand = rand(80.0:0.1:100.0) # Orientation: near-vertical @ depth             
          else
              Angle_rand = rand(-10.0:0.1:10.0) # Orientation: near-vertical @ shallower depth     
          end
          dike = Dike(;
              Angle=[Angle_rand],
              Center=cen,
              W=W_in,
              H=H_in,
              DikeType=:ElasticDike,
              T=T_in,
          ) # "Reference" dike with given thickness,radius and T

          @copy Tnew_cpu Array(thermal.T)
          Tracers, Tnew_cpu,  = InjectDike(Tracers, parts_semilagrange, Tnew_cpu, xvi, dike, nTr_dike)   # Add dike, move hostrocks
        #   _, Tnew_cpu,  = InjectDike(Tracers, parts_semilagrange, Tnew_cpu, xvi, dike, nTr_dike)   # Add dike, move hostrocks
          @parallel assign!(thermal.Told, Tnew_cpu)
          @parallel assign!(thermal.T, Tnew_cpu)
          #        InjectVol +=    Vol                                                                 # Keep track of injected volume
          #        println("Added new dike; total injected magma volume = $(round(InjectVol/km³,digits=2)) km³; rate Q=$(round(InjectVol/(time),digits=2)) m³/s")
          println("injected dike")
        end
        # Thermal solver ---------------
        solve!(
            thermal,
            thermal_bc,
            stokes,
            phase_v,
            MatParam,
            (; P=stokes.P, T=thermal.T),
            di,
            dt
        )
        # ------------------------------

        @show it += 1
        t += dt

        # Plotting ---------------------
        # if it == 1 || rem(it, 1) == 0
        #     fig = Figure(resolution = (900, 1600), title = "t = $t")
        #     ax1 = Axis(fig[1,1], aspect = ar, title = "T")
        #     ax2 = Axis(fig[2,1], aspect = ar, title = "Vy")
        #     ax3 = Axis(fig[3,1], aspect = ar, title = "τII")
        #     ax4 = Axis(fig[4,1], aspect = ar, title = "η")        
        #     h1 = heatmap!(ax1, xvi[1], xvi[2], Array(thermal.T) , colormap=:batlow)
        #     h2 = heatmap!(ax2, xci[1], xvi[2], Array(stokes.V.Vy[2:end-1,:]) , colormap=:batlow)
        #     h3 = heatmap!(ax3, xci[1], xci[2], Array(stokes.τ.II) , colormap=:romaO) 
        #     h4 = heatmap!(ax4, xci[1], xci[2], Array(log10.(η)) , colormap=:batlow)
        #     Colorbar(fig[1,2], h1)
        #     Colorbar(fig[2,2], h2)
        #     Colorbar(fig[3,2], h3)
        #     Colorbar(fig[4,2], h4)
        #     fig
        #     save( joinpath(figdir, "$(it).png"), fig)
        # end
        
        # Visualization
        if it == 1 || rem(it, 1) == 0
            Vy_c = (stokes.V.Vy[:,2:end] + stokes.V.Vy[:,1:end-1])/2

            x_v = ustrip.(dimensionalize(xvi[1], km, CharDim))
            y_v = ustrip.(dimensionalize(xvi[2], km, CharDim))
            x_c = ustrip.(dimensionalize(xci[1], km, CharDim))
            y_c = ustrip.(dimensionalize(xci[2], km, CharDim))
            T_d = ustrip.(dimensionalize(Array(thermal.T), C, CharDim))
            η_d = ustrip.(dimensionalize(Array(η), Pas, CharDim))
            Vy_d= ustrip.(dimensionalize(Array(Vy_c),   cm/yr, CharDim));
            ρg_d= ustrip.(dimensionalize(Array(ρg[2]),   kg/m^3*m/s^2, CharDim));
            ρ_d = ρg_d/10;
            t_Myrs = dimensionalize(t, Myrs, CharDim)

            p1 = Plots.heatmap(
                x_v,
                y_v,
                T_d';
                aspect_ratio=2,
                xlims = extrema(x_v),
                ylims = extrema(y_v),
                zlims=(0, 900),
                c=:batlow,
                title="time=$(round(ustrip.(t_Myrs), digits=3)) Myrs",
                titlefontsize = 20,
                colorbar_title = "\nT [C]",
                colorbar_titlefontsize = 12,
            )

            # p2 = Plots.heatmap(
            #     x_c,
            #     y_v,
            #     (@view stokes.V.Vy[2:end-1,:])';
            #     xlims = extrema(x_v),
            #     ylims = extrema(y_v),
            #     aspect_ratio=2,
            #     c=:batlow,
            #     xlabel="width [km]",
            #     colorbar_title = "\n\n\nVy",
            #     colorbar_titlefontsize = 12,
            # )
            p2 = Plots.heatmap(
                x_v,
                y_v,
                log10.(η_d');
                aspect_ratio=1,
                xlims=(minimum(x_v), maximum(x_v)),
                ylims=(minimum(y_v), maximum(y_v)),
                c=:oleron,
                colorbar_title="log10(η [Pas])",
                colorbar_titlefontsize = 12,
                xlabel="width [km]",
            )
            # p3 = Plots.heatmap(x_v, y_v, Vy_d', aspect_ratio=1, xlims=(minimum(x_v), maximum(x_v)), ylims=(minimum(y_v), maximum(y_v)), c=:jet,
            #                         title="Vy [cm/yr]", xlabel="width [km]")

            # p4 = Plots.heatmap(x_v, y_v, ρ_d', aspect_ratio=1, xlims=(minimum(x_v), maximum(x_v)), ylims=(minimum(y_v), maximum(y_v)), c=:jet,
            #                         title="ρ [kg/m³]", xlabel="width [km]")

            Plots.plot(p1, p2 #=, p3, p4=#, layout=(2, 1), size=(1200,1200))

            frame(anim)

            # display( quiver!(Xp[:], Yp[:], quiver=(Vxp[:]*Vscale, Vyp[:]*Vscale), lw=0.1, c=:blue) )
        end
        # ------------------------------

    end

    # return (ni=ni, xci=xci, li=li, di=di), thermal
end

# figdir = "figs2D"
# ni, xci, li, di, figdir ,thermal = DikeInjection_2D();
DikeInjection_2D();