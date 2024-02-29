using LazyGrids, GeoParams
using GLMakie
import GeoParams: Dislocation, Diffusion

struct Thermal{A, B}
    T::A
    Tpot::B
    dTadi::B
    T0::B # Tini::A
    q::A

    function Thermal(ni,Tpot,dTadi,z)
        T           =   zeros(ni)
        Tpot        =   Tpot + 273.15   
        dTadi       =   dTadi
        T0          =   273.15                          # Surface temperature [ K ]
        T           =   @. Tpot + abs.(z/1.0e3)*dTadi   # Initial T-profile [ K ]
        T[1]        =   T0
        q           =   zeros(ni-1)
        new{typeof(T),typeof(Tpot)}(T,Tpot,dTadi,T0,q)  
    end
end

struct Thermal_bc{A,B}
    ubound::A
    utbf::B
    lbound::A
    ltbf::B
    function Thermal_bc(ubound,utbf,lbound,ltbf)
        ubound   =   ubound
        utbf     =   utbf
        lbound   =   lbound
        ltbf     =   ltbf
        new{typeof(ubound),typeof(utbf)}(ubound,utbf,lbound,ltbf)
    end
end

struct Thermal_parameters{A}
    ρ::A
    cp::A
    k::A
    ρcp::A
    H::A
    function Thermal_parameters(ni)
        ρ   =   zeros(ni)
        cp  =   zeros(ni)
        k   =   zeros(ni)
        ρcp =   zeros(ni)
        H   =   zeros(ni)
        new{typeof(ρ)}(ρ,cp,k,ρcp,H)
    end
end

function SolveDiff1Dexplicit_vary!(    
    thermal,
    thermal_parameters,
    thermal_bc,
    rheology,
    phase,
    di,    
    dt,
    args
)    
    nx      =   length(thermal.T)
    T0      =   thermal.T

    compute_density!(thermal_parameters.ρ,rheology,phase,args)
    compute_heatcapacity!(thermal_parameters.cp,rheology,phase,args)
    compute_conductivity!(thermal_parameters.k,rheology,phase,args)
    thermal_parameters.ρcp  .=   @. thermal_parameters.cp * thermal_parameters.ρ
    compute_radioactive_heat!(thermal_parameters.H,rheology,phase,args)

    if thermal_bc.ubound == "const"
        thermal.T[1]    =   T0[1]
    elseif thermal_bc.ubound == "flux"
        kB      =   (thermal_parameters.k[2] + thermal_parameters.k[1])/2.0
        kA      =   (thermal_parameters.k[1] + thermal_parameters.k[1])/2.0
        a       =   (dt*(kA + kB)) / (di^2.0 * thermal_parameters.ρcp[1])
        b       =   1 - (dt*(kA + kB)) / (di^2.0 * thermal_parameters.ρcp[1])
        #c       =   -(kA*dt*2.0*thermal_bc.utbf)/(di * thermal_parameters.ρcp[1])
        c       =   (dt*2.0*thermal_bc.utbf)/(di * thermal_parameters.ρcp[1])
        thermal.T[1]    =   a*T0[2] + b*T0[1] + c + 
                thermal_parameters.H[1]*dt/thermal_parameters.ρcp[1]
    end
    if thermal_bc.lbound == "const"
        thermal.T[nx]   =   T0[nx]
    elseif thermal_bc.lbound == "flux"
        kB      =   (thermal_parameters.k[nx] + thermal_parameters.k[nx])/2.0
        kA      =   (thermal_parameters.k[nx] + thermal_parameters.k[nx-1])/2.0
        a       =   (dt*(kA + kB)) / (di^2.0 * thermal_parameters.ρcp[nx])
        b       =   1 - (dt*(kA + kB)) / (di^2.0 * thermal_parameters.ρcp[nx])
        #c       =   (kB*dt*2.0*thermal_bc.ltbf) / (di * thermal_parameters.ρcp[nx])
        c       =   -(dt*2.0*thermal_bc.ltbf) / (di * thermal_parameters.ρcp[nx])
        thermal.T[nx]   =   a*T0[nx-1] + b*T0[nx] + c
    end

    kAi     =   @. (thermal_parameters.k[1:end-2] + thermal_parameters.k[2:end-1])/2.0
    kBi     =   @. (thermal_parameters.k[2:end-1] + thermal_parameters.k[3:end])/2.0
    ai      =   @. (kBi*dt)/(di^2.0*thermal_parameters.ρcp[2:end-1])
    bi      =   @. 1.0 - (dt*(kAi + kBi))/(di^2.0*thermal_parameters.ρcp[2:end-1])
    ci      =   @. (kAi*dt)/(di^2.0*thermal_parameters.ρcp[2:end-1])
    thermal.T[2:end-1]  =   @. ai*T0[3:end] + bi*T0[2:end-1] + ci*T0[1:end-2] + 
                    thermal_parameters.H[2:end-1]*dt/thermal_parameters.ρcp[2:end-1]
    return thermal, thermal_parameters
end

function CalcSurfaceHeatFlow1D(thermal,thermal_parameters,di;)
    nx      =   length(thermal.T)

    for j=1:nx-1
        thermal.q[j] = -(thermal_parameters.k[j+1] + thermal_parameters.k[j])/2.0 *
             (thermal.T[j+1] - thermal.T[j])/di
    end
    thermal.q[1]    =   - thermal_parameters.k[1]*(thermal.T[2]-thermal.T[1])/di
    thermal.q[nx-1] =   - thermal_parameters.k[nx]*(thermal.T[nx]-thermal.T[nx-1])/di
    
    return thermal
end

function OceanicGeotherm1D(
    ;nx=201,            # Number of grid points
    lx=200e3,           # Hight of the model [ m ]
    phase=ones(Int64,nx),
    rheology=example_OLrheology(),
    dtfac=0.9,          # Diffusion stability criterion
    age=60.0,           # Lithosphere age [ Ma ]
    Tpot=1315.0,        # Potential temperautre [ C ]
    dTadi=0.5,          # Adiabatic temperature gradient [ K/km ]
    ubound="const",     # Upper thermal boundary condition
    lbound="const",     # Lower thermal boundary condition
    utbf=90.0e-3,       # q [W/m^2]
    ltbf=10.0e-3,       # q [W/m^2]
    plotparam=1
)
    # Function to calculate the 1D geotherm for an oceanic lithosphere.   #
    # Temperature is calculated by solving the 1-D heat equation assuming #
    # variable thermal parameters and a radiogenic heat source.           #
    # The equation is solved using a proper conserving finite difference  #
    # scheme.                                                             #
    # The vertical axis is pointing downwards in negative direction.      #
    # --------------------------------------------------------------------#
    #    LF - 09.13.2023 -                                                #
    # =================================================================== #
    # Constants ========================================================= #
    di      =   -lx / (nx -1)                    # Grid resolution
    zvi     =   LinRange(0,-lx,nx)
    zci     =   LinRange(0 + di/2.0,-lx - di/2.0,nx-1)
    # phase   =   ones(Int64,nx)    
    ## Setup initial thermal structures ================================= #
    thermal     =   Thermal(nx,Tpot,dTadi,zvi)
    thermal_bc  =   Thermal_bc(ubound,utbf,lbound,ltbf)  

    args                    =   (;)    
    thermal_parameters      =   Thermal_parameters(nx)

    compute_density!(thermal_parameters.ρ,rheology,phase,args)
    compute_heatcapacity!(thermal_parameters.cp,rheology,phase,args)
    compute_conductivity!(thermal_parameters.k,rheology,phase,args)
    thermal_parameters.ρcp  .=   @. thermal_parameters.cp * thermal_parameters.ρ
    compute_radioactive_heat!(thermal_parameters.H,rheology,phase,args)

    # Thermal diffusivity [ m^2/s ]
    κ       =   maximum(thermal_parameters.k) / 
        minimum(thermal_parameters.ρ) / minimum(thermal_parameters.cp)
    # =================================================================== #
    ## Time stability criterion ========================================= #
    tfac    =   60.0*60.0*24.0*365.25   # Seconds per year
    age     =   age*1.0e6*tfac          # Age in seconds
    dtexp   =   di^2.0/2.0/κ            # Stability criterion for explicit
    dt      =   dtfac*dtexp
    nit     =   Int(ceil(age/dt))       # Number of iterations
    time    =   zeros(nit)              # Time array
    # =================================================================== #
    ## Calculate 1-D temperature profile ================================ #
    for i = 1:nit        
        if i > 1
            time[i]   =   time[i-1] + dt
        end
        # println("Iteration: ",i, " ; time: ",time[i]/tfac/1e6)
        SolveDiff1Dexplicit_vary!(
            thermal,
            thermal_parameters,
            thermal_bc,
            rheology,
            phase,
            di,
            dt,
            args)
    end    
    # =================================================================== #
    ## Calculate heaf flow ============================================== #
    #=for j=1:nx-1
       thermal.q[j] = -(thermal_parameters.k[j+1] + thermal_parameters.k[j])/2.0 *
            (thermal.T[j+1] - thermal.T[j])/di
    end
    thermal.q[1]    =   - thermal_parameters.k[1]*(thermal.T[2]-thermal.T[1])/di
    thermal.q[nx-1] =   - thermal_parameters.k[nx]*(thermal.T[nx]-thermal.T[nx-1])/di=#
    thermal      =  CalcSurfaceHeatFlow1D(thermal,thermal_parameters,di)
    ## ================================================================== #
    if plotparam == 1
        PlotTempInfo(thermal,zvi,zci,lx)
    end
    # return thermal, thermal_parameters  
end

function ContinentalGeotherm1D(
    ;nx=201,            # Number of grid points
    lx=200e3,           # Hight of the model [ m ]    
    zUC=10e3,           # Depth of the upper crust [ m ]
    zLC=35e3,           # Depth of the lower crust [ m ]
    zvi=LinRange(0,-lx,nx),
    rheology=example_CLrheology(),
    phase=example_CLphase(zUC,zLC,zvi,nx),
    dtfac=0.9,          # Diffusion stability criterion
    age=1000.0,         # Lithosphere age [ Ma ]
    Tpot=1315.0,        # Potential temperautre [ C ]
    dTadi=0.5,          # Adiabatic temperature gradient [ K/km ]
    ubound="const",     # Upper thermal boundary condition
    lbound="const",     # Lower thermal boundary condition
    utbf=50.0e-3,       # q [W/m^2]
    ltbf=10.0e-3,       # q [W/m^2]
    plotparam=1
)
    # Function to calculate the 1D geotherm for a continental lithosphere.#
    # Temperature is calculated by solving the 1-D heat equation assuming #
    # variable thermal parameters and a radiogenic heat source.           #
    # The equation is solved using a proper conserving finite difference  #
    # scheme.                                                             #
    # The vertical axis is pointing downwards in negative direction.      #
    # --------------------------------------------------------------------#
    #    LF - 09.13.2023 -                                                #
    # =================================================================== #
    ## ================================================================== #
    # Constants --------------------------------------------------------- #
    di          =   - lx / (nx -1 )             # Grid resolution
    zci         =   LinRange(0 + di/2.0,-lx - di/2.0,nx-1)    
    ## Setup initial thermal structures ================================= #
    thermal     =   Thermal(nx,Tpot,dTadi,zvi)
    thermal_bc  =   Thermal_bc(ubound,utbf,lbound,ltbf)  

    args                    =   (;)    
    thermal_parameters      =   Thermal_parameters(nx)    

    compute_density!(thermal_parameters.ρ,rheology,phase,args)
    compute_heatcapacity!(thermal_parameters.cp,rheology,phase,args)
    compute_conductivity!(thermal_parameters.k,rheology,phase,args)
    thermal_parameters.ρcp  .=   @. thermal_parameters.cp * thermal_parameters.ρ
    compute_radioactive_heat!(thermal_parameters.H,rheology,phase,args)

    # Thermal diffusivity [ m^2/s ]
    κ       =   maximum(thermal_parameters.k) / 
        minimum(thermal_parameters.ρ) / minimum(thermal_parameters.cp)
    # =================================================================== #
    ## Time stability criterion ========================================= #
    tfac    =   60.0*60.0*24.0*365.25   # Seconds per year
    age     =   age*1.0e6*tfac          # Age in seconds
    dtexp   =   di^2.0/2.0/κ            # Stability criterion for explicit
    dt      =   dtfac*dtexp
    nit     =   Int(ceil(age/dt))       # Number of iterations
    time    =   zeros(nit)              # Time array
    # =================================================================== #
    ## Calculate 1-D temperature profile ================================ #
    for i = 1:nit
        if i > 1
            time[i]   =   time[i-1] + dt
        end
        # println("Iteration: ",i, " ; time: ",time[i]/tfac/1e6)
        SolveDiff1Dexplicit_vary!(  
            thermal,
            thermal_parameters,
            thermal_bc,
            rheology,
            phase,
            di,
            dt,
            args)
    end
    # =================================================================== #
    ## Calculate heaf flow ============================================== #
    #=for j=1:nx-1
        thermal.q[j] = -(thermal_parameters.k[j+1] + thermal_parameters.k[j])/2.0 *
             (thermal.T[j+1] - thermal.T[j])/di
     end
     thermal.q[1]    =   - thermal_parameters.k[1]*(thermal.T[2]-thermal.T[1])/di
     thermal.q[nx-1] =   - thermal_parameters.k[nx]*(thermal.T[nx]-thermal.T[nx-1])/di=#
     thermal      =  CalcSurfaceHeatFlow1D(thermal,thermal_parameters,di)
    ## ================================================================== #

    if plotparam == 1
        PlotTempInfo(thermal,zvi,zci,lx)
    end
    return thermal, thermal_parameters
end

function Geotherm1DExlicit(
    thermal,thermal_bc,thermal_parameters,rheology,phase,di,args
    ;    
    dtfac=0.9,          # Diffusion stability criterion
    age=1000.0,         # Lithosphere age [ Ma ]
)
    # Function to calculate the 1D geotherm for a continental lithosphere.#
    # Temperature is calculated by solving the 1-D heat equation assuming #
    # variable thermal parameters and a radiogenic heat source.           #
    # The equation is solved using a proper conserving finite difference  #
    # scheme.                                                             #
    # The vertical axis is pointing downwards in negative direction.      #
    # --------------------------------------------------------------------#
    #    LF - 09.13.2023 -                                                #
    # =================================================================== #
    @show thermal_parameters.ρ
    @show phase
    ## Update thermal parameters ======================================== #
    compute_density!(thermal_parameters.ρ,rheology,phase,args)
    compute_heatcapacity!(thermal_parameters.cp,rheology,phase,args)
    compute_conductivity!(thermal_parameters.k,rheology,phase,args)
    thermal_parameters.ρcp  .=   @. thermal_parameters.cp * thermal_parameters.ρ
    compute_radioactive_heat!(thermal_parameters.H,rheology,phase,args)

    # Thermal diffusivity [ m^2/s ]
    κ       =   maximum(thermal_parameters.k) / 
        minimum(thermal_parameters.ρ) / minimum(thermal_parameters.cp)
    # =================================================================== #
    ## Time stability criterion ========================================= #
    tfac    =   60.0*60.0*24.0*365.25   # Seconds per year
    age     =   age*1.0e6*tfac          # Age in seconds
    dtexp   =   di^2.0/2.0/κ            # Stability criterion for explicit
    dt      =   dtfac*dtexp
    nit     =   Int(ceil(age/dt))       # Number of iterations
    time    =   zeros(nit)              # Time array
    # =================================================================== #
    ## Calculate 1-D temperature profile ================================ #
    for i = 1:nit
        if i > 1
            time[i]   =   time[i-1] + dt
        end
        # println("Iteration: ",i, " ; time: ",time[i]/tfac/1e6)
        SolveDiff1Dexplicit_vary!(
            thermal,
            thermal_parameters,
            thermal_bc,
            rheology,
            phase,
            di,
            dt,
            args)
    end
    # =================================================================== #
    ## Calculate heaf flow ============================================== #    
    thermal      =  CalcSurfaceHeatFlow1D(thermal,thermal_parameters,di)
    ## ================================================================== #    
    return thermal, thermal_parameters
end

function example_CLphase(zUC,zLC,zvi,nx;
    )
    UC          =   zvi .>= -zUC                # upper crust indices
    LC          =   -zUC .> zvi .>= -zLC        # Lower crust indices
    phase       =   3 .*ones(Int64,nx)
    phase[UC]   .=  1
    phase[LC]   .=  2
    return phase
end

function example_CLrheology(;    
    ρM=3.0e3,           # Density [ kg/m^3 ]
    cpM=1.0e3,          # Specific heat capacity [ J/kg/K ]
    kM=2.3,             # Thermal conductivity [ W/m/K ]
    HM=0.0,             # Radiogenic heat source per mass [H] = W/kg; [H] = [Q/rho]
    ρUC=2.7e3,          # Density [ kg/m^3 ]
    cpUC=1.0e3,         # Specific heat capacity [ J/kg/K ]
    kUC=3.0,            # Thermal conductivity [ W/m/K ]
    HUC=617.0e-12,      # Radiogenic heat source per mass [H] = W/kg; [H] = [Q/rho]
    ρLC=2.9e3,          # Density [ kg/m^3 ]
    cpLC=1.0e3,         # Specific heat capacity [ J/kg/K ]
    kLC=2.0,            # Thermal conductivity [ W/m/K ]
    HLC=43.0e-12,       # Radiogenic heat source per mass [H] = W/kg; [H] = [Q/rho]
)    

    rheology = (
        # Name              = "UpperCrust",
        SetMaterialParams(;
            Phase               = 1,
            #Density           = PT_Density(; ρ0=ρUC, β=0.0, T0=0.0, α=0.0),
            Density             =   ConstantDensity(; ρ=ρUC),
            HeatCapacity        =   ConstantHeatCapacity(; cp=cpUC),
            Conductivity        =   ConstantConductivity(; k=kUC),
            RadioactiveHeat     =   ConstantRadioactiveHeat(; H_r=HUC*ρUC),     # [H] = W/m^3
            CompositeRheology   =   SetDislocationCreep(Dislocation.granite_Tirel_2008),
            Plasticity          =   DruckerPrager(ϕ=15.0, C=20MPa),
        ),
        # Name              = "LowerCrust",
        SetMaterialParams(;
            Phase               =   2,
            #Density         = PT_Density(; ρ0=ρLC, β=0.0, T0=0.0, α=0.0),
            Density             =   ConstantDensity(; ρ=ρLC),
            HeatCapacity        =   ConstantHeatCapacity(; cp=cpLC),
            Conductivity        =   ConstantConductivity(; k=kLC),
            RadioactiveHeat     =   ConstantRadioactiveHeat(; H_r=HLC*ρLC),     # [H] = W/m^3
            CompositeRheology   =   SetDislocationCreep(Dislocation.diabase_Caristan_1982),
            Plasticity          =   DruckerPrager(ϕ=15.0, C=20MPa),
        ),
        # Name              = "LithosphericMantle",
        SetMaterialParams(;
            Phase               =   3,
            #Density           = PT_Density(; ρ0=ρM, β=0.0, T0=0.0, α=0.0),
            Density             =   ConstantDensity(; ρ=ρM),
            HeatCapacity        =   ConstantHeatCapacity(; cp=cpM),
            Conductivity        =   ConstantConductivity(; k=kM),
            RadioactiveHeat     =   ConstantRadioactiveHeat(; H_r=HM*ρM),       # [H] = W/m^3
            CompositeRheology   =   CompositeRheology(
                                        SetDiffusionCreep(Diffusion.dry_olivine_Hirth_2003),
                                        SetDislocationCreep(Dislocation.dry_olivine_Hirth_2003)),
            Plasticity          =   DruckerPrager(ϕ=15.0, C=20MPa),
        ),
    )
    return rheology
end

function example_OLrheology(;    
    ρ=3.0e3,            # Density [ kg/m^3 ]
    cp=1.0e3,           # Specific heat capacity [ J/kg/K ]
    k=3.0,              # Thermal conductivity [ W/m/K ]
)
    rheology = (
        # Name              = "UpperCrust",
        SetMaterialParams(;
            Phase               =   1,
            #Density           = PT_Density(;ρ0=ρ, β=0.0, T0=0.0, α=0.0),
            Density             =   ConstantDensity(; ρ=ρ),
            HeatCapacity        =   ConstantHeatCapacity(;cp=cp),
            Conductivity        =   ConstantConductivity(;k=k),
            RadioactiveHeat     =   ConstantRadioactiveHeat(; H_r=0.0),
            CompositeRheology   =   CompositeRheology(
                                        SetDiffusionCreep(Diffusion.dry_olivine_Hirth_2003),
                                        SetDislocationCreep(Dislocation.dry_olivine_Hirth_2003)),
            Plasticity          =   DruckerPrager(ϕ=15.0, C=20MPa),
        ),
    )  
    return rheology  
end

function PlotStressEnvelope(thermal,τ,zvi,zci,lx;
    linestyle=:solid,
    linewidth=1,
    color=nothing,
    label=nothing,
    title="",
    fig=nothing,
    filename=nothing,
    res=(1200,900),
    legendsize=15,
    labelsize=35,
)

    if isnothing(fig)
        fig     =   Figure(; fontsize=25, resolution=res)
    end

    ax1 = Axis(fig[1,1],
        title  = "Temperature",
        xlabel = "T [ K ]",
        ylabel = "z [ km ]",
        xlabelsize=labelsize,
        ylabelsize=labelsize,
    )
    ax2 = Axis(fig[1,2],
        title  = "Heat Flux",
        xlabel = "q [ mW/m^2 ]",
        ylabel = "z [ km ]",
        xlabelsize=labelsize,
        ylabelsize=labelsize,
    )
    ax3 = Axis(fig[1,3],
        title  = "Stress",
        xlabel = "τ_{II} [ MPa ]",
        ylabel = "z [ km ]",
        xlabelsize=labelsize,
        ylabelsize=labelsize,
    )
    
    lines!(ax1,thermal.T,zvi./1e3)
    ylims!(ax1, [-lx/1e3, 0.0])
    xlims!(ax1,[thermal.T0 maximum(thermal.T)])
    lines!(ax2,thermal.q*1e3,zci./1e3)
    ylims!(ax2,[-lx/1e3, 0.0])
    xlims!(ax2,[0 maximum(thermal.q*1e3)])
    lines!(ax3,τ./1e6,zvi./1e3)
    ylims!(ax3,[-lx/1e3, 0.0])
    xlims!(ax3,[0 maximum(τ./1e6)])

    #axislegend(ax1; labelsize=legendsize)
    #axislegend(ax2; labelsize=legendsize)

    if !isnothing(filename)
        save(filename,fig)
    else
        display(fig)
    end
    
    return fig
end

function solveStress(rheology, phase, ε, P, T)
    # solve for stress
    nz = length(T)
    τ  = zeros(Float64, nz)
    for i = 1 : nz
        Pres = P[i]
        Temp = T[i]
        args = (T=Temp, P=Pres)
        Mat  = rheology[phase[i]]
        τ[i] = compute_τII(Mat.CompositeRheology[1], ε, args)

        F    = compute_yieldfunction(Mat.Plasticity[1], P=Pres, τII=τ[i])
        if F > 0
            c = Mat.Plasticity[1].C.val
            τ[i] = Pres * Mat.Plasticity[1].sinϕ.val + Mat.Plasticity[1].cosϕ.val * c
        end
    end
    return τ
end

function main(
    ;plotparam=0,
)
    type        =   "continental"
    # Constants ========================================================= #
    lx, nx      =   200e3, 201                  # Layer thickness, resolution
    di          =   - lx / (nx -1 )             # Grid resolution
    @show di
    zvi         =   LinRange(0.0,-lx,nx)
    zci         =   LinRange(0 + di/2.0,-lx - di/2.0,nx-1)    
    g           =   9.81                        # Gravitational acceleration
    ε           =   1e-15                       # Background strain-rate
    # Define crustal structure ------------------------------------------ # 
    zUC         =   20e3                        # Depth of the upper crust [ m ]
    zLC         =   35e3                        # Depth of the lower crust [ m ]
    UC          =   zvi .>= -zUC                # upper crust indices
    LC          =   -zUC .> zvi .>= -zLC        # Lower crust indices    
    # Time settings ----------------------------------------------------- #
    dtfac       =   0.9                         # Diffusion stability criterion
    age         =   120.0                       # Lithosphere age [ Ma ]    
    # Thermal boundary conditions ---------------------------------------- #    
    Tpot        =   1315.0                      # Potential temperautre [ C ]
    dTadi       =   0.5                         # Adiabatic temperature gradient [ K/km ]
    ubound      =   "const"                     # Upper thermal boundary condition
    lbound      =   "const"                     # Lower thermal boundary condition
    utbf        =   50.0e-3                     # q [W/m^2]
    ltbf        =   10.0e-3                     # q [W/m^2]
    thermal_bc  =   Thermal_bc(ubound,utbf,lbound,ltbf) 
    thermal     =   Thermal(nx,Tpot,dTadi,zvi)
    args                    =   (;)    
    thermal_parameters      =   Thermal_parameters(nx)
    # Rheology setting --------------------------------------------------- #
    if type == "continental"
        rheology    =   example_CLrheology() 
        phase       =   3 .*ones(Int64,nx)
        phase[UC]   .=  1
        phase[LC]   .=  2
    elseif type == "oceanic"
        rheology    =   example_OLrheology()
        phase       =   ones(Int64,nx)
    end

    thermal,thermal_parameters = Geotherm1DExlicit(
        thermal,
        thermal_bc,
        thermal_parameters,
        rheology,
        phase,
        di,
        args;
        dtfac,
        age,
    )

    P   =   LithPres(rheology, phase, thermal_parameters.ρ, thermal.T, -di, g)
    τII =   solveStress(rheology, phase, ε, P, thermal.T)
    
    #args=   (T=thermal.T,P=P)
    #η   =   compute_viscosity_τII(rheology, τII, args)

    PlotStressEnvelope(thermal,τII,zvi,zci,lx)
end

main()

# OceanicGeotherm1D()

# ContinentalGeotherm1D()

