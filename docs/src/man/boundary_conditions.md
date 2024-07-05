# Flow boundary conditions

Supported boundary conditions:

1. Free slip

    $\frac{\partial u_i}{\partial x_i} = 0$ at the boundary $\Gamma$

2. No slip

    $u_i = 0$ at the boundary $\Gamma$

3. Free surface

    $\sigma_z = 0 \rightarrow \tau_z = P$ at the top boundary

## Defining the boundary conditions
We have two ways of defining the boundary condition formulations:
    - `VelocityBoundaryConditions`, and
    - `DisplacementBoundaryConditions`.
The first one is used for the velocity-pressure formulation, and the second one is used for the displacement-pressure formulation. The flow boundary conditions can be switched on and off by setting them as `true` or `false` at the appropriate boundaries. Valid boundary names are `left` and `right`, `top` and `bot`, and for the 3D case, `front` and `back`.


For example, if we want to have free free-slip in every single boundary in a 2D simulation, we need to instantiate `VelocityBoundaryConditions` or `DisplacementBoundaryConditions` as:
```julia
bcs = VelocityBoundaryConditions(;
    no_slip      = (left=false, right=false, top=false, bot=false),
    free_slip    = (left=true, right=true, top=true, bot=true),
    free_surface = false
)
bcs = DisplacementBoundaryConditions(;
    no_slip      = (left=false, right=false, top=false, bot=false),
    free_slip    = (left=true, right=true, top=true, bot=true),
    free_surface = false
)
```

The equivalent for the 3D case would be:
```julia
bcs = VelocityBoundaryConditions(;
    no_slip      = (left=false, right=false, top=false, bot=false, front=false, back=false),
    free_slip    = (left=true, right=true, top=true, bot=true, front=true, back=true),
    free_surface = false
)
bcs = DisplacementBoundaryConditions(;
    no_slip      = (left=false, right=false, top=false, bot=false, front=false, back=false),
    free_slip    = (left=true, right=true, top=true, bot=true, front=true, back=true),
    free_surface = false
)
```
## Prescribing the velocity/displacement boundary conditions
Normally, one would prescribe the velocity/displacement boundary conditions by setting the velocity/displacement field at the boundary through the application of a background strain rate `εbg`.
Depending on the formulation, the velocity/displacement field is set as follows for the 2D case:
### Velocity formulation
```julia
stokes.V.Vx .= PTArray(backend)([ x*εbg for x in xvi[1], _ in 1:ny+2]) # Velocity in x direction
stokes.V.Vy .= PTArray(backend)([-y*εbg for _ in 1:nx+2, y in xvi[2]]) # Velocity in y direction
```
Make sure to apply the set velocity to the boundary conditions. You do this by calling the `flow_bcs!` function,
```julia
flow_bcs!(stokes, flow_bcs)
```
and then applying the velocities to the halo
```julia
update_halo!(@velocity(stokes)...)
```
### Displacement formulation
```julia
stokes.U.Ux .= PTArray(backend)([ x*εbg*lx*dt for x in xvi[1], _ in 1:ny+2]) # Displacement in x direction
stokes.U.Uy .= PTArray(backend)([-y*εbg*ly*dt for _ in 1:nx+2, y in xvi[2]]) # Displacement in y direction
flow_bcs!(stokes, flow_bcs)
```
Make sure to initialize the displacement according to the extent of your domain. Here, lx and ly are the domain lengths in the x and y directions, respectively.
Also for the displacement formulation it is important that the displacement is converted to velocity before updating the halo. This can be done by calling the `displacement2velocity!` function.
```julia
displacement2velocity!(stokes, dt) # convert displacement to velocity
update_halo!(@velocity(stokes)...)
```
