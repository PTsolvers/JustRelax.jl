# Flow boundary conditions

Supported boundary conditions:

1. Free slip

    $\frac{\partial u_i}{\partial x_i} = 0$ at the boundary $\Gamma$

2. No slip

    $u_i = 0$ at the boundary $\Gamma$

3. Free surface

    $\sigma_z = 0 \rightarrow \tau_z = P$ at the top boundary

## Defining the boundary contions
Information regarding flow boundary conditions is defined in the `FlowBoundaryConditions` object. They can be switched on and off by setting them as `true` or `false` at the appropriate boundaries. Valid boundary names are `left` and `right`, `top` and `bot`, and for the 3D case, `front` and `back`. 

For example, if we want to have free free-slip in every single boundary in a 2D simulation, we need to instantiate `FlowBoundaryConditions` as:
```julia
bcs = FlowBoundaryConditions(;
    no_slip      = (left=false, right=false, top=false, bot=false),
    free_slip    = (left=true, right=true, top=true, bot=true),
    free_surface = false
)
```

The equivalent for the 3D case would be:
```julia
bcs = FlowBoundaryConditions(;
    no_slip      = (left=false, right=false, top=false, bot=false, front=false, back=false),
    free_slip    = (left=true, right=true, top=true, bot=true, front=true, back=true),
    free_surface = false
)
```