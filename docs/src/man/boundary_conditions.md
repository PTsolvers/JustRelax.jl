# Flow boundary conditions

Supported boundary conditions:

1. Free slip

    $\frac{\partial u_i}{\partial x_i} = 0$ at the boundary $\Gamma$

2. No slip

    $u_i = 0$ at the boundary $\Gamma$

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
)
bcs = DisplacementBoundaryConditions(;
    no_slip      = (left=false, right=false, top=false, bot=false),
    free_slip    = (left=true, right=true, top=true, bot=true),
)
```

The equivalent for the 3D case would be:
```julia
bcs = VelocityBoundaryConditions(;
    no_slip      = (left=false, right=false, top=false, bot=false, front=false, back=false),
    free_slip    = (left=true, right=true, top=true, bot=true, front=true, back=true),
)
bcs = DisplacementBoundaryConditions(;
    no_slip      = (left=false, right=false, top=false, bot=false, front=false, back=false),
    free_slip    = (left=true, right=true, top=true, bot=true, front=true, back=true),
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

# Thermal Boundary Conditions

Thermal boundary conditions are collected in `TemperatureBoundaryConditions`.
The same type is used in 2D and 3D.

Supported thermal boundary conditions:

1. No flux

    $\frac{\partial T}{\partial x_i} = 0$ at the boundary $\Gamma$

2. Constant temperature on the outer boundary

    $T = T_\Gamma$ at the boundary $\Gamma$

3. Constant heat flux in the pseudo-transient diffusion kernels

    $q_T = q_\Gamma$ at the boundary $\Gamma$

4. Periodic temperature

    $T(\Gamma_i) = T(\Gamma_j)$ across paired boundaries

5. Mask-based Dirichlet values inside the domain

    $T = f(x_i)$ at selected points in $\Omega$

## Face Names

In 2D, boundary tuples use `left`, `right`, `top`, and `bot`.
In 3D, add `front` and `back`:

```julia
thermal_bc = TemperatureBoundaryConditions(;
    no_flux = (
        left = true,
        right = true,
        front = true,
        back = true,
        top = false,
        bot = false,
    ),
)
```

Faces omitted from `no_flux`, `constant_flux`, `constant_value`, or `periodic`
are treated as inactive for that condition. The dimensionality is inferred from
the longest tuple you provide, so a tuple with `front` or `back` creates a 3D
boundary-condition set.

## No-Flux Boundaries

Use `no_flux` to copy the adjacent interior temperature into the ghost layer.
For example, this applies no-flux boundaries on the left and right sides of a 2D
domain:

```julia
thermal_bc = TemperatureBoundaryConditions(;
    no_flux = (left = true, right = true, top = false, bot = false),
)

thermal_bcs!(thermal, thermal_bc)
```

## Constant-Value Boundaries

Use `constant_value` for fixed-temperature outer boundaries. These values are
applied by `thermal_bcs!` through the ghost-cell relation
`Tghost = 2 * Tboundary - Tinterior`.

```julia
thermal_bc = TemperatureBoundaryConditions(;
    no_flux = (left = true, right = true),
    constant_value = (top = 273.0, bot = 1573.0),
)

thermal_bcs!(thermal, thermal_bc)
```

If `no_flux` and `constant_value` are both active on the same face, `thermal_bcs!`
applies `constant_value` first and `no_flux` second.

## Periodic Boundaries

Use `periodic` to copy the opposite interior temperature into the ghost layer.
For example, this applies periodic temperature boundaries on the left and right
sides of a 2D domain:

```julia
thermal_bc = TemperatureBoundaryConditions(;
    periodic = (left = true, right = true, top = false, bot = false),
)

thermal_bcs!(thermal, thermal_bc)
```

In 3D, include `front` and `back` when those faces should also be periodic:

```julia
thermal_bc = TemperatureBoundaryConditions(;
    periodic = (
        left = true,
        right = true,
        front = true,
        back = true,
        top = false,
        bot = false,
    ),
)
```

If multiple ghost-cell conditions are active on the same face, `thermal_bcs!`
applies `constant_value` first, `no_flux` second, and `periodic` last.

## Constant-Flux Boundaries

Use `constant_flux` to prescribe flux values in the pseudo-transient heat diffusion
solver. These values are consumed by the PT `compute_flux!` kernels, not by
`thermal_bcs!`.

```julia
thermal_bc = TemperatureBoundaryConditions(;
    no_flux = (left = true, right = true, front = true, back = true),
    constant_flux = (top = 0.0, bot = 0.03),
)
```

## Mask-Based Dirichlet Conditions

Use `dirichlet` for fixed values inside the domain, selected by a mask:

```julia
thermal_bc = TemperatureBoundaryConditions(;
    no_flux = (left = true, right = true, top = false, bot = false),
    dirichlet = (; constant = 273.0, mask = mask),
)
```
