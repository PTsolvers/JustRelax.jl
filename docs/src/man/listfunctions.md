# Index

Every exported function and type in one flat list. For the same functions
grouped by topic, see the other pages under API reference in the sidebar.

```@autodocs
Modules = [JustRelax, JustRelax.JustRelax2D, JustRelax.JustRelax3D, JustRelax.DataIO]
```

The variational-Stokes entry point is `solve_VariationalStokes!` in
`JustRelax.JustRelax2D` and `JustRelax.JustRelax3D`. Its volume-fraction
container, update routines, and `surface_pressure` keyword are documented on the
[variational Stokes](./variational_stokes.md) page.

The dynamic-relaxation counterpart is `solve_VariationalDYREL!`, also in both
modules. It requires a `RockRatio` in both the `DYREL` constructor and solver
call; see [DYREL](./DYREL.md).
