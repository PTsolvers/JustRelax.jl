# Constitutive equation

In the more general case, JustRelax.jl implements a elasto-visco-elastoplastic rheology. Where the constitutive equation is:

$$
\begin{align}
\boldsymbol{\dot\varepsilon} = 
\boldsymbol{\dot\varepsilon}^{\text{viscous}} + 
\boldsymbol{\dot\varepsilon}^{\text{elastic}} + 
\boldsymbol{\dot\varepsilon}^{\text{plastic}}
\end{align}
$$

or 

$$
\begin{align}
\boldsymbol{\dot\varepsilon} = 
\frac{1}{2\eta_{\text{eff}}}\boldsymbol{\tau} +
\frac{1}{2G} \frac{D\boldsymbol{\tau}}{Dt}  + \dot\lambda\frac{\partial Q}{\partial \boldsymbol{\tau_{II}}}
\end{align}
$$

where $\eta_{\text{eff}}$ is the effective viscosity, $G$ is the elastic shear modulus, $\dot\lambda$ is the plastic multiplier, and $Q$ is the plastic flow potential.

## Elastic stress

### Method (1): Jaumann derivative
$$
\begin{align}
\frac{D\boldsymbol{\tau}}{Dt} =
\boldsymbol{v}\frac{\partial\boldsymbol{\tau}}{\partial t} +
\boldsymbol{\omega}\boldsymbol{\tau} -
\boldsymbol{\tau}\boldsymbol{\omega}^T
\end{align}
$$

where $\boldsymbol{\omega}$ is the vorticity tensor

$$
\begin{align}
\boldsymbol{\omega} = 
\frac{1}{2} \left(\nabla\boldsymbol{v} - \nabla^T \boldsymbol{v} \right)
\end{align}
$$

### Method (2): Euler-Rodrigues rotation

```julia
# from Anton's talk
@inline Base.@propagate_inbounds function rotate_elastic_stress3D(ωi, τ, dt)
    # vorticity
    ω = √(sum(x^2 for x in ωi))
    # unit rotation axis
    n = SVector{3,  Float64}(inv(ω) * ωi[i] for i in 1:3)
    # integrate rotation angle
    θ = dt * 0.5 * ω
    # Euler Rodrigues rotation matrix
    R = rodrigues_euler(θ, n)
    # rotate tensor
    τij = voigt2tensor(τ)
    τij_rot = R * (τij * R')
    tensor2voigt(τij_rot)
end
```

1. Compute unit rotation axis
$$
\begin{align}
    \alpha = \sqrt{\boldsymbol{\omega} \cdot \boldsymbol{u}} \\
    \boldsymbol{n} = \alpha \boldsymbol{I} \frac{1}{\boldsymbol{\omega}}
\end{align}
$$

where $\boldsymbol{u}$ is a unit vector of length $\mathbb{R}^n$.

2. Integrate rotation angle

$$
\begin{align}
    \theta = \frac{\alpha}{2\Delta t}
\end{align}
$$

3. Euler-Rodrigues rotation matrix
$$
\begin{align}
    \boldsymbol{c} = \boldsymbol{n}\sin{\theta} \\
    \boldsymbol{R_1} = 
    \left[
        c0   -c_3   c_2 \\
        c_3    c0  -c_1 \\
       -c_2   c_1    c0 \\
    \right] \\
    \boldsymbol{R_2} = (1-\cos{\theta}) \boldsymbol{n} \boldsymbol{n}^T \\
    \boldsymbol{R}=\boldsymbol{R_1}+\boldsymbol{R_2}
\end{align}
$$

4. Rotate stress tensor

$$
\begin{align}
    \boldsymbol{\tau}^{\star}=\boldsymbol{R}\boldsymbol{\tau}\boldsymbol{R}^T
\end{align}
$$

## Plastic formulation

JustRelax.jl implements the regularised plasticity model from (refs). In this formulation, the yield function is given by

$$
\begin{align}
F = \tau_y - \left( P \sin{\phi} + C \cos{\phi} + \eta_{\text{reg}} + \dot\lambda \right) \leq 0
\end{align}
$$

where $\eta_{\text{reg}}$ is a regularization term and 

$$
\begin{align}
\dot\lambda = \frac{F}{x}
\end{align}
$$

Note that since we are using an iterative method to solve the APT Stokes equation, the non-linearities are dealt by the iterative scheme. Othwersie, one would need to solve a non-linear problem to compute $\dot\lambda$, which typically requires to compute $\frac{\partial F}{\partial \dot\lambda}$