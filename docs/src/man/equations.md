# Stokes equations

Stokes equations for compressible flow are described by

$$\nabla\cdot\boldsymbol{\tau} + \nabla p = \boldsymbol{f} $$
    
$$\nabla\cdot\boldsymbol{v} + \beta \frac{\partial p}{\partial t} + \alpha \frac{\partial T}{\partial t} = 0$$

where $\nabla = \left(\frac{\partial }{\partial x_i}, ..., \frac{\partial }{\partial x_n} \right)$ is the nabla operator, $\boldsymbol{\tau}$ is the deviatoric stress tensor, $p$ is the pressure, $\boldsymbol{f}$ is the body forces vector, $\boldsymbol{v}$ is the velocity field, and $\beta$ is in the inverse of the bulk modulus. In the simple case of a linear rheology, the constitutive equation for an isotropic flow is $\boldsymbol{\tau} = 2\eta\dot{\boldsymbol{\varepsilon}}$, where $\dot{\boldsymbol{\varepsilon}}$ is the deviatoric strain rate tensor. Heat diffusion

$$\rho C_p \frac{\partial T}{\partial t} = - \nabla q + Q$$
    
$$q = - K \nabla T$$

where $\rho$ is the density, $C_p$ is the heat diffusion, $K$ is the heat conductivity, $Q$ is the sum of any amount of source terms, and $T$ is the temperature.

# Pseudo-transient iterative method

The pseudo-transient method consists in augmenting the right-hand-side of the target PDE with a pseudo-time derivative (where $\psi$ is the pseudo-time) of the primary variables. 

## Stokes

The pseudo-transient formulation of the Stokes equations yields:
    
$$\widetilde{\rho}\frac{\partial \boldsymbol{u}}{\partial \psi} + \nabla\cdot\boldsymbol{\tau} - \nabla p = \boldsymbol{f}$$

$$\frac{1}{\widetilde{K}}\frac{\partial p}{\partial \psi} + \nabla\cdot\boldsymbol{v} = \beta \frac{\partial p}{\partial t} + \alpha \frac{\partial T}{\partial t}$$

We also do a continuation on the constitutive law:

$$\frac{1}{2\widetilde{G}} \frac{\partial\boldsymbol{\tau}}{\partial\psi}+ \frac{1}{2G}\frac{D\boldsymbol{\tau}}{Dt} + \frac{\boldsymbol{\tau}}{2\eta} = \dot{\boldsymbol{\varepsilon}}$$

where the wide tile denotes the effective damping coefficients and $\psi$ is the pseudo-time step. These are defined as in [Raess et al, 2022](https://doi.org/10.5194/gmd-2021-411):

$$\widetilde{\rho} = Re\frac{\eta}{\widetilde{V}L}, \qquad
  \widetilde{G} = \frac{\widetilde{\rho} \widetilde{V}^2}{r+2}, \qquad
  \widetilde{K} = r \widetilde{G}$$

and

$$\widetilde{V} = \sqrt{ \frac{\widetilde{K} +2\widetilde{G}}{\widetilde{\rho}}}, \qquad
    r = \frac{\widetilde{K}}{\widetilde{G}}, \qquad
    Re = \frac{\widetilde{\rho}\widetilde{V}L}{\eta}$$

where the P-wave $\widetilde{V}=V_p$ is the characteristic velocity scale for Stokes, and $Re$ is the Reynolds number. 

## Heat diffusion

The pseudo-transient heat-diffusion equation is:

$$\widetilde{\rho}\frac{\partial T}{\partial \psi} + \rho C_p \frac{\partial T}{\partial t} = \nabla(K\nabla T) = -\nabla q$$

We use a second order pseudo-transient scheme were continuation is also done on the flux, so that:

$$\widetilde{\theta}\frac{\partial q}{\partial \psi} + q  = -K\nabla T$$
