"""
    PSNS3D

A pseudo-spectral solver for the three-dimensional incompressible
Navier--Stokes equations on a cubic periodic box,

    ∂u_i/∂t = -u_j ∂u_i/∂x_j - ∂p/∂x_i + ν ∂²u_i/∂x_j∂x_j - ∂τ_ij/∂x_j + f_i,
    ∂u_i/∂x_i = 0,

where `τ_ij` is the subgrid-scale stress of a large-eddy simulation (zero
for a DNS). In Fourier space the pressure is eliminated by the projection
`P_il(k) = δ_il - k_i k_l / k²` onto divergence-free fields, leaving

    ∂û_i/∂t = P_il(k) [ (u × ω)^_l + Ĝ_l + f̂_l ] - ν k² û_i,

with `Ĝ = -(∂τ_ij/∂x_j)^`, the convective term written in rotational form
(the gradient part `∇(u·u/2)` is annihilated by `P`), and the nonlinear
terms dealiased by the 2/3 rule.

## Layout

  - `grid.jl`       --- [`PeriodicGrid`](@ref), the cubic box.
  - `transform.jl`  --- `SpectralPlan` and the normalized real transform.
  - `workspace.jl`  --- [`NSWorkspace`](@ref): wavenumbers, mask, plans.
  - `forcing.jl`    --- the [`AbstractForcing`](@ref) extension point.
  - `sgs.jl`        --- the [`AbstractSGSModel`](@ref) extension point, `Smagorinsky`.
  - `problem.jl`    --- [`NSProblem`](@ref) and the right-hand side.
  - `integrators.jl`--- the [`AbstractIntegrator`](@ref) extension point, `RK4`.
  - `timestep.jl`   --- [`advance`](@ref), one step of the integrator.
  - `simulate.jl`   --- [`simulate`](@ref), a plain time loop.
  - `stability.jl`  --- CFL and viscous margins, `cfl_dt`.
  - `utils.jl`      --- curl, Leray projection, divergence, spectrum.
  - `ic.jl`         --- initial conditions.

## Array contract

Downstream code may rely on the following, which is part of the public
API:

  - A velocity field is an array of size `(N, N, N, 3)`, component axis
    last, on the grid's device and in its element type.
  - Its transform is `û = rfft(u, 1:3) / N³`, of size `(N÷2+1, N, N, 3)`.
    [`advance`](@ref) consumes and returns `û` in exactly this convention,
    so a caller holding its own plans can pass arrays straight through.

`SpectralPlan`, `forward_transform`, `inverse_transform`, `wavenumbers`
and `allocate_state` are public but not exported, since their names are
generic; call them as `PSNS3D.forward_transform` and so on.

Fields live on whatever array type the chosen backend supplies, so the
same code runs on CPU and on an NVIDIA GPU; see `backend.jl` and
`ext/PSNS3DCUDAExt.jl`.
"""
module PSNS3D

using FFTW
using Random

# --- backend -------------------------------------------------------------
export Backend, CPUBackend, CUDABackend, backend, arraytype, adapt_to, adapt_to_like

# --- grid, workspace, problem --------------------------------------------
export PeriodicGrid, NSWorkspace, workspace, NSProblem, rhs_spectral

# --- extension points ----------------------------------------------------
export AbstractForcing, NoForcing, forcing_rhs, begin_step!
export AbstractSGSModel, NoSGS, Smagorinsky, sgs_stress_divergence
export AbstractIntegrator, RK4, imag_reach, real_reach

# --- time stepping -------------------------------------------------------
export advance, advance_physical, simulate

# --- stability -----------------------------------------------------------
export cfl_number, viscous_number, spectral_tail, cfl_dt

# --- utilities -----------------------------------------------------------
export curl, project, leray_project!, divergence_residual, spectral_divergence_residual
export energy_spectrum

# --- initial conditions --------------------------------------------------
export turbulent_puff_ic, abc_flow_ic

include("backend.jl")
include("grid.jl")
include("transform.jl")
include("workspace.jl")
include("forcing.jl")
include("sgs.jl")
include("integrators.jl")
include("problem.jl")
include("timestep.jl")
include("simulate.jl")
include("stability.jl")
include("utils.jl")
include("ic.jl")

end # module PSNS3D
