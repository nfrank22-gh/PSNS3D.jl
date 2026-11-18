"""
    AbstractForcing

The body force `f` of the momentum equation. A forcing is a subtype that
implements

    forcing_rhs(f, û, t, ws) -> F̂

returning the force in spectral space, `(N÷2+1, N, N, 3)`, in the
normalization of the array contract. The solver then adds `P(k)·(mask·F̂)`
to the right-hand side: it dealiases the force and projects it onto the
divergence-free subspace itself, so a forcing never has to worry about
either, and a forcing with a gradient part cannot break continuity ---
the gradient is simply absorbed by the pressure.

`forcing_rhs` is evaluated at every stage of the time integrator, on that
stage's state and time, so state-dependent forcings (linear forcing,
constant-power band forcing) keep the integrator's order.

A forcing may also implement

    begin_step!(f, û, t, dt, ws)

called once at the start of each step, before any stage. This is where a
stochastic forcing draws its noise, so the same realization is held
across every stage of the step (re-drawing per stage is wrong, and white
noise needs its `√dt` scaling applied here). The default does nothing.

## Forcing in physical space

A forcing that is formed pointwise from the velocity, `f(u, t)`, can
instead declare

    forcing_space(f) = :physical
    physical_forcing!(C, f, u, t, ws)   # C .+= f(u, t), in physical space

and is then added to the physical-space convective product `u × ω`
before its forward transform, sharing that transform, the 2/3 mask and
the projection. Since the mask and `P(k)` are linear this is the same
`P(k)·(mask·F̂)` as above, at the cost of one broadcast instead of the
transforms a spectral `forcing_rhs` of `f(u)` would pay. `forcing_rhs`
need not be defined then.
"""
abstract type AbstractForcing end

"""
    forcing_space(f) -> :spectral | :physical

Where forcing `f` is evaluated: `:spectral` (the default) through
[`forcing_rhs`](@ref), or `:physical` through [`physical_forcing!`](@ref).
See [`AbstractForcing`](@ref).
"""
forcing_space(::AbstractForcing) = :spectral

"""
    physical_forcing!(C, f, u, t, ws)

Add the force of a `:physical` forcing `f` (see [`forcing_space`](@ref))
on the physical-space velocity `u` at time `t` to `C` in place, both of
size `(N, N, N, 3)`. Returns `C`.
"""
function physical_forcing! end

"""
    NoForcing()

Free (decaying) flow. Short-circuits the forcing term entirely, so it
costs nothing.
"""
struct NoForcing <: AbstractForcing end

"""
    forcing_rhs(f, û, t, ws)

The spectral force `F̂` of forcing `f` on the state `û` at time `t`. See
[`AbstractForcing`](@ref).
"""
function forcing_rhs end

"""
    begin_step!(f, û, t, dt, ws)

Per-step setup for forcing `f`; the identity by default. See
[`AbstractForcing`](@ref).
"""
begin_step!(::AbstractForcing, û, t, dt, ws) = nothing

# TODO: LinearForcing (Lundgren 2003; Rosales & Meneveau 2005).
#
#     struct LinearForcing{T} <: AbstractForcing
#         A::T
#     end
#     forcing_rhs(f::LinearForcing, û, t, ws) = f.A .* û
#
# f = A u is already solenoidal, so the projection is a no-op on it. The
# usual refinements: exclude the mean (k = 0) mode, and optionally set
# `A = ε / (2E)` each step (via `begin_step!`) for a target dissipation ε.
