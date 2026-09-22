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
"""
abstract type AbstractForcing end

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
