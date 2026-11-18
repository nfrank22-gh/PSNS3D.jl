"""
    NSProblem(grid; ν, sgs=NoSGS(), forcing=NoForcing(), integrator=RK4())

The problem statement: a [`PeriodicGrid`](@ref), the kinematic viscosity
`ν`, an [`AbstractSGSModel`](@ref), an [`AbstractForcing`](@ref) and an
[`AbstractIntegrator`](@ref). Carries no arrays; those live in the
[`NSWorkspace`](@ref) built by [`workspace`](@ref).

`ν = 0` is allowed and skips the viscous term entirely --- the usual
setting for an LES at infinite Reynolds number, where the SGS model does
all the dissipating.
"""
struct NSProblem{G<:PeriodicGrid,T<:Real,S<:AbstractSGSModel,F<:AbstractForcing,
                 I<:AbstractIntegrator}
    grid::G
    ν::T
    sgs::S
    forcing::F
    integrator::I
end
NSProblem(grid::PeriodicGrid; ν::Real, sgs::AbstractSGSModel=NoSGS(),
          forcing::AbstractForcing=NoForcing(), integrator::AbstractIntegrator=RK4()) =
    NSProblem(grid, ν, sgs, forcing, integrator)

"""
    workspace(prob)

The [`NSWorkspace`](@ref) for `prob`'s grid.
"""
workspace(prob::NSProblem) = NSWorkspace(prob.grid)

"""
    project(Ĉ, ws)
    project!(P, Ĉ, ws, z)

`P_il(k) Ĉ_l`, the projection of a spectral vector field onto its
divergence-free part, with `P_il = δ_il - k_i k_l / k²` and the `k = 0`
mode left unchanged. `project` returns a new array; `project!` writes
into `P` (which may be `Ĉ` itself) and uses `z`, of size `(N÷2+1, N, N)`,
for `k·Ĉ`.
"""
function project!(P, Ĉ, ws::NSWorkspace, z)
    Ĉ1, Ĉ2, Ĉ3 = _comp(Ĉ, 1), _comp(Ĉ, 2), _comp(Ĉ, 3)
    @. z = ws.k1 * Ĉ1 + ws.k2 * Ĉ2 + ws.k3 * Ĉ3
    # `unit` is hoisted out of the `@.`: inside it, `one(eltype(ws.ksq))`
    # would itself be broadcast, giving an array of `DataType` intermediates
    # --- silently fine on the host, `Any`-typed and not GPU-compatible.
    unit = one(eltype(ws.ksq))
    P1, P2, P3 = _comp(P, 1), _comp(P, 2), _comp(P, 3)
    @. P1 = Ĉ1 - ws.k1 * z / ifelse(ws.ksq == 0, unit, ws.ksq)
    @. P2 = Ĉ2 - ws.k2 * z / ifelse(ws.ksq == 0, unit, ws.ksq)
    @. P3 = Ĉ3 - ws.k3 * z / ifelse(ws.ksq == 0, unit, ws.ksq)
    return P
end
project(Ĉ, ws::NSWorkspace) = project!(similar(Ĉ), Ĉ, ws, similar(Ĉ, size(Ĉ)[1:3]))

"""
    nonlinear(û, ws)
    nonlinear!(Ĉ, û, ws, rc; forcing=NoForcing(), t=0)

The dealiased convective term `(u × ω)^`, evaluated pseudo-spectrally in
*rotational* form, before projection.

Writing `u_j ∂_j u_i = -(u × ω)_i + ∂_i(u_k u_k / 2)`, the gradient part
is annihilated by the projection `P_il` that [`rhs_spectral`](@ref)
applies, since `P_il k_l = 0`. This costs 6 inverse transforms (`u` and
`ω`) plus 3 forward, against 15 for the convective form `u_j ∂_j u_i`
--- and its aliasing errors are smaller.

`nonlinear!` writes into `Ĉ` and works in the [`RHSCache`](@ref) `rc`,
leaving `rc.U` holding the velocity. A `forcing` whose
[`forcing_space`](@ref) is `:physical` is added to the product in
physical space, before the transform, so it shares its transform and
mask; see [`physical_forcing!`](@ref).
"""
function nonlinear!(Ĉ, û, ws::NSWorkspace, rc::RHSCache;
                    forcing::AbstractForcing=NoForcing(), t=0)
    u, ω, c = rc.U, rc.Ω, rc.C
    inverse_transform!(u, û, ws.plan)
    curl!(ω, û, ws, rc.ŝ1)

    u1, u2, u3 = _comp(u, 1), _comp(u, 2), _comp(u, 3)
    o1, o2, o3 = _comp(ω, 1), _comp(ω, 2), _comp(ω, 3)
    c1, c2, c3 = _comp(c, 1), _comp(c, 2), _comp(c, 3)
    @. c1 = u2 * o3 - u3 * o2
    @. c2 = u3 * o1 - u1 * o3
    @. c3 = u1 * o2 - u2 * o1
    forcing_space(forcing) === :physical && physical_forcing!(c, forcing, u, t, ws)

    forward_transform!(Ĉ, c, ws.plan)
    Ĉ .*= ws.mask
    return Ĉ
end
nonlinear(û, ws::NSWorkspace) = nonlinear!(similar(û), û, ws, RHSCache(ws))

"""
    rhs_spectral(prob, û, t, ws)
    rhs_spectral!(R, prob, û, t, ws, rc)

The right-hand side of the spectral momentum equation,

    ∂û_i/∂t = P_il(k) [ (u × ω)^_l + Ĝ_l + mask·f̂_l ] - ν k² û_i,

that the time integrator marches, with `Ĝ` the SGS term of `prob.sgs`
(see [`AbstractSGSModel`](@ref)). The convective and SGS terms share one
projection, and so does a forcing evaluated in physical space
([`forcing_space`](@ref)); a spectral forcing is masked and projected on
its own. With [`NoSGS`](@ref), [`NoForcing`](@ref) or `ν = 0` the
corresponding term is skipped entirely.

`rhs_spectral!` writes into `R` (which must not be `û`) and works in the
[`RHSCache`](@ref) `rc`; `rhs_spectral` allocates both.
"""
function rhs_spectral!(R, prob::NSProblem, û, t, ws::NSWorkspace, rc::RHSCache)
    T = real(eltype(û))
    nonlinear!(R, û, ws, rc; forcing=prob.forcing, t=t)
    add_sgs!(R, prob.sgs, û, ws, rc)
    project!(R, R, ws, rc.z1)
    iszero(prob.ν) || (R .-= T(prob.ν) .* ws.ksq .* û)
    return _add_forcing!(R, prob.forcing, û, t, ws, rc)
end
rhs_spectral(prob::NSProblem, û, t, ws::NSWorkspace) =
    rhs_spectral!(similar(û), prob, û, t, ws, RHSCache(ws))

_add_forcing!(R, f::AbstractForcing, û, t, ws, rc) =
    forcing_space(f) === :physical ? R : _add_spectral_forcing!(R, f, û, t, ws, rc)
_add_forcing!(R, ::NoForcing, û, t, ws, rc) = R

function _add_spectral_forcing!(R, f, û, t, ws, rc)
    F = forcing_rhs(f, û, t, ws) .* ws.mask
    R .+= project!(F, F, ws, rc.z1)
    return R
end
