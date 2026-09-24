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

`P_il(k) Ĉ_l`, the projection of a spectral vector field onto its
divergence-free part, with `P_il = δ_il - k_i k_l / k²` and the `k = 0`
mode left unchanged. Returns a new array.
"""
function project(Ĉ, ws::NSWorkspace)
    Ĉ1, Ĉ2, Ĉ3 = _comp(Ĉ, 1), _comp(Ĉ, 2), _comp(Ĉ, 3)
    kdotC = @. ws.k1 * Ĉ1 + ws.k2 * Ĉ2 + ws.k3 * Ĉ3
    # `unit` is hoisted out of the `@.`: inside it, `one(eltype(ws.ksq))`
    # would itself be broadcast, giving an array of `DataType` intermediates
    # --- silently fine on the host, `Any`-typed and not GPU-compatible.
    unit = one(eltype(ws.ksq))
    safe = @. ifelse(ws.ksq == 0, unit, ws.ksq)
    p1 = @. Ĉ1 - ws.k1 * kdotC / safe
    p2 = @. Ĉ2 - ws.k2 * kdotC / safe
    p3 = @. Ĉ3 - ws.k3 * kdotC / safe
    return cat(p1, p2, p3; dims=4)
end

"""
    nonlinear(û, ws)

The dealiased convective term `(u × ω)^`, evaluated pseudo-spectrally in
*rotational* form, before projection.

Writing `u_j ∂_j u_i = -(u × ω)_i + ∂_i(u_k u_k / 2)`, the gradient part
is annihilated by the projection `P_il` that [`rhs_spectral`](@ref)
applies, since `P_il k_l = 0`. This costs 6 inverse transforms (`u` and
`ω`) plus 3 forward, against 15 for the convective form `u_j ∂_j u_i`
--- and its aliasing errors are smaller.
"""
function nonlinear(û, ws::NSWorkspace)
    u = inverse_transform(û, ws.plan)
    ω = curl(û, ws)

    u1, u2, u3 = _comp(u, 1), _comp(u, 2), _comp(u, 3)
    o1, o2, o3 = _comp(ω, 1), _comp(ω, 2), _comp(ω, 3)
    c1 = @. u2 * o3 - u3 * o2
    c2 = @. u3 * o1 - u1 * o3
    c3 = @. u1 * o2 - u2 * o1

    Ĉ = forward_transform(cat(c1, c2, c3; dims=4), ws.plan)
    Ĉ .*= ws.mask
    return Ĉ
end

"""
    rhs_spectral(prob, û, t, ws)

The right-hand side of the spectral momentum equation,

    ∂û_i/∂t = P_il(k) [ (u × ω)^_l + Ĝ_l + mask·f̂_l ] - ν k² û_i,

that the time integrator marches, with `Ĝ` the SGS term of `prob.sgs`
(see [`AbstractSGSModel`](@ref)). The convective and SGS terms share one
projection. With [`NoSGS`](@ref), [`NoForcing`](@ref) or `ν = 0` the
corresponding term is skipped entirely.
"""
function rhs_spectral(prob::NSProblem, û, t, ws::NSWorkspace)
    T = real(eltype(û))
    R = project(_add_sgs!(nonlinear(û, ws), prob.sgs, û, ws), ws)
    iszero(prob.ν) || (R .-= T(prob.ν) .* ws.ksq .* û)
    return _add_forcing!(R, prob.forcing, û, t, ws)
end

_add_sgs!(Ĉ, ::NoSGS, û, ws) = Ĉ
_add_sgs!(Ĉ, m::AbstractSGSModel, û, ws) = (Ĉ .+= sgs_stress_divergence(m, û, ws); Ĉ)

_add_forcing!(R, ::NoForcing, û, t, ws) = R
_add_forcing!(R, f::AbstractForcing, û, t, ws) =
    (R .+= project(forcing_rhs(f, û, t, ws) .* ws.mask, ws); R)
