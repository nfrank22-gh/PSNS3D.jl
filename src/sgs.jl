"""
    AbstractSGSModel

The subgrid-scale stress `τ_ij` of a large-eddy simulation, which enters
the filtered momentum equation as `-∂τ_ij/∂x_j`. A model is a subtype
that implements

    sgs_stress_divergence(m, û, ws) -> Ĝ

returning `Ĝ_i = -(∂τ_ij/∂x_j)^`, the SGS term *as it appears on the
right-hand side*, in spectral space, `(N÷2+1, N, N, 3)`, in the
normalization of the array contract, and dealiased. The solver adds `Ĝ`
to the convective term before the projection, so the two share one
`P(k)`: a model never has to remove its own divergence, and the isotropic
part of `τ_ij` is simply absorbed by the pressure.

`sgs_stress_divergence` is evaluated at every stage of the time
integrator, on that stage's state. Unlike a constant viscosity, an eddy
viscosity is not diagonal in `k`, so it is marched explicitly with the
convection.
"""
abstract type AbstractSGSModel end

"""
    NoSGS()

No subgrid model: a direct numerical simulation. Short-circuits the SGS
term entirely, so it costs nothing.
"""
struct NoSGS <: AbstractSGSModel end

"""
    Smagorinsky(Cs, Δ)
    Smagorinsky(grid; Cs=0.17)

The standard Smagorinsky model. The deviatoric stress is proportional to
the resolved strain rate,

    τ_ij - ⅓τ_kk δ_ij = -2ν_t S_ij,   S_ij = ½(∂_j u_i + ∂_i u_j),

with the eddy viscosity

    ν_t = (Cs Δ)² |S|,   |S| = √(2 S_ij S_ij),

so that `Ĝ_i = i k_j (2ν_t S_ij)^`. `Cs` is the Smagorinsky constant and
`Δ` the filter width; the grid form takes `Δ = grid.dx`.
"""
struct Smagorinsky{T<:Real} <: AbstractSGSModel
    Cs::T
    Δ::T
end
Smagorinsky(Cs::Real, Δ::Real) = Smagorinsky(promote(Cs, Δ)...)
Smagorinsky(g::PeriodicGrid{T}; Cs::Real=0.17) where {T} = Smagorinsky(T(Cs), g.dx)

"""
    sgs_stress_divergence(m, û, ws)

The SGS term `Ĝ = -(∂τ_ij/∂x_j)^` of model `m` on the state `û`. See
[`AbstractSGSModel`](@ref).
"""
function sgs_stress_divergence end

"""
    sgs_stress_divergence(m::Smagorinsky, û, ws)

`Ĝ_i = i k_j T̂_ij` with `T_ij = 2ν_t S_ij`, evaluated pseudo-spectrally:
the six independent components of `Ŝ_ij = (i/2)(k_j û_i + k_i û_j)` are
brought to physical space (6 inverse transforms), `|S|` and `T_ij` are
formed pointwise, and `T_ij` is brought back (6 forward transforms) and
masked by the 2/3 rule before the divergence is taken.

Uses the derivative wavenumbers `kd` (Nyquist zeroed), as [`curl`](@ref)
does. `|S|` is not a polynomial in `u`, so the 2/3 rule does not remove
its aliases entirely. No traceless correction is made: `S_kk = ∇·u`
vanishes for a solenoidal field, and any isotropic part of `T_ij` is
annihilated by the projection anyway.
"""
function sgs_stress_divergence(m::Smagorinsky, û, ws::NSWorkspace)
    T = real(eltype(û))
    half = T(1) / 2
    û1, û2, û3 = _comp(û, 1), _comp(û, 2), _comp(û, 3)

    # The plans transform three components at a time, so the diagonal and
    # the off-diagonal strain go in two batches.
    Sd = inverse_transform(cat(@.(im * ws.kd1 * û1),
                               @.(im * ws.kd2 * û2),
                               @.(im * ws.kd3 * û3); dims=4), ws.plan)
    So = inverse_transform(cat(@.(im * half * (ws.kd2 * û1 + ws.kd1 * û2)),
                               @.(im * half * (ws.kd3 * û1 + ws.kd1 * û3)),
                               @.(im * half * (ws.kd3 * û2 + ws.kd2 * û3)); dims=4),
                           ws.plan)

    s11, s22, s33 = _comp(Sd, 1), _comp(Sd, 2), _comp(Sd, 3)
    s12, s13, s23 = _comp(So, 1), _comp(So, 2), _comp(So, 3)
    c = 2 * T(m.Cs * m.Δ)^2
    # 2ν_t = 2(Cs Δ)² √(2 S_ij S_ij), off-diagonal terms counted twice.
    twoνt = @. c * sqrt(2 * (s11^2 + s22^2 + s33^2) + 4 * (s12^2 + s13^2 + s23^2))

    Sd .*= twoνt
    So .*= twoνt
    T̂d = forward_transform(Sd, ws.plan) .* ws.mask
    T̂o = forward_transform(So, ws.plan) .* ws.mask

    t11, t22, t33 = _comp(T̂d, 1), _comp(T̂d, 2), _comp(T̂d, 3)
    t12, t13, t23 = _comp(T̂o, 1), _comp(T̂o, 2), _comp(T̂o, 3)
    Ĝ1 = @. im * (ws.kd1 * t11 + ws.kd2 * t12 + ws.kd3 * t13)
    Ĝ2 = @. im * (ws.kd1 * t12 + ws.kd2 * t22 + ws.kd3 * t23)
    Ĝ3 = @. im * (ws.kd1 * t13 + ws.kd2 * t23 + ws.kd3 * t33)
    return cat(Ĝ1, Ĝ2, Ĝ3; dims=4)
end
