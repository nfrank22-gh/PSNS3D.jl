"""
    AbstractIntegrator

A time integrator for the spectral momentum equation. A subtype
implements

    advance(integ, prob, û, t, dt, ws) -> û′

one step of size `dt` from `û` at time `t`, and the two stability reaches

    imag_reach(integ)    # extent of the stability region along the imaginary axis
    real_reach(integ)    # ... and along the negative real axis (`Inf` if unbounded)

which [`cfl_number`](@ref), [`viscous_number`](@ref) and [`cfl_dt`](@ref)
turn into margins. An integrating-factor scheme, for instance, would
report `real_reach = Inf`, and the viscous limit drops out.
"""
abstract type AbstractIntegrator end

"""
    RK4()

The classical fourth-order Runge--Kutta method, with the viscous term
`-νk²û` marched explicitly alongside the convection. With
`R = rhs_spectral`,

    k₁ = R(û, t)
    k₂ = R(û + (Δt/2) k₁, t + Δt/2)
    k₃ = R(û + (Δt/2) k₂, t + Δt/2)
    k₄ = R(û + Δt k₃,     t + Δt)
    û' = û + (Δt/6)(k₁ + 2k₂ + 2k₃ + k₄)

Four evaluations of `R`, each nine transforms. Because the viscous term is
explicit, `Δt` carries a viscous limit (`viscous_number ≤ real_reach`) as
well as the advective one.
"""
struct RK4 <: AbstractIntegrator end

"""
    imag_reach(::RK4), real_reach(::RK4)

Where the RK4 stability polynomial `1 + z + z²/2 + z³/6 + z⁴/24` leaves
the unit disk along the imaginary and the negative real axis: `2√2` and
`≈ 2.785`. The first bounds the advective step, the second the viscous
one.
"""
imag_reach(::RK4) = 2sqrt(2)
real_reach(::RK4) = 2.785293563405282
