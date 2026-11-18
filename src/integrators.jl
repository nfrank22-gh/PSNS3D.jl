"""
    AbstractIntegrator

A time integrator for the spectral momentum equation. A subtype
implements

    advance!(ûP, integ, prob, û, t, dt, ws, sc)

one step of size `dt` from `û` at time `t` into `ûP`, working in the
[`StepCache`](@ref) `sc`, and the two stability reaches

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

Four evaluations of `R`, each nine transforms, plus twelve with the
[`Smagorinsky`](@ref) model. Because the viscous term is
explicit, `Δt` carries a viscous limit (`3 viscous_number ≤ real_reach`,
see [`viscous_number`](@ref)) as well as the advective one.
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

"""
    RK2()

Heun's method (the explicit trapezoidal rule, also SSP-RK2), with the
viscous term `-νk²û` marched explicitly alongside the convection. With
`R = rhs_spectral`,

    k₁ = R(û, t)
    k₂ = R(û + Δt k₁, t + Δt)
    û' = û + (Δt/2)(k₁ + k₂)

Two evaluations of `R`, half the cost of [`RK4`](@ref) per step. It is a
convex combination of two forward-Euler steps, so it is strong-stability
preserving.
"""
struct RK2 <: AbstractIntegrator end

"""
    imag_reach(::RK2), real_reach(::RK2)

The RK2 stability polynomial `1 + z + z²/2` has `|R(iy)|² = 1 + y⁴/4`,
which exceeds `1` for every `y ≠ 0`: no step is strictly stable for pure
advection, so the imaginary reach is `0` and only dissipation (viscous,
SGS, forcing) holds the slow `y⁴/8`-per-step growth in check. Along the
negative real axis it reaches `2`.
"""
imag_reach(::RK2) = 0.0
real_reach(::RK2) = 2.0

"""
    RK3()

Williamson's low-storage third-order Runge--Kutta method (Williamson
1980, "Low-storage Runge-Kutta schemes", J. Comput. Phys. 35, 48), the
three-stage scheme common in pseudo-spectral turbulence codes, with the
viscous term `-νk²û` marched explicitly alongside the convection. With
`R = rhs_spectral` and `q₀ = 0`, for `i = 1, 2, 3`

    q_i = A_i q_{i-1} + Δt R(û_{i-1}, t + c_i Δt)
    û_i = û_{i-1} + B_i q_i

with `A = (0, -5/9, -153/128)`, `B = (1/3, 15/16, 8/15)` and
`c = (0, 1/3, 3/4)`; `û' = û₃`. Only the state and the one register `q`
are carried between stages ("2N storage").

Three evaluations of `R` per step, against two for [`RK2`](@ref). Every
three-stage third-order method has the stability polynomial
`1 + z + z²/2 + z³/6`, which, unlike RK2's, reaches along the imaginary
axis: advective modes are stable for `|z| ≤ √3`.
"""
struct RK3 <: AbstractIntegrator end

"""
    imag_reach(::RK3), real_reach(::RK3)

Where the stability polynomial `1 + z + z²/2 + z³/6` of every
three-stage third-order Runge--Kutta method leaves the unit disk along the
imaginary and the negative real axis: `√3` and `≈ 2.513`.
"""
imag_reach(::RK3) = sqrt(3)
real_reach(::RK3) = 2.512745326618329

# Williamson's (1980) low-storage coefficients, Eq. (RK3).
const _RK3_A = (0.0, -5 / 9, -153 / 128)
const _RK3_B = (1 / 3, 15 / 16, 8 / 15)
const _RK3_C = (0.0, 1 / 3, 3 / 4)
