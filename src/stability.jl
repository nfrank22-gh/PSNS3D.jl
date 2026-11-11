"""
    courant_number(prob, u, dt, ws)

The Courant number of a step `dt` on the field `u`,

    Δt · max|u_i| · k_max,

the maximum taken over every point and component, independent of the
integrator, with `k_max` the top wavenumber per direction surviving 2/3
dealiasing. It is the advective exponent `|z| = Δt |k·u|` of the top mode
along one axis. The mask keeps a cube of modes, though, and toward its
corners the exponent reaches

    |z| ≤ Δt Σ_i |k_i| |u_i| ≤ 3 courant_number,

so, with frozen coefficients, `courant_number ≤ imag_reach(integrator)`
is necessary for stability and `3 courant_number ≤ imag_reach` sufficient.
"""
function courant_number(prob::NSProblem, u::AbstractArray, dt::Real, ws::NSWorkspace)
    umax = Float64(maximum(abs, u))
    return dt * umax * ws.kmax
end

"""
    cfl_number(prob, u, dt, ws)

The advective margin of a step `dt` on the field `u`,

    courant_number / imag_reach(integrator),

a ratio to the integrator's limit. By [`courant_number`](@ref), `> 1`
means unstable, while only `≤ 1/3` guarantees that every mode of the
cubic 2/3 mask is within the limit. `Inf` for an integrator with no
imaginary reach ([`RK2`](@ref)) on a nonzero field.
"""
cfl_number(prob::NSProblem, u::AbstractArray, dt::Real, ws::NSWorkspace) =
    courant_number(prob, u, dt, ws) / imag_reach(prob.integrator)

"""
    viscous_number(prob, dt, ws)

`Δt · ν · k_max²`, the viscous exponent `|z|` of the top mode along one
axis. The 2/3 mask keeps a cube of modes, whose corners have
`|k|² = 3 k_max²`, so the largest viscous exponent is
`3 viscous_number`. For an integrator that marches `-νk²` explicitly
that is a limit: stability requires
`3 viscous_number ≤ real_reach(integrator)`.
"""
viscous_number(prob::NSProblem, dt::Real, ws::NSWorkspace) = dt * prob.ν * ws.kmax^2

"""
    cfl_dt(prob, u, cfl, ws)

The `Δt` at which [`cfl_number`](@ref) equals the target `cfl`,

    Δt = cfl · imag_reach / (max|u_i| · k_max).

`cfl ≤ 1/3` keeps every advective exponent of the cubic 2/3 mask within
the integrator's reach; at `cfl = 1` only those of the modes along an
axis are guaranteed to be.

`Inf` for a zero field, and `0` for an integrator with no imaginary
reach ([`RK2`](@ref)), which has no stable advective step.
"""
function cfl_dt(prob::NSProblem, u::AbstractArray, cfl::Real, ws::NSWorkspace)
    umax = Float64(maximum(abs, u))
    umax == 0 && return Inf
    return cfl * imag_reach(prob.integrator) / (umax * ws.kmax)
end

"""
    spectral_tail(u, ws)

The fraction of the dealiased energy `½Σ|û|²` carried by
`|k| > (2/3) k_max`. It stays small while the field is resolved and
climbs when energy piles up against the dealiasing cutoff. A CFL spike
with a clean tail is a time-step problem; one that follows a rising tail
is a resolution problem that a smaller `Δt` only delays.

On the half-spectrum axis every `j₁ > 0` bin stands for itself and its
conjugate, hence the weight `2` there (the Nyquist bin is outside the
mask, so needs no exception). Costs one forward transform.
"""
function spectral_tail(u::AbstractArray, ws::NSWorkspace)
    T = real(eltype(u))
    û = forward_transform(u, ws.plan)
    e = sum(abs2, û; dims=4) .* ws.mask .* ifelse.(ws.k1 .== 0, one(T), T(2))
    total = Float64(sum(e))
    total == 0 && return 0.0
    kc2 = T((2 * ws.kmax / 3)^2)
    return Float64(sum(e .* (ws.ksq .> kc2))) / total
end
