"""
    cfl_number(prob, u, dt, ws)

The advective margin of a step `dt` on the field `u`,

    Δt · max|u| · k_max / imag_reach(integrator),

a ratio to the integrator's limit, so `> 1` means unstable. `k_max` is
the top mode surviving 2/3 dealiasing.
"""
function cfl_number(prob::NSProblem, u::AbstractArray, dt::Real, ws::NSWorkspace)
    umax = Float64(maximum(abs, u))
    return dt * umax * ws.kmax / imag_reach(prob.integrator)
end

"""
    viscous_number(prob, dt, ws)

`Δt · ν · k_max²`, the magnitude of the largest viscous exponent `|z|`.
For an integrator that marches `-νk²` explicitly this *is* a limit:
stability requires `viscous_number ≤ real_reach(integrator)`.
"""
viscous_number(prob::NSProblem, dt::Real, ws::NSWorkspace) = dt * prob.ν * ws.kmax^2

"""
    cfl_dt(prob, u, cfl, ws)

The `Δt` at which [`cfl_number`](@ref) equals the target `cfl`,

    Δt = cfl · imag_reach / (max|u| · k_max).

`Inf` for a zero field.
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
