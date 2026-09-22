"""
    advance(prob, û, t, dt, ws)

One step of `prob.integrator` of size `dt` from the spectral state `û`
(in the normalization of the array contract) at time `t`. Returns the new
spectral state; `û` is not modified.

The incoming state is dealiased once. Every stage then stays inside the
2/3 band without further masking --- the convective and forcing terms are
masked as they are formed and `-νk²û` is diagonal --- and the result is
masked again only to make that explicit to callers.
"""
advance(prob::NSProblem, û, t, dt, ws::NSWorkspace) =
    advance(prob.integrator, prob, û, t, dt, ws)

function advance(::RK4, prob::NSProblem, û, t, dt, ws::NSWorkspace)
    h = real(eltype(û))(dt)
    û0 = û .* ws.mask
    begin_step!(prob.forcing, û0, t, dt, ws)

    k = rhs_spectral(prob, û0, t, ws)
    acc = copy(k)
    k = rhs_spectral(prob, û0 .+ (h / 2) .* k, t + dt / 2, ws)
    acc .+= 2 .* k
    k = rhs_spectral(prob, û0 .+ (h / 2) .* k, t + dt / 2, ws)
    acc .+= 2 .* k
    k = rhs_spectral(prob, û0 .+ h .* k, t + dt, ws)
    acc .+= k

    ûP = û0 .+ (h / 6) .* acc
    ûP .*= ws.mask
    return ûP
end

"""
    advance_physical(prob, u, t, dt, ws)

[`advance`](@ref) on a physical-space field: forward transform, one step,
inverse transform. Returns `(u′, û′)`, the new field and its transform,
so a caller that needs both pays for no extra round trip.
"""
function advance_physical(prob::NSProblem, u::AbstractArray, t, dt, ws::NSWorkspace)
    ûP = advance(prob, forward_transform(u, ws.plan), t, dt, ws)
    return inverse_transform(ûP, ws.plan), ûP
end
