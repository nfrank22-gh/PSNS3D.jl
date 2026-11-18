StepCache(::NSProblem, ws::NSWorkspace) = StepCache(ws)

"""
    advance(prob, û, t, dt, ws)
    advance!(ûP, prob, û, t, dt, ws, sc)

One step of `prob.integrator` of size `dt` from the spectral state `û`
(in the normalization of the array contract) at time `t`. `advance`
returns the new spectral state and leaves `û` alone; `advance!` writes it
into `ûP` (which may be `û` itself), working in the [`StepCache`](@ref)
`sc`, and allocates nothing. The two give bit-identical results.

The incoming state is dealiased once. Every stage then stays inside the
2/3 band without further masking --- the convective, SGS and forcing terms
are masked as they are formed and `-νk²û` is diagonal --- and the result is
masked again only to make that explicit to callers.
"""
advance(prob::NSProblem, û, t, dt, ws::NSWorkspace) =
    advance!(similar(û), prob, û, t, dt, ws, StepCache(ws))

advance!(ûP, prob::NSProblem, û, t, dt, ws::NSWorkspace, sc::StepCache) =
    advance!(ûP, prob.integrator, prob, û, t, dt, ws, sc)

function advance!(ûP, ::RK4, prob::NSProblem, û, t, dt, ws::NSWorkspace, sc::StepCache)
    h = real(eltype(û))(dt)
    û0, acc, stage, k = sc.û0, sc.acc, sc.stage, sc.k
    û0 .= û .* ws.mask
    begin_step!(prob.forcing, û0, t, dt, ws)

    rhs_spectral!(k, prob, û0, t, ws, sc.rhs)
    acc .= k
    stage .= û0 .+ (h / 2) .* k
    rhs_spectral!(k, prob, stage, t + dt / 2, ws, sc.rhs)
    acc .+= 2 .* k
    stage .= û0 .+ (h / 2) .* k
    rhs_spectral!(k, prob, stage, t + dt / 2, ws, sc.rhs)
    acc .+= 2 .* k
    stage .= û0 .+ h .* k
    rhs_spectral!(k, prob, stage, t + dt, ws, sc.rhs)
    acc .+= k

    ûP .= û0 .+ (h / 6) .* acc
    ûP .*= ws.mask
    return ûP
end

function advance!(ûP, ::RK2, prob::NSProblem, û, t, dt, ws::NSWorkspace, sc::StepCache)
    h = real(eltype(û))(dt)
    û0, acc, stage, k = sc.û0, sc.acc, sc.stage, sc.k
    û0 .= û .* ws.mask
    begin_step!(prob.forcing, û0, t, dt, ws)

    rhs_spectral!(acc, prob, û0, t, ws, sc.rhs)
    stage .= û0 .+ h .* acc
    rhs_spectral!(k, prob, stage, t + dt, ws, sc.rhs)
    acc .+= k

    ûP .= û0 .+ (h / 2) .* acc
    ûP .*= ws.mask
    return ûP
end

"""
    advance_physical(prob, u, t, dt, ws)
    advance_physical!(u, ûP, prob, t, dt, ws, sc)

[`advance`](@ref) on a physical-space field: forward transform, one step,
inverse transform. `advance_physical` returns `(u′, û′)`, the new field
and its transform, so a caller that needs both pays for no extra round
trip. `advance_physical!` overwrites `u` with `u′` and writes `û′` into
`ûP`, working in the [`StepCache`](@ref) `sc`; it returns `(u, ûP)`.
"""
function advance_physical(prob::NSProblem, u::AbstractArray, t, dt, ws::NSWorkspace)
    u′ = copy(u)
    ûP = similar(u, Complex{eltype(u)}, size(ws.mask)..., 3)
    return advance_physical!(u′, ûP, prob, t, dt, ws, StepCache(ws))
end

function advance_physical!(u::AbstractArray, ûP, prob::NSProblem, t, dt, ws::NSWorkspace,
                           sc::StepCache)
    forward_transform!(ûP, u, ws.plan)
    advance!(ûP, prob, ûP, t, dt, ws, sc)
    inverse_transform!(u, ûP, ws.plan)
    return u, ûP
end
