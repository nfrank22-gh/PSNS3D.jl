"""
    simulate(prob, u0, (t0, t1); dt, cfl=nothing, ws=workspace(prob), callback=nothing)

March `u0` from `t0` to `t1` and return `(u, t, nsteps)`.

With `cfl = nothing` every step is `dt`. With a numeric `cfl` each step
is `min(dt, cfl_dt(prob, u, cfl, ws))`, so `dt` acts as the cap. The last
step is shortened to land on `t1` exactly.

`callback(u, t, n)`, if given, is called after every step with the new
field, its time and the step count.
"""
function simulate(prob::NSProblem, u0::AbstractArray, tspan; dt::Real,
                  cfl=nothing, ws::NSWorkspace=workspace(prob), callback=nothing)
    t, t1 = float.(tspan)
    u = u0
    n = 0
    while t < t1
        h = cfl === nothing ? float(dt) : min(float(dt), cfl_dt(prob, u, cfl, ws))
        last = t1 - t <= h
        last && (h = t1 - t)
        u, _ = advance_physical(prob, u, t, h, ws)
        t = last ? t1 : t + h
        n += 1
        callback === nothing || callback(u, t, n)
    end
    return (u=u, t=t, nsteps=n)
end
