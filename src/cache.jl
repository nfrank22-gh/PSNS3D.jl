"""
    RHSCache(ws)

The scratch arrays one evaluation of [`rhs_spectral!`](@ref) works in,
allocated once so that a stage allocates nothing on the device:

  - `ŝ1`, `ŝ2`: two spectral three-component arrays, `(N÷2+1, N, N, 3)`
    (the spectral input of an inverse transform, and the two batches of
    the subgrid stress);
  - `U`, `Ω`, `C`: three physical three-component arrays, `(N, N, N, 3)`
    (velocity, vorticity and their product; the subgrid model reuses `U`
    and `Ω` for the diagonal and off-diagonal strain);
  - `r1`: one physical scalar, `(N, N, N)` (the eddy viscosity);
  - `z1`: one spectral scalar, `(N÷2+1, N, N)` (`k·Ĉ` of the projection).

At `N = 256` in `Float32` this is about 1.3 GB.
"""
struct RHSCache{AC,AR,AC1,AR1}
    ŝ1::AC
    ŝ2::AC
    U::AR
    Ω::AR
    C::AR
    r1::AR1
    z1::AC1
end

function RHSCache(ws::NSWorkspace)
    g = ws.grid
    u = allocate_state(g)
    ŝ = similar(u, Complex{eltype(u)}, size(ws.mask)..., 3)
    return RHSCache(ŝ, similar(ŝ), u, similar(u), similar(u),
                    similar(u, size(u)[1:3]), similar(ŝ, size(ŝ)[1:3]))
end

"""
    StepCache(ws)
    StepCache(prob, ws)

Everything one step of [`advance!`](@ref) works in: an [`RHSCache`](@ref)
and the four spectral stage arrays of the Runge--Kutta integrators
(`û0`, the masked initial state; `acc`, the accumulated slope; `stage`,
the state a stage is evaluated at; `k`, its slope). About 2 GB at
`N = 256` in `Float32`. Build one per run and pass it to every step;
[`simulate`](@ref) does.
"""
struct StepCache{R<:RHSCache,AC}
    rhs::R
    û0::AC
    acc::AC
    stage::AC
    k::AC
end

function StepCache(ws::NSWorkspace)
    rc = RHSCache(ws)
    return StepCache(rc, similar(rc.ŝ1), similar(rc.ŝ1), similar(rc.ŝ1), similar(rc.ŝ1))
end
