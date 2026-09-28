"""
The validation cases and their initial conditions.

Every case is decaying turbulence in the `2π`-periodic box, started from
`turbulent_puff_ic` with `sigma = nothing` (isotropic, box-filling). The
two solvers differ in dealiasing --- PSNS3D truncates to the 2/3 band,
HIT_Spectral pads by 3/2 and keeps every mode up to `N/2` --- so they
agree only up to the energy HIT carries outside the 2/3 band. The cases
are built around that:

  - `lowre`: viscous enough that the out-of-band energy stays negligible,
    so the solvers must agree to near roundoff. The sharp bug detector.
  - `sweep_N32/64/128`: one turbulent flow at three resolutions. The
    out-of-band energy, and with it the disagreement, must fall
    spectrally with `N`. A real bug would not.
  - `les`: the sweep's flow at `N = 64` with `ν = 0` and the Smagorinsky
    model. The two codes implement the model differently (see the
    README), so this is a physics comparison only.

The `N = 128` member of the sweep doubles as the physics comparison at
the target resolution (report only, no threshold).
"""

"""
    Case

`N` points per direction, viscosity `ν`, a fixed step `dt` taken
`nsteps` times, HIT dumps every `dump_every` steps. The initial condition
is `turbulent_puff_ic(k_p, E0, seed)` built on an `N_ic`-point grid and
spectrally truncated to the 2/3 band of `N` --- so the members of a sweep
share their resolved modes exactly.

`Cs` is the Smagorinsky constant of an LES with `Δ = dx`, or `nothing`
for a DNS. HIT_Spectral's `ν_T = Smag_C dx² |S|` takes `Smag_C = Cs²`.
"""
Base.@kwdef struct Case
    name::String
    N::Int
    ν::Float64
    dt::Float64
    nsteps::Int
    dump_every::Int
    k_p::Float64
    E0::Float64
    seed::Int = 1
    N_ic::Int = N
    Cs::Union{Nothing,Float64} = nothing
end

# ∫u_i u_i dx = 3 u'² (2π)³, so u' = 1 initially.
const E0_UNIT = 3 * (2π)^3

const CASES = [
    Case(name="lowre", N=64, ν=0.1, dt=0.005, nsteps=10000, dump_every=20,
         k_p=2, E0=E0_UNIT),
    Case(name="sweep_N32", N=32, ν=0.01, dt=0.0025, nsteps=10000, dump_every=40,
         k_p=4, E0=E0_UNIT, N_ic=128),
    Case(name="sweep_N64", N=64, ν=0.01, dt=0.0025, nsteps=10000, dump_every=40,
         k_p=4, E0=E0_UNIT, N_ic=128),
    #Case(name="sweep_N128", N=128, ν=0.01, dt=0.0025, nsteps=800, dump_every=40,
     #    k_p=4, E0=E0_UNIT, N_ic=128),
    Case(name="les", N=64, ν=0.0, dt=0.0025, nsteps=800, dump_every=40,
         k_p=4, E0=E0_UNIT, N_ic=128, Cs=0.17),
]

findcase(name) = CASES[findfirst(c -> c.name == name, CASES)]

grid(case::Case) = PeriodicGrid(2π, case.N; origin=0)

sgs(case::Case) = isnothing(case.Cs) ? NoSGS() : Smagorinsky(grid(case); Cs=case.Cs)

"""
    strain_magnitude(û, ws)

`|S| = √(2 S_ij S_ij)` in physical space, with the derivative
wavenumbers `kd` over every mode of `û` but Nyquist --- not only the 2/3 band, so for
a HIT field it includes the modes HIT keeps between `N/3` and `N/2`.
"""
function strain_magnitude(û, ws)
    # G[i][:, :, :, j] = ∂_j u_i; the plans transform three components at a time.
    function grad(ûi)
        return PSNS3D.inverse_transform(cat(@.(im * ws.kd1 * ûi), @.(im * ws.kd2 * ûi),
                                            @.(im * ws.kd3 * ûi); dims=4), ws.plan)
    end
    G = [grad(view(û, :, :, :, i)) for i in 1:3]
    S2 = zero(view(G[1], :, :, :, 1))
    for i in 1:3, j in 1:3
        S2 .+= (view(G[i], :, :, :, j) .+ view(G[j], :, :, :, i)) .^ 2 ./ 4
    end
    return sqrt.(2 .* S2)
end

"""
    initial_condition(case)

The case's initial velocity on its own grid, zero outside the 2/3 band
of `N`. Both solvers start from this array: PSNS3D would mask it on its
first step anyway, and masking it up front means HIT does not start with
modes PSNS3D never sees.
"""
function initial_condition(case::Case)
    gs = PeriodicGrid(2π, case.N_ic; origin=0)
    ûs = PSNS3D.forward_transform(turbulent_puff_ic(gs; k_p=case.k_p, E0=case.E0,
                                                    seed=case.seed),
                                  NSWorkspace(gs).plan)
    g = grid(case)
    ws = NSWorkspace(g)
    N, Ns = case.N, case.N_ic
    M = N ÷ 2 + 1
    fine(j) = j >= 0 ? j + 1 : Ns + j + 1          # signed bin -> source index
    jf = PSNS3D.signedbins(N)
    û = zeros(eltype(ûs), M, N, N, 3)
    for c in 1:3, (i3, j3) in enumerate(jf), (i2, j2) in enumerate(jf), i1 in 1:M
        û[i1, i2, i3, c] = ûs[i1, fine(j2), fine(j3), c]
    end
    û .*= ws.mask
    return PSNS3D.inverse_transform(û, ws.plan)
end

"""
    hit_step_limits(case, u0)

The margins of `case.dt` against HIT_Spectral's explicit RK4 limits at
`t = 0`. HIT keeps every mode up to `|k_d| = N/2`, so its reach is set by
`k² = 3(N/2)²` rather than by PSNS3D's 2/3 band: `zvisc = dt ν k²` must
stay below the RK4 real-axis reach `2.785`, and the advective number
`dt Σ_d max|u_d| (N/2)` below the imaginary reach `2√2`. Both are
reported as ratios to the limit, so anything above `1` is unstable.

For an LES, `ν` is replaced by `ν + max ν_t`, with `ν_t = (Cs dx)²|S|` of
`u0` --- HIT's own diffusive CFL uses `ν + max ν_T` the same way.
"""
function hit_step_limits(case::Case, u0)
    kN = case.N / 2
    νmax = case.ν
    if !isnothing(case.Cs)
        g = grid(case)
        ws = NSWorkspace(g)
        νmax += (case.Cs * g.dx)^2 *
                maximum(strain_magnitude(PSNS3D.forward_transform(u0, ws.plan), ws))
    end
    zvisc = case.dt * νmax * 3kN^2 / real_reach(RK4())
    adv = case.dt * kN * sum(maximum(abs, view(u0, :, :, :, d)) for d in 1:3) /
          imag_reach(RK4())
    return (zvisc=zvisc, adv=adv)
end
