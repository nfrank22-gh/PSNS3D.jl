"""
    turbulent_puff_ic(g; k_p, sigma=nothing, window_n=2, center=nothing, seed=0, E0=1.0)

A broadband, exactly solenoidal random velocity field with a
Batchelor-type spectrum peaking near `k_p`, in `g`'s element type and on
`g`'s device. With `sigma` set, the field is additionally *localized* in
a puff of width `sigma` about `center` (default: the centre of the box),
Gaussian by default and super-Gaussian for `window_n > 2`; with
`sigma = nothing` it fills the box, an isotropic random initial
condition.

The construction is host work --- it runs once, and the seeded
`MersenneTwister` keeps a run reproducible independently of the backend
--- so the field is built as an `Array` and moved onto the grid's device
at the end. On a CPU grid that move is the identity.

## Why a vector potential

Solenoidality and localization fight each other. Windowing a
divergence-free field breaks `k_i û_i = 0`; projecting a windowed field
restores it but is nonlocal, and reintroduces a slowly-decaying
potential-flow tail that reaches the box edge.

Building the velocity as a curl avoids the conflict entirely. With
`u_i = ε_ijk ∂_j A_k`,

    k_i û_i = i ε_ijk k_i k_j Â_k ≡ 0

identically, by antisymmetry --- and a curl is a local operator, so if `A`
has compact support then so does `u`. Both properties hold exactly and
simultaneously, with no iteration.

## Construction

1. White noise `A` on the grid, from a seeded RNG (drawn in physical
   space, so `Â` is Hermitian and `A` real by construction).
2. Filter `Â ← Â exp(-k²/k_p²)`. White noise is flat in `|Â|²`, so this
   gives `|Â|² ∝ exp(-2k²/k_p²)`; since `|û|² ~ k²|Â|²` and the
   shell-integrated spectrum carries a further `4πk²`, the velocity
   acquires `E(k) ∝ k⁴ exp(-2k²/k_p²)` --- the Batchelor form, with
   `k⁴` at large scales and a peak near `k_p`.
3. If `sigma` is set, window `A ← A exp(-½(r/σ)ⁿ)`, `n = window_n`, `r`
   the (non-periodic) distance from `center`; `n = 2` is the Gaussian
   `exp(-r²/2σ²)`. This convolves the spectrum with a kernel of width
   `~n/σ`, which leaves it intact provided `k_p ≫ n/σ`. See *Window
   shape* below.
4. Curl, `û_i = i ε_ijk k_j Â_k`, with the Nyquist bins of the derivative
   zeroed.
5. Rescale so `∫ u_i u_i dx = E0` (twice the kinetic energy).

The net momentum `∫u dx` vanishes automatically: it is the `k = 0` mode of
a curl.

## Window shape

The energy density follows the square of the window, `exp(-(r/σ)ⁿ)`, so
its second moment is

    E₂ = ∫r² u_i u_i dx / ∫u_i u_i dx ≈ σ² Γ(5/n) / Γ(3/n)

(`3σ²/2` for the Gaussian), plus a share of the eddies that does not
depend on `σ`. A larger `n` gives the puff a flatter core and a sharper
edge, so more of `E₂` fits inside a given radius: with at most `10⁻³`
of the energy beyond `r = R`, the RMS radius `√E₂` reaches `0.43 R` for
`n = 2`, `0.55 R` for `n = 4` and `0.60 R` for `n = 6` (`0.77 R` is the
top-hat limit). The edge, the radii over which the envelope falls from
90% to 10% of its peak, is `1.2σ` wide at `n = 2` and `0.66σ` at
`n = 4`; it must stay wide against the eddy scale `2π/k_p`, and be
resolved: the field is solenoidal to roundoff at `n = 4` once
`σ ≳ 10Δx` (`~10⁻⁵` at `σ = 4Δx`).

## Resolution

For a puff, two requirements pull against each other. It must stay off
the box faces, `4σ ≲ H/2`, i.e. `σ ≲ N/8` cells; and the eddy scale
`ℓ = 2π/k_p` must be resolved, `ℓ ≳ 8Δx`. Eddies across the puff radius
is then `σ/ℓ ≲ N/64`: about 2 at `N = 128`, 4 at `N = 256`, 8 at
`N = 512`.

The divergence of the result is at roundoff only once the spectrum has
decayed by the per-axis Nyquist bin, where the derivative wavenumber is
zeroed and `k·(∇×A) = 0` stops being an exact identity. The window is
part of that spectrum twice over: its edge must be resolved (above), and
it must have decayed on the box faces, or the periodic copy of `A` has a
jump there --- the Gaussian is still `e^-4.5` on the faces at `σ = H/6`,
a residual of `~10⁻⁵`, where `n = 4` is at `e^-40`.

`sigma` is in physical units (same as `H`); `k_p` is a physical
wavenumber, so the eddy scale is `2π/k_p`.
"""
function turbulent_puff_ic(g::PeriodicGrid{T}; k_p::Real, sigma=nothing, window_n::Real=2,
                           center=nothing, seed::Integer=0, E0::Real=1.0) where {T}
    window_n > 0 || throw(ArgumentError("window_n must be positive, got $window_n"))
    N, H = g.N, g.H
    rng = MersenneTwister(seed)

    A = randn(rng, T, N, N, N, 3)

    kr, kf = wavenumbers(g)
    M = N ÷ 2 + 1
    k1 = reshape(kr, M, 1, 1)
    k2 = reshape(kf, 1, N, 1)
    k3 = reshape(kf, 1, 1, N)
    ksq = k1 .^ 2 .+ k2 .^ 2 .+ k3 .^ 2

    # 2. spectral filter -> E(k) ~ k^4 exp(-2k^2/k_p^2)
    Â = rfft(A, 1:3)
    Â .*= T.(exp.(-ksq ./ T(k_p)^2))
    A = irfft(Â, N, 1:3)

    # 3. localize
    if sigma !== nothing
        xs = Array(g.xs)
        c = center === nothing ? ntuple(_ -> g.origin + H / 2, 3) :
            center isa Real ? ntuple(_ -> center, 3) : Tuple(center)
        r2 = [T((xs[i] - c[1])^2 + (xs[j] - c[2])^2 + (xs[k] - c[3])^2)
              for i in 1:N, j in 1:N, k in 1:N]
        # n = 2 keeps the Gaussian's own expression, so it stays bit-identical
        w = window_n == 2 ? exp.(-r2 ./ T(2 * sigma^2)) :
            exp.(-(r2 ./ T(sigma^2)) .^ T(window_n / 2) ./ 2)
        A .*= reshape(w, N, N, N, 1)
    end

    # 4. curl -> exactly solenoidal (and exactly localized, if windowed)
    Â = rfft(A, 1:3)
    nyq(v, idx) = (w = copy(v); iseven(N) && (w[idx] = 0); w)
    d1 = reshape(nyq(kr, M), M, 1, 1)
    d2 = reshape(nyq(kf, N ÷ 2 + 1), 1, N, 1)
    d3 = reshape(nyq(kf, N ÷ 2 + 1), 1, 1, N)
    Â1, Â2, Â3 = view(Â, :, :, :, 1), view(Â, :, :, :, 2), view(Â, :, :, :, 3)
    û = cat(
        (@. im * (d2 * Â3 - d3 * Â2)),
        (@. im * (d3 * Â1 - d1 * Â3)),
        (@. im * (d1 * Â2 - d2 * Â1)),
        dims=4,
    )
    u = irfft(û, N, 1:3)

    # 5. normalize E0
    E0_now = g.dx^3 * sum(Float64, sum(abs2, u; dims=4))
    u .*= T(sqrt(E0 / E0_now))

    return adapt_to_like(g.xs, u)
end

"""
    abc_flow_ic(g; A=1, B=1, C=1, m=1, E0=nothing)

The single-mode Arnold--Beltrami--Childress flow on the grid, in `g`'s
element type and on `g`'s device,

    u = (A sin κz + C cos κy,  B sin κx + A cos κz,  C sin κy + B cos κx),

with `κ = 2πm/H`. Every mode has `|k| = κ`, and the flow is Beltrami,
`ω = κ u`, so `u × ω = 0` and the projected nonlinear term vanishes
identically.

That makes it an exact solution of the unforced equations: the field
can only decay viscously,

    u(t) = u(0) exp(-ν κ² t),

to roundoff. With `E0` set the field is rescaled so `∫ u_i u_i dx = E0`.
"""
function abc_flow_ic(g::PeriodicGrid{T}; A::Real=1, B::Real=1, C::Real=1,
                     m::Integer=1, E0=nothing) where {T}
    N, H = g.N, g.H
    κ = 2π * m / H
    xs = Array(g.xs)
    u = zeros(T, N, N, N, 3)
    for k in 1:N, j in 1:N, i in 1:N
        x, y, z = xs[i], xs[j], xs[k]
        u[i, j, k, 1] = A * sin(κ * z) + C * cos(κ * y)
        u[i, j, k, 2] = B * sin(κ * x) + A * cos(κ * z)
        u[i, j, k, 3] = C * sin(κ * y) + B * cos(κ * x)
    end
    if E0 !== nothing
        E0_now = g.dx^3 * sum(Float64, sum(abs2, u; dims=4))
        u .*= T(sqrt(E0 / E0_now))
    end
    return adapt_to_like(g.xs, u)
end
