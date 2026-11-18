"""
    curl(û, ws)
    curl!(ω, û, ws, ŝ)

`ω = ∇ × u` in physical space, from the spectral velocity `û`. Uses the
derivative wavenumbers `kd` (Nyquist zeroed). Three inverse transforms.
`curl!` writes into `ω`, `(N, N, N, 3)`, using the spectral `ŝ` as
scratch.
"""
function curl!(ω, û, ws::NSWorkspace, ŝ)
    û1, û2, û3 = _comp(û, 1), _comp(û, 2), _comp(û, 3)
    ŵ1, ŵ2, ŵ3 = _comp(ŝ, 1), _comp(ŝ, 2), _comp(ŝ, 3)
    @. ŵ1 = im * (ws.kd2 * û3 - ws.kd3 * û2)
    @. ŵ2 = im * (ws.kd3 * û1 - ws.kd1 * û3)
    @. ŵ3 = im * (ws.kd1 * û2 - ws.kd2 * û1)
    return inverse_transform_scratch!(ω, ŝ, ws.plan)
end
curl(û, ws::NSWorkspace) = curl!(allocate_state(ws.grid), û, ws, similar(û))

"""
    leray_project!(u, ws; uhat=nothing, scratch=nothing)

The Leray projection of a physical-space velocity, in place:
`û_i ← P_il(k) û_l` with `P_il = δ_il - k_i k_l / k²`, then back to
physical space. Returns `u`.

`uhat`, a spectral array `(N÷2+1, N, N, 3)`, receives the projected
transform, so on return it is the normalized transform of the new `u`
(to roundoff) and a caller can reuse it. `scratch`, `(N÷2+1, N, N)`
complex, holds `k·û`. Either is allocated if not given.

Uses `k`, not the derivative `kd`, so that [`divergence_residual`](@ref)
--- which measures `k_i û_i` --- lands at roundoff. That leaves the Nyquist
bins: on the self-conjugate planes of the real transform, `P(k)` is not
Hermitian-symmetric about the stored `-N/2` wavenumber, so a projected
Nyquist mode does not survive the inverse transform intact. Those bins
have no representable derivative anyway, so they are zeroed outright. The
2/3 band is *not* imposed.
"""
function leray_project!(u::AbstractArray, ws::NSWorkspace; uhat=nothing, scratch=nothing)
    û = uhat === nothing ? similar(ws.plan.buf, size(ws.mask)..., 3) : uhat
    kdotu = scratch === nothing ? similar(û, size(ws.mask)) : scratch
    forward_transform!(û, u, ws.plan)
    û1, û2, û3 = _comp(û, 1), _comp(û, 2), _comp(û, 3)
    @. kdotu = ws.k1 * û1 + ws.k2 * û2 + ws.k3 * û3
    # See `project!` for why `unit` is hoisted out of the `@.`.
    unit = one(eltype(ws.ksq))
    @. û1 = ifelse((ws.k1 == ws.kd1) & (ws.k2 == ws.kd2) & (ws.k3 == ws.kd3),
                   û1 - ws.k1 * kdotu / ifelse(ws.ksq == 0, unit, ws.ksq), zero(û1))
    @. û2 = ifelse((ws.k1 == ws.kd1) & (ws.k2 == ws.kd2) & (ws.k3 == ws.kd3),
                   û2 - ws.k2 * kdotu / ifelse(ws.ksq == 0, unit, ws.ksq), zero(û2))
    @. û3 = ifelse((ws.k1 == ws.kd1) & (ws.k2 == ws.kd2) & (ws.k3 == ws.kd3),
                   û3 - ws.k3 * kdotu / ifelse(ws.ksq == 0, unit, ws.ksq), zero(û3))
    inverse_transform!(u, û, ws.plan)
    return u
end

"""
    divergence_residual(u, g)

`‖k_i û_i‖ / ‖k‖‖û‖`, a scale-free measure of how far `u` is from
divergence-free. Needs no workspace, and runs on `u`'s device.

The [`spectral_divergence_residual`](@ref) method takes the transform
directly, for callers that already hold `û`.
"""
divergence_residual(u::AbstractArray, g::PeriodicGrid) =
    spectral_divergence_residual(rfft(u, 1:3), g)

"""
    spectral_divergence_residual(û, g)

[`divergence_residual`](@ref) on a field that has already been
transformed; `û` is `rfft(u, 1:3)` (any normalization), of size
`(N÷2+1, N, N, 3)`, on any device.
"""
function spectral_divergence_residual(û::AbstractArray, g::PeriodicGrid)
    k1, k2, k3 = _wavenumbers_like(û, g)
    û1, û2, û3 = view(û, :, :, :, 1), view(û, :, :, :, 2), view(û, :, :, :, 3)
    d = @. k1 * û1 + k2 * û2 + k3 * û3
    scale = sqrt(sum(abs2, û) * maximum(k1 .^ 2 .+ k2 .^ 2 .+ k3 .^ 2))
    return scale == 0 ? 0.0 : sqrt(sum(abs2, d)) / scale
end

"""The wavenumbers of `g` as broadcastable `(M,1,1)`, `(1,N,1)`, `(1,1,N)` arrays on `ref`'s device."""
function _wavenumbers_like(ref::AbstractArray, g::PeriodicGrid)
    N = g.N
    kr, kf = wavenumbers(g)
    M = N ÷ 2 + 1
    dev(v, shape) = adapt_to_like(ref, reshape(v, shape))
    return dev(kr, (M, 1, 1)), dev(kf, (1, N, 1)), dev(kf, (1, 1, N))
end

"""
    energy_spectrum(u, g; uhat=nothing)

The shell-averaged energy spectrum `E(k)`, i.e. `½Σ|û|²` binned by `|k|`
and divided by the bin width, so that `∫E(k)dk = ½∫u_iu_i dx / H³`.
Returned as `(k_centres, E)`.

The per-mode energies and their shells are formed on `u`'s device and
only the binning is done on the host, from one array of each. `uhat`,
if given, is the normalized transform `rfft(u)/N³` (as
[`forward_transform`](@ref) returns it) and saves transforming `u`.
"""
function energy_spectrum(u::AbstractArray, g::PeriodicGrid; uhat=nothing)
    N, H = g.N, g.H
    û = uhat === nothing ? rfft(u, 1:3) ./ N^3 : uhat
    k1, k2, k3 = _wavenumbers_like(û, g)
    M = N ÷ 2 + 1
    dk = 2π / H
    nbin = N ÷ 2
    j1 = adapt_to_like(û, reshape(collect(0:(M-1)), M, 1, 1))
    û1, û2, û3 = view(û, :, :, :, 1), view(û, :, :, :, 2), view(û, :, :, :, 3)
    # Hermitian pairs: every bin of the halved axis but j₁ = 0 and the
    # even-N Nyquist stands for itself and its conjugate.
    nyq = iseven(N) ? M - 1 : -1
    e = @. ifelse((j1 == 0) | (j1 == nyq), 0.5, 1.0) *
           (Float64(abs2(û1)) + Float64(abs2(û2)) + Float64(abs2(û3)))
    b = @. clamp(round(Int32, sqrt(Float64(k1)^2 + Float64(k2)^2 + Float64(k3)^2) / dk),
                 Int32(1), Int32(nbin))
    eh, bh = Array(e), Array(b)
    E = zeros(Float64, nbin)
    @inbounds for i in eachindex(eh, bh)
        E[bh[i]] += eh[i]
    end
    return [(b - 0.5) * dk for b in 1:nbin], E ./ dk
end
