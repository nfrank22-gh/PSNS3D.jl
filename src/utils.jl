"""
    curl(û, ws)

`ω = ∇ × u` in physical space, from the spectral velocity `û`. Uses the
derivative wavenumbers `kd` (Nyquist zeroed). Three inverse transforms.
"""
function curl(û, ws::NSWorkspace)
    û1, û2, û3 = _comp(û, 1), _comp(û, 2), _comp(û, 3)
    ŵ1 = @. im * (ws.kd2 * û3 - ws.kd3 * û2)
    ŵ2 = @. im * (ws.kd3 * û1 - ws.kd1 * û3)
    ŵ3 = @. im * (ws.kd1 * û2 - ws.kd2 * û1)
    return inverse_transform(cat(ŵ1, ŵ2, ŵ3; dims=4), ws.plan)
end

"""
    leray_project!(u, ws)

The Leray projection of a physical-space velocity, in place:
`û_i ← P_il(k) û_l` with `P_il = δ_il - k_i k_l / k²`, then back to
physical space. Returns `u`.

Uses `k`, not the derivative `kd`, so that [`divergence_residual`](@ref)
--- which measures `k_i û_i` --- lands at roundoff. That leaves the Nyquist
bins: on the self-conjugate planes of the real transform, `P(k)` is not
Hermitian-symmetric about the stored `-N/2` wavenumber, so a projected
Nyquist mode does not survive the inverse transform intact. Those bins
have no representable derivative anyway, so they are zeroed outright. The
2/3 band is *not* imposed.
"""
function leray_project!(u::AbstractArray, ws::NSWorkspace)
    û = forward_transform(u, ws.plan)
    û1, û2, û3 = _comp(û, 1), _comp(û, 2), _comp(û, 3)
    kdotu = @. ws.k1 * û1 + ws.k2 * û2 + ws.k3 * û3
    # See `project` for why `unit` is hoisted out of the `@.`.
    unit = one(eltype(ws.ksq))
    safe = @. ifelse(ws.ksq == 0, unit, ws.ksq)
    keep = @. (ws.k1 == ws.kd1) & (ws.k2 == ws.kd2) & (ws.k3 == ws.kd3)
    @. û1 = ifelse(keep, û1 - ws.k1 * kdotu / safe, zero(û1))
    @. û2 = ifelse(keep, û2 - ws.k2 * kdotu / safe, zero(û2))
    @. û3 = ifelse(keep, û3 - ws.k3 * kdotu / safe, zero(û3))
    u .= inverse_transform(û, ws.plan)
    return u
end

"""
    divergence_residual(u, g)

`‖k_i û_i‖ / ‖k‖‖û‖`, a scale-free measure of how far `u` is from
divergence-free. Computed on the host, so it needs no workspace.

The [`spectral_divergence_residual`](@ref) method takes the transform
directly, for callers that already hold `û`.
"""
divergence_residual(u::AbstractArray, g::PeriodicGrid) =
    spectral_divergence_residual(rfft(Array(u), 1:3), g)

"""
    spectral_divergence_residual(û, g)

[`divergence_residual`](@ref) on a field that has already been
transformed; `û` is `rfft(u, 1:3)` (any normalization), of size
`(N÷2+1, N, N, 3)`.
"""
function spectral_divergence_residual(û::AbstractArray, g::PeriodicGrid)
    N = g.N
    kr, kf = wavenumbers(g)
    M = N ÷ 2 + 1
    k1 = reshape(kr, M, 1, 1); k2 = reshape(kf, 1, N, 1); k3 = reshape(kf, 1, 1, N)
    û1, û2, û3 = view(û, :, :, :, 1), view(û, :, :, :, 2), view(û, :, :, :, 3)
    d = @. k1 * û1 + k2 * û2 + k3 * û3
    scale = sqrt(sum(abs2, û) * maximum(k1 .^ 2 .+ k2 .^ 2 .+ k3 .^ 2))
    return scale == 0 ? 0.0 : sqrt(sum(abs2, d)) / scale
end

"""
    energy_spectrum(u, g)

The shell-averaged energy spectrum `E(k)`, i.e. `½Σ|û|²` binned by `|k|`
and divided by the bin width, so that `∫E(k)dk = ½∫u_iu_i dx / H³`.
Returned as `(k_centres, E)`. Computed on the host.
"""
function energy_spectrum(u::AbstractArray, g::PeriodicGrid)
    N, H = g.N, g.H
    kr, kf = wavenumbers(g)
    M = N ÷ 2 + 1
    û = Array(rfft(Array(u), 1:3)) ./ N^3

    dk = 2π / H
    nbin = N ÷ 2
    E = zeros(Float64, nbin)
    for c in 1:3, k3 in 1:N, k2 in 1:N, k1 in 1:M
        kmag = sqrt(kr[k1]^2 + kf[k2]^2 + kf[k3]^2)
        b = clamp(round(Int, kmag / dk), 1, nbin)
        w = (k1 == 1 || (iseven(N) && k1 == M)) ? 1.0 : 2.0   # Hermitian pairs
        E[b] += 0.5 * w * abs2(û[k1, k2, k3, c])
    end
    return [(b - 0.5) * dk for b in 1:nbin], E ./ dk
end
