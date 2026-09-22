"""
    NSWorkspace(g)
    workspace(prob)

Everything built once per run: the transform plans, the wavenumber
arrays and the dealiasing mask.

The wavenumbers are stored as three *broadcastable* arrays of shape
`(M,1,1)`, `(1,N,1)`, `(1,1,N)` with `M = N÷2+1`, rather than three full
`(M,N,N)` arrays: broadcasting expands them on the fly, which costs no
memory and no bandwidth. `kd` is the derivative wavenumber, identical to
`k` except that the Nyquist bin is zeroed --- a real field's Nyquist mode
has no representable derivative, and leaving it in would inject a
spurious imaginary part.

`mask` is the 2/3 dealiasing rule: modes with `|j_d| > N/3` in any
direction are zeroed. Besides removing the aliases of the pseudo-spectral
product, it leaves a band of empty modes between `N/3` and the Nyquist
`N/2`, which a caller that stretches the spectrum (a dilation `k → ak`
with `a < 3/2`) can use as a buffer against wrap-around.

`kmax = 2π(N/3)/H` is the top wavenumber surviving the mask.
"""
struct NSWorkspace{G,P,A1,A2,A3,AM,AS}
    grid::G
    plan::P
    k1::A1
    k2::A2
    k3::A3
    kd1::A1
    kd2::A2
    kd3::A3
    ksq::AS
    mask::AM
    kmax::Float64
end

function NSWorkspace(g::PeriodicGrid{T}) where {T}
    N = g.N
    kr, kf = wavenumbers(g)
    M = N ÷ 2 + 1

    nyq(v, idx) = (w = copy(v); iseven(N) && (w[idx] = 0); w)
    krd = nyq(kr, M)
    kfd = nyq(kf, N ÷ 2 + 1)

    dev(v, shape) = adapt_to_like(g.xs, reshape(v, shape))
    k1 = dev(kr, (M, 1, 1));  k2 = dev(kf, (1, N, 1));  k3 = dev(kf, (1, 1, N))
    kd1 = dev(krd, (M, 1, 1)); kd2 = dev(kfd, (1, N, 1)); kd3 = dev(kfd, (1, 1, N))

    ksq = k1 .^ 2 .+ k2 .^ 2 .+ k3 .^ 2

    cut = N ÷ 3
    jr = [j for j in 0:(N÷2)]
    jf = signedbins(N)
    m_h = [abs(a) <= cut && abs(b) <= cut && abs(cc) <= cut
           for a in jr, b in jf, cc in jf]
    mask = adapt_to_like(g.xs, T.(m_h))

    kmax = 2π * cut / g.H
    return NSWorkspace(g, SpectralPlan(g), k1, k2, k3, kd1, kd2, kd3, ksq, mask, kmax)
end

_comp(A, i) = view(A, :, :, :, i)
