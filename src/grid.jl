"""
    PeriodicGrid{T}(H, N; backend=CPUBackend(), origin=-H/2)
    PeriodicGrid(H, N; ...)                      # T = Float64

A cubic periodic box of side `H` with `N` points per direction, spacing
`dx = H/N`, and grid points

    x_n = origin + n dx,   n = 0, …, N-1

along each axis. The default `origin = -H/2` centres the box on zero,
`x ∈ [-H/2, H/2)`; `origin = 0` gives the usual `[0, H)`. Only the
initial conditions and physical-space forcings read the coordinates ---
the spectral operators see `H` alone.

Carries the length-`N` coordinate vector `xs` (the same in every
direction) on the chosen backend's device. `xs` doubles as the device
reference that [`adapt_to_like`](@ref) moves host arrays next to.
"""
struct PeriodicGrid{T<:Real,V<:AbstractVector{T}}
    H::T
    N::Int
    dx::T
    origin::T
    xs::V
end

function PeriodicGrid{T}(H::Real, N::Integer; backend::Backend=CPUBackend(),
                         origin::Real=-H / 2) where {T<:Real}
    H = convert(T, H)
    origin = convert(T, origin)
    dx = H / N
    xs = adapt_to(backend, T[origin + n * dx for n in 0:(N-1)])
    return PeriodicGrid(H, Int(N), dx, origin, xs)
end
PeriodicGrid(H::Real, N::Integer; kw...) = PeriodicGrid{Float64}(H, N; kw...)

Base.eltype(::PeriodicGrid{T}) where {T} = T

"""
    wavenumbers(g)

The represented wavenumbers `k_j = 2πj/H` along one axis, in the two
layouts the real transform produces:

  - `kr`, length `N÷2+1`, for the transformed (first) axis: `j = 0, …, N÷2`.
  - `kf`, length `N`, for the other two axes, in FFT bin order: `j` runs
    `0, 1, …, N÷2, -(N÷2-1), …, -1` for even `N`.

Returned on the host as plain `Vector`s; callers move them to the device.
"""
function wavenumbers(g::PeriodicGrid{T}) where {T}
    N, H = g.N, g.H
    kr = T[2π * j / H for j in 0:(N÷2)]
    kf = T[2π * (j <= N ÷ 2 ? j : j - N) / H for j in 0:(N-1)]
    return kr, kf
end

"""
    signedbins(N)

The signed Fourier index `j` of each bin of a length-`N` complex FFT axis,
in bin order: `0, 1, …, N÷2, -(N÷2-1), …, -1` for even `N`.
"""
signedbins(N::Integer) = [j <= N ÷ 2 ? j : j - N for j in 0:(N-1)]

"""
    allocate_state(g)

An uninitialized velocity array of size `(N, N, N, 3)` on the same device
and element type as `g`.
"""
allocate_state(g::PeriodicGrid{T}) where {T} = similar(g.xs, T, g.N, g.N, g.N, 3)
