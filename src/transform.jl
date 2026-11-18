"""
    SpectralPlan(g)

Cached real-transform plans over the three spatial axes of a velocity
array of size `(N, N, N, 3)`, together with the normalization `N³` that
puts coefficients in the convention of the array contract (see
[`PSNS3D`](@ref)),

    û = RFFT(u) / N³.

The trailing component axis is not transformed. `fwd` is the forward
`rfft` plan and `inv` the *unnormalized* backward plan (`brfft`): the
contract's `1/N³` sits on the forward side, so the inverse needs no
scaling at all. `buf` is a spectral scratch array that
[`inverse_transform!`](@ref) copies its input into on backends whose
complex-to-real transform overwrites its input (FFTW does; see
[`c2r_preserves_input`](@ref)).
"""
struct SpectralPlan{T,PF,PI,B}
    N::Int
    norm::T
    fwd::PF
    inv::PI
    buf::B
end

function SpectralPlan(g::PeriodicGrid{T}) where {T}
    u = allocate_state(g)
    fill!(u, zero(T))
    fwd = plan_rfft(u, 1:3)
    û = fwd * u
    inv = plan_brfft(û, g.N, 1:3)
    return SpectralPlan{T,typeof(fwd),typeof(inv),typeof(û)}(g.N, T(g.N)^3, fwd, inv, û)
end

"""
    c2r_preserves_input(plan)

Whether `mul!(u, plan, û)` with a complex-to-real `plan` leaves `û`
intact. FFTW's multi-dimensional complex-to-real transforms may overwrite
their input, so the default is `false`; CUFFT.jl copies the input into a
buffer of its own first, so the CUDA extension returns `true` for its
plans.
"""
c2r_preserves_input(::Any) = false

"""
    forward_transform(u, p)
    forward_transform!(û, u, p)

The normalized transform `û = RFFT(u) / N³`. The in-place form writes
into `û`, of size `(N÷2+1, N, N, 3)`, and leaves `u` intact; both give
bit-identical results.
"""
forward_transform(u::AbstractArray, p::SpectralPlan) = (p.fwd * u) ./ p.norm

function forward_transform!(û::AbstractArray, u::AbstractArray, p::SpectralPlan)
    mul!(û, p.fwd, u)
    û ./= p.norm
    return û
end

"""
    inverse_transform(û, p)
    inverse_transform!(u, û, p)

The inverse of [`forward_transform`](@ref). The in-place form writes into
`u` and leaves `û` intact on every backend: where the transform would
overwrite its input (see [`c2r_preserves_input`](@ref)), `û` is first
copied into the plan's scratch buffer.
"""
inverse_transform(û::AbstractArray, p::SpectralPlan) = p.inv * û

function inverse_transform!(u::AbstractArray, û::AbstractArray, p::SpectralPlan)
    if c2r_preserves_input(p.inv)
        mul!(u, p.inv, û)
    else
        copyto!(p.buf, û)
        mul!(u, p.inv, p.buf)
    end
    return u
end

"""
    inverse_transform_scratch!(u, û, p)

[`inverse_transform!`](@ref) for a `û` the caller no longer needs: it may
be overwritten, which saves the copy that preserving it would cost. The
CUDA extension also skips CUFFT.jl's own input copy here.
"""
inverse_transform_scratch!(u::AbstractArray, û::AbstractArray, p::SpectralPlan) =
    (_c2r_scratch!(u, p.inv, û); u)

_c2r_scratch!(u, plan, û) = mul!(u, plan, û)
