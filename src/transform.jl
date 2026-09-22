"""
    SpectralPlan(g)

Cached real-transform plans over the three spatial axes of a velocity
array of size `(N, N, N, 3)`, together with the normalization `N³` that
puts coefficients in the convention of the array contract (see
[`PSNS3D`](@ref)),

    û = RFFT(u) / N³.

The trailing component axis is not transformed.
"""
struct SpectralPlan{T,PF,PI}
    N::Int
    norm::T
    fwd::PF
    inv::PI
end

function SpectralPlan(g::PeriodicGrid{T}) where {T}
    u = allocate_state(g)
    fill!(u, zero(T))
    fwd = plan_rfft(u, 1:3)
    û = fwd * u
    inv = plan_irfft(û, g.N, 1:3)
    return SpectralPlan{T,typeof(fwd),typeof(inv)}(g.N, T(g.N)^3, fwd, inv)
end

"""
    forward_transform(u, p)

The normalized transform `û = RFFT(u) / N³`.
"""
forward_transform(u::AbstractArray, p::SpectralPlan) = (p.fwd * u) ./ p.norm

"""
    inverse_transform(û, p)

The inverse of [`forward_transform`](@ref).
"""
inverse_transform(û::AbstractArray, p::SpectralPlan) = p.inv * (û .* p.norm)
