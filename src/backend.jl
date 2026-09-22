"""
    CPUBackend
    CUDABackend

Selects the array type that a run's fields live on. The whole solver is
written against `AbstractArray`, so the backend is nothing more than which
constructor built the state; the solver never mentions CUDA.

`CUDABackend` is only usable once `CUDA.jl` is loaded, which activates the
package extension in `ext/PSNS3DCUDAExt.jl`. `CUDA` sits in `[weakdeps]`,
so a CPU-only machine never installs it.
"""
abstract type Backend end
struct CPUBackend <: Backend end
struct CUDABackend <: Backend end

"""
    backend(sym)

`:cpu`/`:cuda` to a [`Backend`](@ref).
"""
function backend(sym::Symbol)
    sym === :cpu && return CPUBackend()
    sym === :cuda && return CUDABackend()
    throw(ArgumentError("backend must be :cpu or :cuda, got :$sym"))
end
backend(b::Backend) = b

"""
    arraytype(backend, T)

The concrete array constructor for element type `T` on `backend`. The
`CUDABackend` method lives in the CUDA extension and errors with a usable
message if `CUDA.jl` has not been loaded.

The fallback is declared on `Backend` rather than on `CUDABackend`: a
method here with the *same* signature as the extension's would be
overwritten rather than extended when the extension loads, and Julia
forbids that during precompilation --- which fails the extension and
leaves this error in place on a machine that does have a GPU.
"""
arraytype(::CPUBackend, ::Type{T}) where {T} = Array{T}
arraytype(b::Backend, ::Type{T}) where {T} = error(
    "no array type for $(nameof(typeof(b))). backend=:cuda requires " *
    "CUDA.jl to be loaded. Run `using CUDA` first (CUDA is a weak " *
    "dependency, so it is not loaded automatically)."
)

"""
    adapt_to(backend, A)

Move `A` onto `backend`'s array type, preserving element type.
"""
adapt_to(b::Backend, A::AbstractArray{T}) where {T} = arraytype(b, T)(A)
adapt_to(::CPUBackend, A::Array) = A

"""
    adapt_to_like(ref, A)

`A` on the same device as `ref`, preserving `A`'s element type. Lets the
workspace build wavenumber and mask arrays without naming a device array
type.
"""
adapt_to_like(ref::AbstractArray, A::AbstractArray) =
    copyto!(similar(ref, eltype(A), size(A)), A)
adapt_to_like(::Array, A::AbstractArray) = A
