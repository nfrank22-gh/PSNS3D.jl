"""
    PSNS3DCUDAExt

Activates `backend = :cuda`. Loaded automatically once `CUDA.jl` is in the
session; `CUDA` is a weak dependency, so a CPU-only machine never installs
it.

The solver is written against `AbstractArray` and reaches the GPU through
broadcast, `AbstractFFTs` plans (CUFFT) and `sum`/`maximum` reductions, so
there is no kernel to port and no second code path to keep in step. The
extension adds the array type and two facts about CUFFT's
complex-to-real plans: `mul!` already preserves its input (it copies it
into a plan-owned buffer), and for an input the caller no longer needs
that copy can be skipped.
"""
module PSNS3DCUDAExt

using PSNS3D
using CUDA

const cuFFT = CUDA.cuFFT

PSNS3D.arraytype(::PSNS3D.CUDABackend, ::Type{T}) where {T} = CuArray{T}

PSNS3D.c2r_preserves_input(::cuFFT.CuFFTPlan) = true

# Out-of-place complex-to-real: execute directly on the input, which cuFFT
# then overwrites, instead of on CUFFT.jl's copy of it.
function PSNS3D._c2r_scratch!(u::CuArray{T}, p::cuFFT.CuFFTPlan{T,Complex{T},K,false},
                              û::CuArray{Complex{T}}) where {T<:Real,K}
    cuFFT.assert_applicable(p, û, u)
    cuFFT.unsafe_execute_external_batches!(p, û, u)
    return u
end

end # module
