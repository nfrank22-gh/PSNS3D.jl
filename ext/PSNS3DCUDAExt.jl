"""
    PSNS3DCUDAExt

Activates `backend = :cuda`. Loaded automatically once `CUDA.jl` is in the
session; `CUDA` is a weak dependency, so a CPU-only machine never installs
it.

The whole extension is one method. The solver is written against
`AbstractArray` and reaches the GPU through broadcast, `AbstractFFTs`
plans (CUFFT) and `sum`/`maximum` reductions, so there is no kernel to
port and no second code path to keep in step.
"""
module PSNS3DCUDAExt

using PSNS3D
using CUDA

PSNS3D.arraytype(::PSNS3D.CUDABackend, ::Type{T}) where {T} = CuArray{T}

end # module
