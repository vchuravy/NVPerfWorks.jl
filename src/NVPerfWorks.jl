module NVPerfWorks

using CUDA
import CUDA: APIUtils, CUDA_Runtime

include("nvperf/NVPERF.jl")
include("cuptiext.jl")

export CUPTIExt, NVPERF

end # module NVPerfWorks
