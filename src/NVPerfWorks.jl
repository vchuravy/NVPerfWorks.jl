module NVPerfWorks

using CUDA
import CUDA: APIUtils, CUDA_Runtime

include("cuptiext.jl")
include("nvperf/NVPERF.jl")

export CUPTIExt, NVPERF

end # module NVPerfWorks
