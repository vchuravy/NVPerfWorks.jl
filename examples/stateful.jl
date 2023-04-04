metrics = [
    "sm__cycles_elapsed.avg",
    "sm__cycles_elapsed.avg.per_second",

    "dram__bytes.sum",
    "lts__t_bytes.sum",
    "l1tex__t_bytes.sum",

    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",

    "sm__sass_thread_inst_executed_op_dadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_dmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_dfma_pred_on.sum",

    "sm__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_hmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_hfma_pred_on.sum",
]

using CUDA
using CUDA: i32
using NVPerfWorks

function vadd(a, b, c)
    i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    c[i] = a[i] + b[i]
    return
end

dims = (64,8)
a = CUDA.rand(Float32, dims)
b = CUDA.rand(Float32, dims)
c = similar(a)

# Warmup
@cuda threads=prod(dims) vadd(a, b, c)
CUDA.synchronize()

measure = NVPerfWorks.StatefulMeasure(metrics)

for name in metrics
    m = NVPERF.Metric(measure.mers.me, name)
    description, unit = NVPERF.properties(m)
    @show description
    @show string(unit)
end

for i in 1:10
    NVPerfWorks.start!(measure)
    CUDA.@sync @cuda threads=prod(dims) vadd(a, b, c)
    @show NVPerfWorks.stop!(measure)
end

@show measure.passes
@show measure.sessions