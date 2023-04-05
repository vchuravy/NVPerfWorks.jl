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

# for name in metrics
#     m = NVPERF.Metric(measure.mers.me, name)
#     description, unit = NVPERF.properties(m)
#     @show description
#     @show string(unit)
# end

function process(metrics)
    time = metrics["sm__cycles_elapsed.avg"] / metrics["sm__cycles_elapsed.avg.per_second"]
    D_FLOP = 2*metrics["sm__sass_thread_inst_executed_op_dfma_pred_on.sum"] + 
               metrics["sm__sass_thread_inst_executed_op_dmul_pred_on.sum"] +
               metrics["sm__sass_thread_inst_executed_op_dadd_pred_on.sum"]
    F_FLOP = 2*metrics["sm__sass_thread_inst_executed_op_ffma_pred_on.sum"] + 
               metrics["sm__sass_thread_inst_executed_op_fmul_pred_on.sum"] +
               metrics["sm__sass_thread_inst_executed_op_fadd_pred_on.sum"]
    H_FLOP = 2*metrics["sm__sass_thread_inst_executed_op_hfma_pred_on.sum"] + 
               metrics["sm__sass_thread_inst_executed_op_hmul_pred_on.sum"] +
               metrics["sm__sass_thread_inst_executed_op_hadd_pred_on.sum"]

    AI_D_DRAM = D_FLOP / metrics["dram__bytes.sum"]
    AI_F_DRAM = F_FLOP / metrics["dram__bytes.sum"]
    AI_H_DRAM = H_FLOP / metrics["dram__bytes.sum"]

    AI_D_L2 = D_FLOP / metrics["lts__t_bytes.sum"]
    AI_F_L2 = F_FLOP / metrics["lts__t_bytes.sum"]
    AI_H_L2 = H_FLOP / metrics["lts__t_bytes.sum"]

    AI_D_L1 = D_FLOP / metrics["l1tex__t_bytes.sum"]
    AI_F_L1 = F_FLOP / metrics["l1tex__t_bytes.sum"]
    AI_H_L1 = H_FLOP / metrics["l1tex__t_bytes.sum"]

    D_FLOPs = D_FLOP/time
    H_FLOPs = H_FLOP/time
    F_FLOPs = F_FLOP/time

    @info "Fraction active"  fraction = measure.passes/measure.sessions
    @info "Kernel performance" time D_FLOP F_FLOP H_FLOP D_FLOPs F_FLOPs H_FLOPs
    @info "Arithmetic intensity (DRAM)" AI_D_DRAM AI_F_DRAM AI_H_DRAM
    @info "Arithmetic intensity (L2)" AI_D_L2 AI_F_L2 AI_H_L2
    @info "Arithmetic intensity (L1)" AI_D_L1 AI_F_L1 AI_H_L1
end

measure = NVPerfWorks.StatefulMeasure(metrics)
for i in 1:10
    NVPerfWorks.start!(measure)
    CUDA.@sync @cuda threads=prod(dims) vadd(a, b, c)
    measured_metrics = NVPerfWorks.stop!(measure)
    if measured_metrics === nothing
        continue
    end
    process(measured_metrics)
end
