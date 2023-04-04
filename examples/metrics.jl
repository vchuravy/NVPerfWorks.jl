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

NVPERF.initialize()
CUPTIExt.initialize_profiler()

avail = CUPTIExt.counter_availability()
chip = first(NVPERF.supported_chips())

me = NVPERF.CUDAMetricsEvaluator(chip, avail)

# NVPERF.list_metrics(me)

m = NVPERF.Metric(me, "dram__bytes.sum.per_second")
description, unit = NVPERF.properties(m)
@show description
@show string(unit)

mers = NVPERF.MetricEvalRequestSet(me,[
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
])
raw_metrics = NVPERF.raw(mers)

metricsConfig = NVPERF.CUDARawMetricsConfig(chip, avail; activity=NVPERF.NVPA_ACTIVITY_KIND_REALTIME_PROFILER)
NVPERF.begin_config_group(metricsConfig, 1)
NVPERF.add!(metricsConfig, raw_metrics)
NVPERF.end_config_group(metricsConfig)
NVPERF.generate(metricsConfig)

image = NVPERF.config_image(metricsConfig)

builder = NVPERF.CUDACounterDataBuilder(chip, avail)
NVPERF.add!(builder, raw_metrics)
prefix = NVPERF.prefix(builder)

counterData = CUPTIExt.CounterData(prefix, 1, 1, 64)

function measure(f, counterData, image)
    CUPTIExt.begin_session(counterData, CUPTIExt.CUPTI_UserRange, CUPTIExt.CUPTI_UserReplay)
    CUPTIExt.set_config(image)
    while true
        @info "Running pass"
        CUPTIExt.begin_pass()
        CUPTIExt.enable_profiling()
        CUPTIExt.push_range("metrics")

        f()

        CUPTIExt.pop_range()
        CUPTIExt.disable_profiling()
        if CUPTIExt.end_pass()
            break
        end
    end
    CUPTIExt.flush_counter_data()
    CUPTIExt.unset_config()
    CUPTIExt.end_session()
end

measure(counterData, image) do
    @cuda threads=prod(dims) vadd(a, b, c)
    CUDA.synchronize()
end

# @show NVPERF.get_num_ranges(image)

# meval = NVPERF.CUDAMetricsEvaluator(chip, avail, counterData.image)
NVPERF.set_device_attributes(mers.me, counterData.image)
measured = NVPERF.evaluate(mers, counterData.image, 0)
@show measured


# TODO Eval