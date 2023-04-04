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

mer = NVPERF.MetricEvalRequest(me, "dram__bytes.sum.per_second")
mers = NVPERF.MetricEvalRequestSet(me, [mer])
raw_metrics = NVPERF.raw(mers)

metricsConfig = NVPERF.CUDARawMetricsConfig(chip, avail)#; activity=NVPERF.NVPA_ACTIVITY_KIND_REALTIME_PROFILER)
NVPERF.begin_config_group(metricsConfig, 1)
NVPERF.add!(metricsConfig, raw_metrics)
NVPERF.end_config_group(metricsConfig)
NVPERF.generate(metricsConfig)

image = NVPERF.config_image(metricsConfig)

builder = NVPERF.CUDACounterDataBuilder(chip, avail)
NVPERF.add!(builder, raw_metrics)
prefix = NVPERF.prefix(builder)

counterData = CUPTIExt.CounterData(prefix, 1, 1, 64)

CUPTIExt.begin_session(counterData, CUPTIExt.CUPTI_UserRange, CUPTIExt.CUPTI_UserReplay)
CUPTIExt.set_config(image)

CUPTIExt.begin_pass()
CUPTIExt.enable_profiling()
CUPTIExt.push_range("metrics")

@cuda threads=prod(dims) vadd(a, b, c)

CUPTIExt.pop_range()
CUPTIExt.disable_profiling()
@show CUPTIExt.end_pass()

@show CUPTIExt.flush_counter_data()
CUPTIExt.unset_config()
CUPTIExt.end_session()

# @show NVPERF.get_num_ranges(image)

meval = NVPERF.CUDAMetricsEvaluator(chip, avail, counterData.image)

# TODO Eval