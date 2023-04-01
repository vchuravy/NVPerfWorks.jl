using CUDA
using NVPerfWorks

NVPERF.initialize()
CUPTIExt.initialize_profiler()

avail = CUPTIExt.counter_availability()
chip = first(NVPERF.supported_chips())

me = NVPERF.CUDAMetricsEvaluator(chip, avail)

NVPERF.list_metrics(me)

m = NVPERF.Metric(me, "dram__bytes.sum.per_second")
description, unit = NVPERF.properties(m)
@show description
@show string(unit)

mer = NVPERF.MetricEvalRequest(me, "dram__bytes.sum.per_second")
mers = NVPERF.MetricEvalRequestSet(me, [mer])
raw_metrics = NVPERF.raw(mers)

metricsConfig = NVPERF.CUDARawMetricsConfig(chip, avail; activity=NVPERF.NVPA_ACTIVITY_KIND_REALTIME_PROFILER)
NVPERF.begin_config_group(metricsConfig, 1)
NVPERF.add!(metricsConfig, raw_metrics)
NVPERF.end_config_group(metricsConfig)
NVPERF.generate(metricsConfig)

image = NVPERF.config_image(metricsConfig)

# Need counterDataImage
# then range then 