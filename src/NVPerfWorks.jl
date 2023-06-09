module NVPerfWorks

using CUDA
import CUDA: APIUtils, CUDA_Runtime

module Units
    using Unitful
    using Unitful.DefaultSymbols

    @refunit Instruction "Instruction" Instruction 𝐍 true
    @refunit Cycle "Cycle" Cycle 𝐍 false
    @refunit Byte "Byte" Byte 𝐍 true

    @unit IPC "IPC" InstructionsPerCycle 1Instruction/Cycle false

    # TODO: KiB

    Unitful.register(@__MODULE__);
end

import .Units

using Unitful

include("cuptiext.jl")
include("nvperf/NVPERF.jl")

export CUPTIExt, NVPERF

mutable struct StatefulMeasure
    mers::NVPERF.MetricEvalRequestSet
    image::Vector{UInt8}
    counterData::CUPTIExt.CounterData
    units
    metrics::Vector{String}
    running::Bool
    sessions::Int64
    passes::Int64
end 

function StatefulMeasure(metrics)
    NVPERF.initialize()
    CUPTIExt.initialize_profiler()

    avail = CUPTIExt.counter_availability()
    chip = first(NVPERF.supported_chips())

    me = NVPERF.CUDAMetricsEvaluator(chip, avail)

    mers = NVPERF.MetricEvalRequestSet(me, metrics)
    units = NVPERF.units(mers)
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
    return StatefulMeasure(mers, image, counterData, units, metrics, false, 0, 0)
end

function start!(measure::StatefulMeasure)
    if !measure.running
        CUPTIExt.begin_session(measure.counterData, CUPTIExt.CUPTI_UserRange, CUPTIExt.CUPTI_UserReplay)
        CUPTIExt.set_config(measure.image)
        measure.running = true
        measure.sessions += 1
    end
    measure.passes += 1
    CUPTIExt.begin_pass()
    CUPTIExt.enable_profiling()
    CUPTIExt.push_range("metrics")
end

function stop!(measure::StatefulMeasure)
    @assert measure.running

    CUPTIExt.pop_range()
    CUPTIExt.disable_profiling()
    done = CUPTIExt.end_pass()

    if done
        measure.running = false
        CUPTIExt.flush_counter_data()
        CUPTIExt.unset_config()
        CUPTIExt.end_session()

        NVPERF.set_device_attributes(measure.mers.me, measure.counterData.image)
        measured_metrics = NVPERF.evaluate(measure.mers, measure.counterData.image, 0)
        return Dict(m => val*u for (val, m, u) in zip(measured_metrics, measure.metrics, measure.units))
    end
    return nothing
end



end # module NVPerfWorks
