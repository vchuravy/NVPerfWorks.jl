function initialize()
    params = Ref(NVPW_InitializeHost_Params(NVPW_InitializeHost_Params_STRUCT_SIZE, C_NULL))
    NVPW_InitializeHost(params)
end

function supported_chips()
    params = Ref(NVPW_GetSupportedChipNames_Params(
        NVPW_GetSupportedChipNames_Params_STRUCT_SIZE,
        C_NULL, C_NULL, 0))
    NVPW_GetSupportedChipNames(params)

    names = String[]
    for i in params[].numChipNames
        push!(names, Base.unsafe_string(Base.unsafe_load(params[].ppChipNames, i)))
    end
    return names
end

function scratch_buffer(chipName, counter_availability)
    GC.@preserve chipName counter_availability begin
        params = Ref(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params(
            NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE,
            C_NULL, pointer(chipName), pointer(counter_availability), 0
        ))
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(params)
        sz = params[].scratchBufferSize
    end
    return Vector{UInt8}(undef, sz)
end

abstract type MetricsEvaluator end

mutable struct CUDAMetricsEvaluator <: MetricsEvaluator
    handle::Ptr{NVPW_MetricsEvaluator}
    scratch::Vector{UInt8}
    availability::Vector{UInt8}
    chip::String

    function CUDAMetricsEvaluator(chip, availability)
        scratch = scratch_buffer(chip, availability)

        GC.@preserve chip availability scratch begin
            params = Ref(NVPW_CUDA_MetricsEvaluator_Initialize_Params(
                NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE,
                C_NULL, pointer(scratch), length(scratch), pointer(chip),
                pointer(availability), C_NULL, 0, C_NULL))
            
            NVPW_CUDA_MetricsEvaluator_Initialize(params)
            this =  new(params[].pMetricsEvaluator, scratch, availability, chip)
        end
        finalizer(destroy, this)
        return this
    end
end
Base.unsafe_convert(::Type{Ptr{NVPW_MetricsEvaluator}}, me::CUDAMetricsEvaluator) = me.handle


function destroy(me::MetricsEvaluator)
    GC.@preserve me begin
        params = Ref(NVPW_MetricsEvaluator_Destroy_Params(
            NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, me)
        ))
        NVPW_MetricsEvaluator_Destroy(params)
    end
    return nothing
end

struct MetricsIterator
    me::MetricsEvaluator
    type::NVPW_MetricType
    names::Ptr{Cchar}
    indices::Ptr{Csize_t}
    numMetrics::Csize_t

    function MetricsIterator(me, type)
        GC.@preserve me begin
            params = Ref(NVPW_MetricsEvaluator_GetMetricNames_Params(
                NVPW_MetricsEvaluator_GetMetricNames_Params_STRUCT_SIZE,
                C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, me), type, C_NULL, C_NULL, 0))
            NVPW_MetricsEvaluator_GetMetricNames(params)

            names = Ptr{Cchar}(params[].pMetricNames)
            indices = params[].pMetricNameBeginIndices

            return new(me, type, names, indices, params[].numMetrics)
        end
    end
end

Base.length(metrics::MetricsIterator) = metrics.numMetrics
Base.eltype(::MetricsIterator) = String

function Base.iterate(metrics::MetricsIterator, state=1)
    if state <= metrics.numMetrics
        name = unsafe_string(metrics.names + unsafe_load(metrics.indices, state))
        return (name, state+1)
    else
        return nothing
    end
end

function list_metrics(me::MetricsEvaluator)
    for i in 0:(NVPW_METRIC_TYPE__COUNT-1)
        type = NVPW_MetricType(i)

        for metric in MetricsIterator(me, type)
            @show metric
        end
    end
end

function submetrics(me::MetricsEvaluator, type)
    GC.@preserve me begin
        params = Ref(NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params(
            NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, me), type, C_NULL, 0))
        NVPW_MetricsEvaluator_GetSupportedSubmetrics(params)
        unsafe_wrap(Array, params[].pSupportedSubmetrics, params[].numSupportedSubmetrics)
    end
end

# TODO rollup to string
# TODO submetric to string

# function submetric(m)
#     if m == NVPW_SUBMETRIC_PEAK_SUSTAINED
#         return ".peak_sustained"
#     elseif 

struct Metric
    me::MetricsEvaluator
    type::NVPW_MetricType
    index::Csize_t

    function Metric(me::MetricsEvaluator, name)
        GC.@preserve me name begin
            params = Ref(NVPW_MetricsEvaluator_GetMetricTypeAndIndex_Params(
                NVPW_MetricsEvaluator_GetMetricTypeAndIndex_Params_STRUCT_SIZE,
                C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, me), pointer(name), 0, 0))
            NVPW_MetricsEvaluator_GetMetricTypeAndIndex(params)
            return new(me, NVPW_MetricType(params[].metricType), params[].metricIndex)
        end
    end
end

struct HWUnit
    me::MetricsEvaluator
    hwUnit::UInt32
end

function Base.string(u::HWUnit)
    GC.@preserve u begin
        params = Ref(NVPW_MetricsEvaluator_HwUnitToString_Params(
            NVPW_MetricsEvaluator_HwUnitToString_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, u.me), u.hwUnit,
            C_NULL))
        NVPW_MetricsEvaluator_HwUnitToString(params)
        return unsafe_string(params[].pHwUnitName)
    end
end

function properties(m::Metric)
    if m.type == NVPW_METRIC_TYPE_COUNTER
        GC.@preserve m begin
            params = Ref(NVPW_MetricsEvaluator_GetCounterProperties_Params(
                NVPW_MetricsEvaluator_GetCounterProperties_Params_STRUCT_SIZE,
                C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, m.me), m.index,
                C_NULL, 0))
            NVPW_MetricsEvaluator_GetCounterProperties(params)
            description = unsafe_string(params[].pDescription)
            hwUnit = params[].hwUnit
            return (; description, unit=HWUnit(m.me, hwUnit))
        end
    elseif m.type == NVPW_METRIC_TYPE_RATIO
        GC.@preserve m begin
            params = Ref(NVPW_MetricsEvaluator_GetRatioMetricProperties_Params(
                NVPW_MetricsEvaluator_GetRatioMetricProperties_Params_STRUCT_SIZE,
                C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, m.me), m.index,
                C_NULL, 0))
            NVPW_MetricsEvaluator_GetRatioMetricProperties(params)
            description = unsafe_string(params[].pDescription)
            hwUnit = params[].hwUnit
            return (; description, unit=HWUnit(m.me, hwUnit))
        end
    else
        error("Not implemented for $(m.type)")
    end
end

struct MetricEvalRequest
    data::NVPW_MetricEvalRequest

    function MetricEvalRequest(me::MetricsEvaluator, name)
        eval_request = Ref{NVPW_MetricEvalRequest}()
        GC.@preserve me name eval_request begin
            params = Ref(NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params(
                NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE,
                C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, me), pointer(name),
                Base.unsafe_convert(Ptr{NVPW_MetricEvalRequest}, eval_request), NVPW_MetricEvalRequest_STRUCT_SIZE))
            NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(params)
            return new(eval_request[])
        end
    end
end

mutable struct MetricEvalRequestSet
    me::MetricsEvaluator
    mers::Vector{MetricEvalRequest}
end

function raw(mers::MetricEvalRequestSet; keepInstances=true, isolated=true)
    GC.@preserve mers begin
        params = Ref(NVPW_MetricsEvaluator_GetMetricRawDependencies_Params(
            NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, mers.me), 
            Base.unsafe_convert(Ptr{NVPW_MetricEvalRequest}, mers.mers),
            length(mers.mers), NVPW_MetricEvalRequest_STRUCT_SIZE, sizeof(NVPW_MetricEvalRequest), C_NULL, 0, C_NULL, 0))
        NVPW_MetricsEvaluator_GetMetricRawDependencies(params)
        sz = params[].numRawDependencies
        # @show params[].numOptionalRawDependencies
    end

    rawDeps = Vector{Ptr{Cchar}}(undef, sz)
    GC.@preserve mers rawDeps begin
        params = Ref(NVPW_MetricsEvaluator_GetMetricRawDependencies_Params(
            NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{NVPW_MetricsEvaluator}, mers.me), 
            Base.unsafe_convert(Ptr{NVPW_MetricEvalRequest}, mers.mers),
            length(mers.mers), NVPW_MetricEvalRequest_STRUCT_SIZE, sizeof(NVPW_MetricEvalRequest), pointer(rawDeps), sz, C_NULL, 0))
        NVPW_MetricsEvaluator_GetMetricRawDependencies(params)
    end

    reqs = Vector{NVPA_RawMetricRequest}(undef, sz)
    for i in eachindex(reqs)
        reqs[i] = NVPA_RawMetricRequest(
            NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE,
            C_NULL, rawDeps[i], isolated, keepInstances)
    end

    return reqs
end

abstract type RawMetricsConfig end
mutable struct CUDARawMetricsConfig <: RawMetricsConfig
    handle::Ptr{NVPA_RawMetricsConfig}

    function CUDARawMetricsConfig(chipName, counterAvailability; activity=NVPA_ACTIVITY_KIND_PROFILER)
        GC.@preserve chipName counterAvailability begin
            params = Ref(NVPW_CUDA_RawMetricsConfig_Create_V2_Params(
                NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE,
                C_NULL, activity, pointer(chipName), pointer(counterAvailability), C_NULL))
            NVPW_CUDA_RawMetricsConfig_Create_V2(params)
            this = new(params[].pRawMetricsConfig)
        end
        finalizer(destroy, this)
        return this
    end
end
Base.unsafe_convert(::Type{Ptr{NVPA_RawMetricsConfig}}, config::CUDARawMetricsConfig) = config.handle

function destroy(rmc::RawMetricsConfig)
    GC.@preserve rmc begin
        params = Ref(NVPW_RawMetricsConfig_Destroy_Params(
            NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{NVPA_RawMetricsConfig}, rmc)))
        NVPW_RawMetricsConfig_Destroy(params)
    end
    return nothing
end

function begin_config_group(config::RawMetricsConfig, maxPassCount)
    GC.@preserve config begin
        params = Ref(NVPW_RawMetricsConfig_BeginPassGroup_Params(
                    NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE,
                    C_NULL, Base.unsafe_convert(Ptr{NVPA_RawMetricsConfig}, config), maxPassCount))
        NVPW_RawMetricsConfig_BeginPassGroup(params)
    end
end

function add!(config::RawMetricsConfig, metrics::Vector{NVPA_RawMetricRequest})
    GC.@preserve config metrics begin
        params = Ref(NVPW_RawMetricsConfig_AddMetrics_Params(
            NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{NVPA_RawMetricsConfig}, config),
            pointer(metrics), length(metrics)
        ))
        NVPW_RawMetricsConfig_AddMetrics(params)
    end
    return nothing
end

function end_config_group(config::RawMetricsConfig)
    GC.@preserve config begin
        params = Ref(NVPW_RawMetricsConfig_EndPassGroup_Params(
                    NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE,
                    C_NULL, Base.unsafe_convert(Ptr{NVPA_RawMetricsConfig}, config)))
        NVPW_RawMetricsConfig_EndPassGroup(params)
    end
end

function generate(config::RawMetricsConfig, mergeAllPassGroups=true)
    GC.@preserve config begin
        params = Ref(NVPW_RawMetricsConfig_GenerateConfigImage_Params(
                    NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE,
                    C_NULL, Base.unsafe_convert(Ptr{NVPA_RawMetricsConfig}, config), mergeAllPassGroups))
        NVPW_RawMetricsConfig_GenerateConfigImage(params)
    end
end

function config_image(config::RawMetricsConfig)
    GC.@preserve config begin
        params = Ref(NVPW_RawMetricsConfig_GetConfigImage_Params(
                    NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE,
                    C_NULL, Base.unsafe_convert(Ptr{NVPA_RawMetricsConfig}, config), 0, C_NULL, 0))
        NVPW_RawMetricsConfig_GetConfigImage(params)

        sz = params[].bytesCopied
    end

    image = Vector{UInt8}(undef, sz)
    GC.@preserve config image begin
        params = Ref(NVPW_RawMetricsConfig_GetConfigImage_Params(
                    NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE,
                    C_NULL, Base.unsafe_convert(Ptr{NVPA_RawMetricsConfig}, config), length(image), pointer(image), 0))
        NVPW_RawMetricsConfig_GetConfigImage(params)
    end
    return image
end

import ..CUPTIExt: CounterDataBuilder, prefix
mutable struct CUDACounterDataBuilder <: CounterDataBuilder
    handle::Ptr{NVPA_CounterDataBuilder}

    function CUDACounterDataBuilder(chipName, counterAvailability)
        GC.@preserve chipName counterAvailability begin
            params = Ref(NVPW_CUDA_CounterDataBuilder_Create_Params(
                NVPW_CUDA_CounterDataBuilder_Create_Params_STRUCT_SIZE,
                C_NULL, pointer(chipName), pointer(counterAvailability), C_NULL))
            NVPW_CUDA_CounterDataBuilder_Create(params)
            this = new(params[].pCounterDataBuilder)
        end
        finalizer(destroy, this)
        return this
    end

end
Base.unsafe_convert(::Type{Ptr{NVPA_CounterDataBuilder}}, builder::CUDACounterDataBuilder) = builder.handle

function destroy(builder::CounterDataBuilder)
    GC.@preserve builder begin
        params = Ref(NVPW_CounterDataBuilder_Destroy_Params(
                NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE,
                C_NULL, Base.unsafe_convert(Ptr{NVPA_CounterDataBuilder}, builder)))
        NVPW_CounterDataBuilder_Destroy(params)
    end
    return nothing
end

function add!(builder::CounterDataBuilder, metrics::Vector{NVPA_RawMetricRequest})
    GC.@preserve builder metrics begin
        params = Ref(NVPW_CounterDataBuilder_AddMetrics_Params(
            NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{NVPA_CounterDataBuilder}, builder),
            pointer(metrics), length(metrics)
        ))
        NVPW_CounterDataBuilder_AddMetrics(params)
    end
    return nothing
end

function prefix(builder::CounterDataBuilder)
    GC.@preserve builder begin
        params = Ref(NVPW_CounterDataBuilder_GetCounterDataPrefix_Params(
                    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE,
                    C_NULL, Base.unsafe_convert(Ptr{NVPA_CounterDataBuilder}, builder), 0, C_NULL, 0))
        NVPW_CounterDataBuilder_GetCounterDataPrefix(params)

        sz = params[].bytesCopied
    end

    prefix = Vector{UInt8}(undef, sz)
    GC.@preserve builder prefix begin
        params = Ref(NVPW_CounterDataBuilder_GetCounterDataPrefix_Params(
                    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE,
                    C_NULL, Base.unsafe_convert(Ptr{NVPA_CounterDataBuilder}, builder), length(prefix), pointer(prefix), 0))
        NVPW_CounterDataBuilder_GetCounterDataPrefix(params)
    end
    return prefix
end


