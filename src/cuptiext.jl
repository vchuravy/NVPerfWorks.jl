module CUPTIExt
    using CUDA
    import CUDA: CUPTI

    excluded = Set([var"#eval",])
    for name = names(CUPTI, all=true)
        if startswith(String(name), "#") ||
            name == :CUPTI || name == :eval || name == :include
            continue
        end
        @eval const $name = CUPTI.$name
    end

    function initialize_profiler()
        params = Ref(CUpti_Profiler_Initialize_Params(
            CUpti_Profiler_Initialize_Params_STRUCT_SIZE,
            C_NULL))
        cuptiProfilerInitialize(params)
    end

    function deinitalize_profiler()
        params = Ref(CUpti_Profiler_DeInitialize_Params(
            CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE,
            C_NULL))
        cuptiProfilerDeInitialize(params)
    end

    function counter_availability(ctx = context())
        # 1. Query size
        params = Ref(CUpti_Profiler_GetCounterAvailability_Params(
            CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE,
            C_NULL, ctx.handle, 0, C_NULL))
        cuptiProfilerGetCounterAvailability(params)

        sz = params[].counterAvailabilityImageSize
        buffer = Vector{UInt8}(undef, sz)

        GC.@preserve buffer begin
            params = Ref(CUpti_Profiler_GetCounterAvailability_Params(
            CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE,
            C_NULL, ctx.handle, sz, pointer(buffer)))
            cuptiProfilerGetCounterAvailability(params)
        end
        return buffer
    end

    abstract type CounterDataBuilder end
    function prefix end

    mutable struct CounterData
        images::Vector{UInt8}
        scratch::Vector{UInt8}
    end

    function CounterData(builder, maxNumRanges, maxNumRangeTreeNodes, maxRangeNameLength)
        p = prefix(builder)
        GC.@preserve p begin
            options = Ref(CUpti_Profiler_CounterDataImageOptions(
                CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
                C_NULL, pointer(p), length(p), maxNumRanges, maxNumRangeTreeNodes, maxRangeNameLength))

            GC.@preserve options begin
                params = Ref(CUpti_Profiler_CounterDataImage_CalculateSize_Params(
                    CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE,
                    C_NULL, CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE, pointer(options), 0))
                CUpti_Profiler_CounterDataImage_CalculateSize(params)
                sz = params[].CounterDataImageSize
            end 
            images = Vector{UInt8}(undef, sz)
            GC.@preserve options images begin
                params = Ref(CUpti_Profiler_CounterDataImage_Initialize_Params(
                    CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE,
                    C_NULL, CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE, pointer(options),
                    sz, pointer(images)))
                CUpti_Profiler_CounterDataImage_Initialize(params)
            end
        end

        GC.@preserve images begin
            params = Ref(CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params(
                CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE,
                C_NULL, sz, pointer(images),0))
            CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize(params)
            scratch_sz = params[].counterDataScratchBufferSize
        end
        scratch = Vector{UInt8}(undef, scratch_sz)
        GC.@preserve images scratch begin
            params = Ref(CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params(
                CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE,
                C_NULL, sz, pointer(images),scratch_sz, pointer(scratch)))
            CUpti_Profiler_CounterDataImage_InitializeScratchBuffer(params)
        end

        return CounterData(images, scratch)
    end

end