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
        image::Vector{UInt8}
        scratch::Vector{UInt8}
        maxNumRanges::Int64
        maxNumRangeTreeNodes::Int64

        function CounterData(prefix, maxNumRanges, maxNumRangeTreeNodes, maxRangeNameLength)
            GC.@preserve prefix begin
                options = Ref(CUpti_Profiler_CounterDataImageOptions(
                    CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
                    C_NULL, pointer(prefix), length(prefix), maxNumRanges, maxNumRangeTreeNodes, maxRangeNameLength))

                GC.@preserve options begin
                    params = Ref(CUpti_Profiler_CounterDataImage_CalculateSize_Params(
                        CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE,
                        C_NULL, CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE, 
                        Base.unsafe_convert(Ptr{CUpti_Profiler_CounterDataImageOptions}, options), 0))
                    cuptiProfilerCounterDataImageCalculateSize(params)
                    sz = params[].counterDataImageSize
                end 
                image = Vector{UInt8}(undef, sz)
                GC.@preserve options image begin
                    params = Ref(CUpti_Profiler_CounterDataImage_Initialize_Params(
                        CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE,
                        C_NULL, CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE, 
                        Base.unsafe_convert(Ptr{CUpti_Profiler_CounterDataImageOptions}, options),
                        sz, pointer(image)))
                    cuptiProfilerCounterDataImageInitialize(params)
                end
            end

            GC.@preserve image begin
                params = Ref(CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params(
                    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE,
                    C_NULL, sz, pointer(image),0))
                cuptiProfilerCounterDataImageCalculateScratchBufferSize(params)
                scratch_sz = params[].counterDataScratchBufferSize
            end
            scratch = Vector{UInt8}(undef, scratch_sz)
            GC.@preserve image scratch begin
                params = Ref(CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params(
                    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE,
                    C_NULL, sz, pointer(image), scratch_sz, pointer(scratch)))
                cuptiProfilerCounterDataImageInitializeScratchBuffer(params)
            end

            return new(image, scratch, maxNumRanges, maxNumRangeTreeNodes)
        end
    end

    # NOTE: CD must be live across begin_session/end_session
    function begin_session(cd::CounterData, range, replayMode;
                           maxRanges=cd.maxNumRanges, maxLaunches=cd.maxNumRanges, ctx=context())
        params = Ref(CUpti_Profiler_BeginSession_Params(
            CUpti_Profiler_BeginSession_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{CUDA.CUctx_st}, ctx),
            length(cd.image), pointer(cd.image),
            length(cd.scratch), pointer(cd.scratch),
            false, C_NULL, range, replayMode, maxRanges, maxLaunches))

        cuptiProfilerBeginSession(params)
    end

    function set_config(config; minNestingLevel=1, numNestingLevels=1, passIndex=1, ctx=context())
        GC.@preserve config ctx begin
            params = Ref(CUpti_Profiler_SetConfig_Params(
                CUpti_Profiler_SetConfig_Params_STRUCT_SIZE,
                C_NULL, Base.unsafe_convert(Ptr{CUDA.CUctx_st}, ctx),
                pointer(config), length(config),
                minNestingLevel, numNestingLevels, passIndex, 0
            ))
            cuptiProfilerSetConfig(params)
        end
    end

    function begin_pass(;ctx=context())
        params = Ref(CUpti_Profiler_BeginPass_Params(
            CUpti_Profiler_BeginPass_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{CUDA.CUctx_st}, ctx)))
        cuptiProfilerBeginPass(params)
    end

    function enable_profiling(;ctx=context())
        params = Ref(CUpti_Profiler_EnableProfiling_Params(
            CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{CUDA.CUctx_st}, ctx)))
        cuptiProfilerEnableProfiling(params)
    end

    function push_range(name; ctx=context())
        GC.@preserve name begin
            params = Ref(CUpti_Profiler_PushRange_Params(
                CUpti_Profiler_PushRange_Params_STRUCT_SIZE,
                C_NULL, Base.unsafe_convert(Ptr{CUDA.CUctx_st}, ctx),
                pointer(name), length(name)))
            cuptiProfilerPushRange(params)
        end
    end

    function pop_range(;ctx=context())
        params = Ref(CUpti_Profiler_PopRange_Params(
            CUpti_Profiler_PopRange_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{CUDA.CUctx_st}, ctx)))
        cuptiProfilerPopRange(params)
    end

    function disable_profiling(;ctx=context())
        params = Ref(CUpti_Profiler_DisableProfiling_Params(
            CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{CUDA.CUctx_st}, ctx)))
        cuptiProfilerDisableProfiling(params)
    end

    function end_pass(passIndex=1 ;ctx=context())
        params = Ref(CUpti_Profiler_EndPass_Params(
            CUpti_Profiler_EndPass_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{CUDA.CUctx_st}, ctx),
            0, passIndex, 0))
        cuptiProfilerEndPass(params)
        return !(params[].allPassesSubmitted == 0)
    end

    function flush_counter_data(;ctx=context())
        params = Ref(CUpti_Profiler_FlushCounterData_Params(
            CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{CUDA.CUctx_st}, ctx),
            0,0))
        cuptiProfilerFlushCounterData(params)
        numRangesDropped = params[].numRangesDropped
        numTraceBytesDropped = params[].numTraceBytesDropped
        return (;numRangesDropped, numTraceBytesDropped)
    end

    function unset_config(;ctx=context())
        params = Ref(CUpti_Profiler_UnsetConfig_Params(
            CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{CUDA.CUctx_st}, ctx)))
        cuptiProfilerUnsetConfig(params)
    end

    function end_session(;ctx=context())
        params = Ref(CUpti_Profiler_EndSession_Params(
            CUpti_Profiler_EndSession_Params_STRUCT_SIZE,
            C_NULL, Base.unsafe_convert(Ptr{CUDA.CUctx_st}, ctx)))
        cuptiProfilerEndSession(params)
    end

end