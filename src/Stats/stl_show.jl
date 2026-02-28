function Base.show(io::IO, result::STLResult)
    println(io, "STL decomposition")
    println(io, "Seasonal component (first 10 values): ", result.seasonal[1:min(end,10)])
    println(io, "Trend component    (first 10 values): ", result.trend[1:min(end,10)])
    println(io, "Remainder          (first 10 values): ", result.remainder[1:min(end,10)])
    println(io, "Windows: seasonal=", result.seasonal_window, ", trend=", result.trend_window, ", lowpass=", result.lowpass_window)
    println(io, "Degrees: seasonal=", result.seasonal_degree, ", trend=", result.trend_degree, ", lowpass=", result.lowpass_degree)
    println(io, "Jumps: seasonal=", result.seasonal_jump, ", trend=", result.trend_jump, ", lowpass=", result.lowpass_jump)
    println(io, "Inner iterations: ", result.inner_iterations, ", Outer iterations: ", result.outer_iterations)
    return
end

function summary(result::STLResult; digits::Integer=4)
    n = length(result.seasonal)
    data = result.seasonal .+ result.trend .+ result.remainder
    comps = Dict(
        :seasonal => result.seasonal,
        :trend    => result.trend,
        :remainder=> result.remainder,
        :data     => data,
    )

    function iqr(v::AbstractVector)
        q25, q75 = quantile(v, [0.25, 0.75])
        return q75 - q25
    end
    println("STL decomposition summary")
    println("Time series components:")
    for (name, vec) in comps
        println("  ", name)
        mv = mean(vec)
        sv = std(vec)
        mn = minimum(vec)
        mx = maximum(vec)
        iqr_v = iqr(vec)

        component_fmt = string("% .", digits, "f")
        full_fmt = "    mean=" * component_fmt * "  sd=" * component_fmt *
                   "  min=" * component_fmt * "  max=" * component_fmt *
                   "  IQR=" * component_fmt
        f = Printf.Format(full_fmt)
        println(Printf.format(f, mv, sv, mn, mx, iqr_v))
    end
    println("IQR as percentage of total:")
    iqr_vals = Dict(name => iqr(vec) for (name, vec) in comps)
    total_iqr = iqr_vals[:data]
    for (name, v) in iqr_vals
        pct = total_iqr == 0 ? NaN : 100.0 * v / total_iqr
        pct_str = isnan(pct) ? "NaN" : string(round(pct; digits=1))
        println("  ", Symbol(name), ": ", pct_str, "%")
    end

    if all(w -> w == 1.0, result.weights)
        println("Weights: all equal to 1")
    else
        w = result.weights
        mv = mean(w)
        sv = std(w)
        mn = minimum(w)
        mx = maximum(w)
        iqr_w = iqr(w)
        println("Weights summary:")
        s_mean = string(round(mv; digits=digits))
        s_sd   = string(round(sv; digits=digits))
        s_min  = string(round(mn; digits=digits))
        s_max  = string(round(mx; digits=digits))
        s_iqr  = string(round(iqr_w; digits=digits))
        println("  mean=", s_mean, "  sd=", s_sd,
                "  min=", s_min, "  max=", s_max, "  IQR=", s_iqr)
    end
    println("Other components: seasonal_window=", result.seasonal_window,
            ", trend_window=", result.trend_window,
            ", lowpass_window=", result.lowpass_window,
            ", inner=", result.inner_iterations,
            ", outer=", result.outer_iterations)
    return nothing
end

function Base.show(io::IO, res::MSTLResult)
    n = length(res.data)
    preview = min(n, 10)
    println(io, "MSTL decomposition")
    println(io, "  length: ", n)
    if isempty(res.m)
        println(io, "  periods: (none)")
    else
        println(io, "  periods: ", res.m)
    end
    println(io, "  lambda: ", isnothing(res.lambda) ? "nothing" : string(res.lambda))

    println(io, "Trend     (first $preview): ", res.trend[1:preview])
    if !isempty(res.seasonals)
        for (i, period) in enumerate(res.m)
            println(io, "Seasonal($period) (first $preview): ", res.seasonals[i][1:preview])
        end
    else
        println(io, "Seasonal: (none)")
    end
    println(io, "Remainder (first $preview): ", res.remainder[1:preview])
    return
end

function summary(res::MSTLResult; digits::Integer=4)
    total_seasonal = isempty(res.seasonals) ? zeros(eltype(res.data), length(res.data)) :
                                              reduce(+, res.seasonals)
    reconstructed = res.trend .+ total_seasonal .+ res.remainder

    comps = Dict{Symbol,AbstractVector{<:Real}}(
        :data      => reconstructed,
        :trend     => res.trend,
        :remainder => res.remainder,
    )

    if !isempty(res.seasonals)
        comps[:seasonal_total] = total_seasonal
        for (i, period) in enumerate(res.m)
            comps[Symbol("seasonal_$period")] = res.seasonals[i]
        end
    end

    iqr(v) = begin
        q25, q75 = quantile(v, (0.25, 0.75))
        q75 - q25
    end

    println("MSTL decomposition summary")
    println("Components (mean, sd, min, max, IQR):")
    fmt(x) = isnan(x) ? "NaN" : string(round(x; digits=digits))

    for (name, vec) in sort(collect(comps); by=first)
        println("  ", rpad(string(name), 16), " ",
                "mean=", fmt(mean(vec)), "  sd=", fmt(std(vec)),
                "  min=", fmt(minimum(vec)), "  max=", fmt(maximum(vec)),
                "  IQR=", fmt(iqr(vec)))
    end

    println("IQR as % of data IQR:")
    data_iqr = iqr(comps[:data])
    for (name, vec) in sort(collect(comps); by=first)
        pct = iszero(data_iqr) ? NaN : 100 * iqr(vec) / data_iqr
        println("  ", rpad(string(name), 16), " ", fmt(pct), "%")
    end

    println("Metadata: periods=", isempty(res.m) ? "[]" : string(res.m),
            ", lambda=", isnothing(res.lambda) ? "nothing" : string(res.lambda))
    return nothing
end
