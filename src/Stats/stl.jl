"""
    STLResult

Container for the results of an STL decomposition.  Fields store the
seasonal, trend and remainder components directly along with the
robustness weights, smoothing windows, local polynomial degrees, jump
parameters and iteration counts.
"""
struct STLResult{T<:Real}
    seasonal::Vector{T}
    trend::Vector{T}
    remainder::Vector{T}
    weights::Vector{T}
    seasonal_window::Int
    trend_window::Int
    lowpass_window::Int
    seasonal_degree::Int
    trend_degree::Int
    lowpass_degree::Int
    seasonal_jump::Int
    trend_jump::Int
    lowpass_jump::Int
    inner_iterations::Int
    outer_iterations::Int
end

function loess_estimate!(y::AbstractVector{Float64}, n::Int, bandwidth::Int, degree::Int,
                         eval_point::Float64, left_bound::Int, right_bound::Int,
                         w::AbstractVector{Float64}, use_weights::Bool, robustness_weights::AbstractVector{Float64})

    range = float(n) - 1.0
    half_width = max(eval_point - float(left_bound), float(right_bound) - eval_point)
    if bandwidth > n
        half_width += float((bandwidth - n) ÷ 2)
    end
    upper_threshold = 0.999 * half_width
    lower_threshold = 0.001 * half_width

    weight_sum = 0.0
    for j in left_bound:right_bound
        r = abs(float(j) - eval_point)
        if r <= upper_threshold
            if r <= lower_threshold || half_width == 0.0
                w[j] = 1.0
            else
                rr = r / half_width
                w[j] = (1.0 - rr^3)^3
            end
            if use_weights
                w[j] *= robustness_weights[j]
            end
            weight_sum += w[j]
        else
            w[j] = 0.0
        end
    end

    if weight_sum <= 0.0
        return 0.0, false
    end

    inv_weight_sum = 1.0 / weight_sum
    for j in left_bound:right_bound
        w[j] *= inv_weight_sum
    end

    if half_width > 0.0 && degree > 0

        a_mean = 0.0
        for j in left_bound:right_bound
            a_mean += w[j] * float(j)
        end
        b = eval_point - a_mean
        c = 0.0
        for j in left_bound:right_bound
            d = float(j) - a_mean
            c += w[j] * d^2
        end

        if sqrt(c) > 0.001 * range
            b /= c
            for j in left_bound:right_bound
                w[j] = w[j] * (b * (float(j) - a_mean) + 1.0)
            end
        end
    end

    ys = 0.0
    for j in left_bound:right_bound
        ys += w[j] * y[j]
    end
    return ys, true
end

function loess_smooth!(y::AbstractVector{Float64}, n::Int, bandwidth::Int, degree::Int, jump::Int,
                       use_weights::Bool, robustness_weights::AbstractVector{Float64},
                       ys::AbstractVector{Float64}, res::AbstractVector{Float64})

    if n < 2

        ys[firstindex(ys)] = y[1]
        return
    end

    new_jump = min(jump, n - 1)

    left_bound = 1
    right_bound = min(bandwidth, n)

    if bandwidth >= n
        left_bound = 1
        right_bound = n
        i = 1
        while i <= n
            eval_point = float(i)
            ysi, ok = loess_estimate!(y, n, bandwidth, degree, eval_point, left_bound, right_bound, res, use_weights, robustness_weights)
            if ok
                ys[firstindex(ys) - 1 + i] = ysi
            else
                ys[firstindex(ys) - 1 + i] = y[i]
            end
            i += new_jump
        end
    else
        if new_jump == 1
            half_bandwidth = (bandwidth + 1) ÷ 2
            left_bound = 1
            right_bound = bandwidth
            for i in 1:n
                if (i > half_bandwidth) && (right_bound != n)
                    left_bound += 1
                    right_bound += 1
                end
                eval_point = float(i)
                ysi, ok = loess_estimate!(y, n, bandwidth, degree, eval_point, left_bound, right_bound, res, use_weights, robustness_weights)
                if ok
                    ys[firstindex(ys) - 1 + i] = ysi
                else
                    ys[firstindex(ys) - 1 + i] = y[i]
                end
            end
        else
            half_bandwidth = (bandwidth + 1) ÷ 2
            i = 1
            while i <= n
                if i < half_bandwidth
                    left_bound = 1
                    right_bound = bandwidth
                elseif i >= n - half_bandwidth + 1
                    left_bound = n - bandwidth + 1
                    right_bound = n
                else
                    left_bound = i - half_bandwidth + 1
                    right_bound = bandwidth + i - half_bandwidth
                end
                eval_point = float(i)
                ysi, ok = loess_estimate!(y, n, bandwidth, degree, eval_point, left_bound, right_bound, res, use_weights, robustness_weights)
                if ok
                    ys[firstindex(ys) - 1 + i] = ysi
                else
                    ys[firstindex(ys) - 1 + i] = y[i]
                end
                i += new_jump
            end
        end
    end

    if new_jump != 1
        i = 1
        while i <= n - new_jump
            ysi = ys[firstindex(ys) - 1 + i]
            ysj = ys[firstindex(ys) - 1 + i + new_jump]
            delta = (ysj - ysi) / float(new_jump)
            for j in (i + 1):(i + new_jump - 1)
                ys[firstindex(ys) - 1 + j] = ysi + delta * float(j - i)
            end
            i += new_jump
        end

        k = ((n - 1) ÷ new_jump) * new_jump + 1
        if k != n

            eval_point = float(n)
            ysn, ok = loess_estimate!(y, n, bandwidth, degree, eval_point, left_bound, right_bound, res, use_weights, robustness_weights)
            if ok
                ys[firstindex(ys) - 1 + n] = ysn
            else
                ys[firstindex(ys) - 1 + n] = y[n]
            end
            if k != n - 1

                valk = ys[firstindex(ys) - 1 + k]
                valn = ys[firstindex(ys) - 1 + n]
                delta2 = (valn - valk) / float(n - k)
                for j in (k + 1):(n - 1)
                    ys[firstindex(ys) - 1 + j] = valk + delta2 * float(j - k)
                end
            end
        end
    end
    return
end

function moving_average!(x::AbstractVector{Float64}, n::Int, window::Int, ave::AbstractVector{Float64})

    if window <= 0 || n < window
        return
    end
    newn = n - window + 1

    v = 0.0
    for i in 1:window
        v += x[i]
    end
    fwindow = float(window)
    ave[1] = v / fwindow
    if newn > 1
        k = window
        m = 0
        for j in 2:newn
            k += 1
            m += 1
            v = v - x[m] + x[k]
            ave[j] = v / fwindow
        end
    end
    return
end

function lowpass_filter!(x::AbstractVector{Float64}, n::Int, period::Int,
                         trend::AbstractVector{Float64}, work::AbstractVector{Float64})

    moving_average!(x, n, period, trend)

    moving_average!(trend, n - period + 1, period, work)

    moving_average!(work, n - 2 * period + 2, 3, trend)
    return
end

function seasonal_smooth!(y::AbstractVector{Float64}, n::Int, period::Int, seasonal_bandwidth::Int, seasonal_degree::Int,
                           seasonal_jump::Int, use_weights::Bool, robustness_weights::AbstractVector{Float64},
                           season_ext::AbstractVector{Float64},
                           work1::AbstractVector{Float64}, work2::AbstractVector{Float64},
                           work3::AbstractVector{Float64}, work4::AbstractVector{Float64})

    if period < 1
        return
    end

    for j in 1:period

        k = ((n - j) ÷ period) + 1

        for i in 1:k
            idx = (i - 1) * period + j
            work1[i] = y[idx]
        end

        if use_weights
            for i in 1:k
                idx = (i - 1) * period + j
                work3[i] = robustness_weights[idx]
            end
        end

        loess_smooth!(work1, k, seasonal_bandwidth, seasonal_degree, seasonal_jump, use_weights, work3, view(work2, 2:(k + 1)), work4)

        eval_point = 0.0
        right_bound = min(seasonal_bandwidth, k)
        yfit, ok = loess_estimate!(work1, k, seasonal_bandwidth, seasonal_degree, eval_point, 1, right_bound, work4, use_weights, work3)
        if !ok
            yfit = work2[2]
        end
        work2[1] = yfit

        eval_point = float(k + 1)
        left_bound = max(1, k - seasonal_bandwidth + 1)
        yfit, ok = loess_estimate!(work1, k, seasonal_bandwidth, seasonal_degree, eval_point, left_bound, k, work4, use_weights, work3)
        if !ok
            yfit = work2[k + 1]
        end
        work2[k + 2] = yfit

        for idx in 1:(k + 2)
            pos = (idx - 1) * period + j
            season_ext[pos] = work2[idx]
        end
    end
    return
end


function robustness_weights!(y::AbstractVector{Float64}, fit::AbstractVector{Float64},
                             rw::AbstractVector{Float64})
    n = min(length(y), length(fit), length(rw))

    tmp = Vector{Float64}(undef, n)
    for i in 1:n
        tmp[i] = abs(y[i] - fit[i])
    end

    mad = median(tmp)
    cmad = 6.0 * mad

    c9 = 0.999 * cmad
    c1 = 0.001 * cmad
    for i in 1:n
        r = abs(y[i] - fit[i])
        if r <= c1
            rw[i] = 1.0
        elseif r <= c9 && cmad > 0.0
            x = r / cmad
            rw[i] = (1.0 - x^2)^2
        else
            rw[i] = 0.0
        end
    end
    return
end

function stl_inner_loop!(y::AbstractVector{Float64}, n::Int, period::Int,
                         seasonal_bandwidth::Int, trend_bandwidth::Int, lowpass_bandwidth::Int,
                         seasonal_degree::Int, trend_degree::Int, lowpass_degree::Int,
                         seasonal_jump::Int, trend_jump::Int, lowpass_jump::Int,
                         n_inner::Int, use_weights::Bool, robustness_weights::AbstractVector{Float64},
                         season::AbstractVector{Float64}, trend::AbstractVector{Float64})

    n_extended = n + 2 * period
    detrended = zeros(Float64, n_extended)
    seasonal_ext = zeros(Float64, n_extended)
    lowpass_smoothed = zeros(Float64, n_extended)
    work_a = zeros(Float64, n_extended)
    work_b = zeros(Float64, n_extended)
    for _iter in 1:n_inner

        for i in 1:n
            detrended[i] = y[i] - trend[i]
        end

        seasonal_smooth!(detrended, n, period, seasonal_bandwidth, seasonal_degree, seasonal_jump, use_weights, robustness_weights, seasonal_ext, lowpass_smoothed, work_a, work_b, season)
        lowpass_filter!(seasonal_ext, n_extended, period, lowpass_smoothed, detrended)
        loess_smooth!(lowpass_smoothed, n, lowpass_bandwidth, lowpass_degree, lowpass_jump, false, work_a, detrended, work_b)
        for i in 1:n
            season[i] = seasonal_ext[period + i] - detrended[i]
        end
        for i in 1:n
            detrended[i] = y[i] - season[i]
        end
        loess_smooth!(detrended, n, trend_bandwidth, trend_degree, trend_jump, use_weights, robustness_weights, trend, lowpass_smoothed)
    end
    return
end

function stl_outer_loop!(
    y::AbstractVector{Float64},
    period::Int,
    seasonal_bandwidth::Int,
    trend_bandwidth::Int,
    lowpass_bandwidth::Int,
    seasonal_degree::Int,
    trend_degree::Int,
    lowpass_degree::Int,
    seasonal_jump::Int,
    trend_jump::Int,
    lowpass_jump::Int,
    n_inner::Int,
    n_outer::Int,
    rw::AbstractVector{Float64},
    season::AbstractVector{Float64},
    trend::AbstractVector{Float64},
)
    n = length(y)
    fill!(trend, 0.0)
    new_seasonal_bandwidth = max(3, seasonal_bandwidth)
    if new_seasonal_bandwidth % 2 == 0
        new_seasonal_bandwidth += 1
    end
    new_trend_bandwidth = max(3, trend_bandwidth)
    if new_trend_bandwidth % 2 == 0
        new_trend_bandwidth += 1
    end
    new_lowpass_bandwidth = max(3, lowpass_bandwidth)
    if new_lowpass_bandwidth % 2 == 0
        new_lowpass_bandwidth += 1
    end
    new_period = max(2, period)
    use_weights = false
    k = 0
    while true
        stl_inner_loop!(
            y,
            n,
            new_period,
            new_seasonal_bandwidth,
            new_trend_bandwidth,
            new_lowpass_bandwidth,
            seasonal_degree,
            trend_degree,
            lowpass_degree,
            seasonal_jump,
            trend_jump,
            lowpass_jump,
            n_inner,
            use_weights,
            rw,
            season,
            trend,
        )
        k += 1
        if k > n_outer
            break
        end
        fit = Vector{Float64}(undef, n)
        for i = 1:n
            fit[i] = trend[i] + season[i]
        end
        robustness_weights!(y, fit, rw)
        use_weights = true
    end
    if n_outer <= 0
        for i = 1:n
            rw[i] = 1.0
        end
    end
    return
end

function stl_decompose(
    y::AbstractVector{Float64},
    period::Int,
    seasonal_bandwidth::Int,
    trend_bandwidth::Int,
    lowpass_bandwidth::Int,
    seasonal_degree::Int,
    trend_degree::Int,
    lowpass_degree::Int,
    seasonal_jump::Int,
    trend_jump::Int,
    lowpass_jump::Int,
    n_inner::Int,
    n_outer::Int,
)
    n = length(y)
    season = zeros(Float64, n)
    trend = zeros(Float64, n)
    rw = zeros(Float64, n)
    stl_outer_loop!(
        y,
        period,
        seasonal_bandwidth,
        trend_bandwidth,
        lowpass_bandwidth,
        seasonal_degree,
        trend_degree,
        lowpass_degree,
        seasonal_jump,
        trend_jump,
        lowpass_jump,
        n_inner,
        n_outer,
        rw,
        season,
        trend,
    )
    return season, trend, rw
end

function _validate_degree(deg, name::AbstractString)
    d = Int(deg)
    if d < 0 || d > 1
        throw(ArgumentError("$name must be 0 or 1"))
    end
    return d
end



"""
    stl(x, m; kwargs...)

Seasonal-trend decomposition based on Loess (STL).

Decomposes the one-dimensional numeric array `x` into **seasonal**,
**trend** and **remainder** components.  `m` specifies the seasonal
period (number of observations per cycle) and must be at least two.

# References

- R. B. Cleveland, W. S. Cleveland, J. E. McRae, and I. Terpenning (1990).
  *STL: A Seasonal-Trend Decomposition Procedure Based on Loess.*
  Journal of Official Statistics, 6, 3–73.
- G. Bodin, *SeasonalTrendLoess.jl*, <https://github.com/guilhermebodin/SeasonalTrendLoess.jl>

# Arguments

* `x`: A numeric vector containing the time series to be decomposed.
* `m`: An integer specifying the frequency (periodicity) of the series.

# Keyword arguments

Defaults follow the R `stl` implementation.

* `seasonal_window`: Span of the seasonal smoothing window.  May be an
  integer (interpreted as a span and rounded to the nearest odd value) or
  the symbol `:periodic` to request a periodic seasonal component.
  Defaults to `:periodic`.
* `seasonal_degree`: Degree of the local polynomial used for seasonal
  smoothing (0 or 1).  Defaults to 0.
* `trend_window`: Span of the trend smoothing window.  If omitted, a
  default based on `m` and `seasonal_window` is computed.  Must be odd.
* `trend_degree`: Degree of the local polynomial used for trend
  smoothing (0 or 1).  Defaults to 1.
* `lowpass_window`: Span of the low-pass filter.  Defaults to the nearest
  odd integer greater than or equal to `m`.
* `lowpass_degree`: Degree of the local polynomial used for the low-pass
  filter.  Defaults to the value of `trend_degree`.
* `seasonal_jump`, `trend_jump`, `lowpass_jump`: Subsampling step sizes
  used when evaluating the loess smoother.  Defaults are one tenth of
  the corresponding window lengths (rounded up).
* `robust`: Logical flag indicating whether to compute robustness
  weights.  When true up to 15 outer iterations are performed; when
  false no robustness iterations are used.
* `inner`: Number of inner loop iterations.  Defaults to 1 when
  `robust` is true and 2 otherwise.
* `outer`: Number of outer robustness iterations.  Defaults to 15
  when `robust` is true and 0 otherwise.

# Returns

An [`STLResult`](@ref) containing the seasonal, trend and remainder
components along with ancillary information.

# Examples
```julia
res = stl(AirPassengers, 12)                          # periodic seasonal
res = stl(AirPassengers, 12; seasonal_window=7)       # fixed window
res = stl(AirPassengers, 12; robust=true)             # robust fitting
```
"""
function stl(
    x::AbstractVector{T},
    m::Integer;
    seasonal_window::Union{Int,Symbol} = :periodic,
    seasonal_degree::Integer = 0,
    trend_window::Union{Nothing,Integer} = nothing,
    trend_degree::Integer = 1,
    lowpass_window::Union{Nothing,Integer} = nothing,
    lowpass_degree::Integer = trend_degree,
    seasonal_jump::Union{Nothing,Integer} = nothing,
    trend_jump::Union{Nothing,Integer} = nothing,
    lowpass_jump::Union{Nothing,Integer} = nothing,
    robust::Bool = false,
    inner::Union{Nothing,Integer} = nothing,
    outer::Union{Nothing,Integer} = nothing,
) where {T<:Real}

    n = length(x)
    if m < 2 || n <= 2 * m
        throw(ArgumentError("series is not periodic or has less than two periods"))
    end

    if any(ismissing, x)
        throw(ArgumentError(
            "Input data contains missing values; consider imputing or removing them before calling stl",
        ))
    end

    periodic = false
    if isa(seasonal_window, Symbol)
        if seasonal_window === :periodic
            periodic = true
            seasonal_window_val = 10 * n + 1
            seasonal_degree = 0
        else
            throw(ArgumentError("unknown symbol value for seasonal_window: $seasonal_window"))
        end
    elseif isa(seasonal_window, Integer)
        seasonal_window_val = nearest_odd(seasonal_window)
    else
        seasonal_window_val = nearest_odd(round(Int, seasonal_window))
    end

    seasonal_degree = _validate_degree(seasonal_degree, "seasonal_degree")
    trend_degree = _validate_degree(trend_degree, "trend_degree")
    lowpass_degree = _validate_degree(lowpass_degree, "lowpass_degree")

    if isnothing(trend_window)
        trend_window_val = nearest_odd(ceil(Int, 1.5 * m / (1.0 - 1.5 / seasonal_window_val)))
    else
        trend_window_val = nearest_odd(trend_window)
    end

    if isnothing(lowpass_window)
        lowpass_window_val = nearest_odd(m)
    else
        lowpass_window_val = nearest_odd(lowpass_window)
    end

    if isnothing(seasonal_jump)
        seasonal_jump_val = max(1, Int(ceil(seasonal_window_val / 10)))
    else
        seasonal_jump_val = seasonal_jump
    end
    if isnothing(trend_jump)
        trend_jump_val = max(1, Int(ceil(trend_window_val / 10)))
    else
        trend_jump_val = trend_jump
    end
    if isnothing(lowpass_jump)
        lowpass_jump_val = max(1, Int(ceil(lowpass_window_val / 10)))
    else
        lowpass_jump_val = lowpass_jump
    end

    if isnothing(inner)
        inner_val = robust ? 1 : 2
    else
        inner_val = inner
    end
    if isnothing(outer)
        outer_val = robust ? 15 : 0
    else
        outer_val = outer
    end

    xvec = collect(float.(x))

    season, trend, weights = stl_decompose(
        xvec,
        m,
        seasonal_window_val,
        trend_window_val,
        lowpass_window_val,
        seasonal_degree,
        trend_degree,
        lowpass_degree,
        seasonal_jump_val,
        trend_jump_val,
        lowpass_jump_val,
        inner_val,
        outer_val,
    )
    remainder = xvec .- season .- trend

    if periodic
        cycle = [(i - 1) % m + 1 for i = 1:n]
        mean_by_cycle = zeros(Float64, m)
        counts = zeros(Int, m)
        for i = 1:n
            idx = cycle[i]
            mean_by_cycle[idx] += season[i]
            counts[idx] += 1
        end
        for j = 1:m
            if counts[j] > 0
                mean_by_cycle[j] /= counts[j]
            end
        end
        for i = 1:n
            season[i] = mean_by_cycle[cycle[i]]
        end
        remainder = xvec .- season .- trend
    end
    return STLResult{Float64}(
        season,
        trend,
        remainder,
        weights,
        seasonal_window_val,
        trend_window_val,
        lowpass_window_val,
        seasonal_degree,
        trend_degree,
        lowpass_degree,
        seasonal_jump_val,
        trend_jump_val,
        lowpass_jump_val,
        inner_val,
        outer_val,
    )
end


"""
    Base.show(io::IO, result::STLResult)

Pretty print an `STLResult`.
"""
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

"""
    summary(result::STLResult; digits=4)

Display a statistical summary of an `STLResult`.
"""
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
