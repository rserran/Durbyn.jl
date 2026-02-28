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
