"""
    DecomposedTimeSeries

A structure representing a decomposed time series, which includes the original
time series, seasonal, trend, random components, and associated metadata.

# Fields
- `x::AbstractVector`: The original time series data.
- `seasonal::AbstractVector`: The seasonal component of the time series.
- `trend::AbstractVector`: The trend component of the time series.
- `random::AbstractVector`: The random or residual component of the time series.
- `figure::AbstractVector`: The estimated seasonal figure only.
- `type::Symbol`: A symbol indicating the type of decomposition (`:additive` or `:multiplicative`).
- `m::Int`: The frequency of the x.

# Example
```julia
x = [1.0, 2.0, 3.0, 4.0, 5.0]
seasonal = [0.1, 0.2, 0.1, 0.2, 0.1]
trend = [0.5, 1.0, 1.5, 2.0, 2.5]
random = [0.4, 0.8, 1.4, 1.8, 2.4]
figure = []
type = :additive
m = 2
```
"""
struct DecomposedTimeSeries
    x::AbstractVector
    seasonal::AbstractVector
    trend::AbstractVector
    random::AbstractVector
    figure::AbstractVector
    type::Symbol
    m::Int
end

_is_missing_or_nan(v) = (ismissing(v) || (v isa AbstractFloat && isnan(v)))

_nanmean_skip(a) = begin
    if isempty(a)
        NaN
    else
        if eltype(a) <: Union{Missing,Number}
            xs = collect(skipmissing(a))
            xs = xs[.!isnan.(Float64.(xs))]
            isempty(xs) ? NaN : mean(Float64.(xs))
        else
            xs = Float64.(a)
            xs = xs[.!isnan.(xs)]
            isempty(xs) ? NaN : mean(xs)
        end
    end
end


function _symmetric_filter(
    x::AbstractVector{<:Number},
    w::AbstractVector{<:Number},
)::Vector{Float64}
    nx = length(x)
    nf = length(w)
    out = fill(NaN, nx)
    shift = nf ÷ 2

    xf = Float64.(x)
    wf = Float64.(w)

    for i = 1:nx
        i1 = i + shift - (nf - 1)
        i2 = i + shift
        if i1 < 1 || i2 > nx
            out[i] = NaN
            continue
        end

        s = 0.0
        valid = true
        @inbounds for j = 1:nf
            idx = i + shift - j + 1
            x_val = xf[idx]
            if _is_missing_or_nan(x_val)
                valid = false
                break
            end
            s += wf[j] * x_val
        end
        out[i] = valid ? s : NaN
    end
    return out
end

function _circular_filter(
    x::AbstractVector{<:Number},
    w::AbstractVector{<:Number},
)::Vector{Float64}
    nx = length(x)
    nf = length(w)
    out = fill(NaN, nx)
    shift = nf ÷ 2
    xf = Float64.(x)
    wf = Float64.(w)

    for i = 1:nx
        s = 0.0
        valid = true
        @inbounds for j = 1:nf
            idx = i + shift - j + 1
            while idx < 1
                idx += nx
            end
            while idx > nx
                idx -= nx
            end
            x_val = xf[idx]
            if _is_missing_or_nan(x_val)
                valid = false
                break
            end
            s += wf[j] * x_val
        end
        out[i] = valid ? s : NaN
    end
    return out
end


"""
    decompose(; x::AbstractVector, m::Int, type::Symbol=:additive, filter=nothing)

Classical Seasonal Decomposition by Moving Averages.

Decompose a vector of time series into seasonal, trend and irregular components
using moving averages. Deals with additive or multiplicative seasonal component.

# Arguments
- `x::AbstractVector`: A vector of one time series.
- `m::Int`: The frequency of the time series.
- `type::Symbol`: The type of seasonal component. Can be either `:additive` or `:multiplicative`.
- `filter`: A vector of filter coefficients in reverse time order
  (as for AR or MA coefficients), used for filtering out the seasonal component.
  If `nothing`, a moving average with symmetric window is performed.

# Returns
An object of class `DecomposedTimeSeries` with following components:
- `x`: The original series.
- `seasonal`: The seasonal component (i.e., the repeated seasonal figure).
- `trend`: The trend component.
- `random`: The remainder part.
- `figure`: The estimated seasonal figure only.
- `type`: The value of type.

# Examples
```julia
ap = air_passengers();
decompose(x = ap, m = 12, type= :multiplicative, filter = NaN)
decompose(x = ap, m = 12, type= :additive, filter = NaN)
```
"""
function decompose(;
    x::AbstractVector,
    m::Int,
    type::Symbol = :additive,
    filter::Union{Nothing,AbstractVector} = nothing,
)

    n = length(x)

    if m <= 1 || length([v for v in x if !_is_missing_or_nan(v)]) < 2 * m
        throw(ArgumentError("time series has no or less than 2 periods"))
    end
    decomp_type = type
    if decomp_type !== :additive && decomp_type !== :multiplicative
        throw(ArgumentError("type must be :additive or :multiplicative"))
    end

    filter_weights = if isnothing(filter)
        (m % 2 == 0) ? vcat([0.5], ones(m - 1), [0.5]) ./ m : ones(m) ./ m
    else
        Float64.(filter)
    end

    trend = _symmetric_filter(x, filter_weights)

    x_float = Float64.(x)
    detrended = similar(trend)
    if decomp_type === :additive
        @inbounds for i = 1:n
            x_val = x_float[i]
            trend_val = trend[i]
            detrended[i] = (_is_missing_or_nan(x_val) || _is_missing_or_nan(trend_val)) ? NaN : (x_val - trend_val)
        end
    else
        @inbounds for i = 1:n
            x_val = x_float[i]
            trend_val = trend[i]
            detrended[i] = (_is_missing_or_nan(x_val) || _is_missing_or_nan(trend_val) || trend_val == 0.0) ? NaN : (x_val / trend_val)
        end
    end

    figure = fill(NaN, m)
    for i = 1:m
        vals = detrended[i:m:n]
        subseries_mean = _nanmean_skip(vals)
        figure[i] = subseries_mean
    end

    figure_mean = _nanmean_skip(figure)
    if decomp_type === :additive
        figure .= figure .- figure_mean
    else
        figure .= figure ./ figure_mean
    end

    n_repeats = ceil(Int, n / m)
    seasonal = repeat(figure, n_repeats)[1:n]

    random = similar(trend)
    if decomp_type === :additive
        @inbounds for i = 1:n
            x_val = x_float[i]
            seasonal_val = seasonal[i]
            trend_val = trend[i]
            random[i] = (_is_missing_or_nan(x_val) || _is_missing_or_nan(seasonal_val) || _is_missing_or_nan(trend_val)) ? NaN : (x_val - seasonal_val - trend_val)
        end
    else
        @inbounds for i = 1:n
            x_val = x_float[i]
            seasonal_val = seasonal[i]
            trend_val = trend[i]
            random[i] =
                (_is_missing_or_nan(x_val) || _is_missing_or_nan(seasonal_val) || _is_missing_or_nan(trend_val) || seasonal_val == 0.0 || trend_val == 0.0) ? NaN :
                (x_val / (seasonal_val * trend_val))
        end
    end

    out = DecomposedTimeSeries(x, seasonal, trend, random, figure, type, m)
    return (out)
end
