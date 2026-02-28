"""
    ACFResult

Container for the results of an ACF (Autocorrelation Function) computation.

# Fields
- `values::Vector{Float64}`: ACF values at each lag (including lag 0)
- `lags::Vector{Int}`: Lag indices (0, 1, 2, ..., n_lags)
- `n::Int`: Length of the original time series
- `m::Int`: Frequency/seasonal period of the data
- `ci::Float64`: Critical value for 95% confidence interval (±1.96/√n)
- `type::Symbol`: Always `:acf`

# Usage
```julia
result = acf(y, m)
result = acf(y, m, 20)
plot(result)
```
"""
struct ACFResult
    values::Vector{Float64}
    lags::Vector{Int}
    n::Int
    m::Int
    ci::Float64
    type::Symbol
end

"""
    PACFResult

Container for the results of a PACF (Partial Autocorrelation Function) computation.

# Fields
- `values::Vector{Float64}`: PACF values at each lag (lags 1, 2, ..., n_lags)
- `lags::Vector{Int}`: Lag indices (1, 2, ..., n_lags)
- `n::Int`: Length of the original time series
- `m::Int`: Frequency/seasonal period of the data
- `ci::Float64`: Critical value for 95% confidence interval (±1.96/√n)
- `type::Symbol`: Always `:pacf`

# Usage
```julia
result = pacf(y, m)
result = pacf(y, m, 20)
plot(result)
```
"""
struct PACFResult
    values::Vector{Float64}
    lags::Vector{Int}
    n::Int
    m::Int
    ci::Float64
    type::Symbol
end

"""
    acf(y, m, n_lags=nothing; demean=true) -> ACFResult

Compute the sample autocorrelation function (ACF) of a time series.

# Arguments
- `y::AbstractVector`: Input time series
- `m::Int`: Frequency/seasonal period of the data
- `n_lags::Union{Int,Nothing}=nothing`: Number of lags to compute. If `nothing`,
  defaults to `min(10*log10(n), n-1)` following R's convention.
- `demean::Bool=true`: Whether to subtract the mean before computing ACF

# Returns
`ACFResult` containing autocorrelations and metadata. Use `plot(result)` to visualize.

# Details
Uses the standard biased estimator:
```math
\\hat{\\rho}(k) = \\frac{\\sum_{t=1}^{n-k} (y_t - \\bar{y})(y_{t+k} - \\bar{y})}{\\sum_{t=1}^{n} (y_t - \\bar{y})^2}
```

# Example
```julia
y = randn(100)
result = acf(y, 12)
result.values
result.lags
plot(result)
```

# References
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
  Time Series Analysis: Forecasting and Control. Wiley.
"""
function acf(y::AbstractVector{T}, m::Int, n_lags::Union{Int,Nothing}=nothing; demean::Bool=true) where T<:Real
    n = length(y)

    if isnothing(n_lags)
        n_lags = min(floor(Int, 10 * log10(n)), n - 1)
    end

    if n_lags < 0
        throw(ArgumentError("n_lags must be non-negative"))
    end
    if n_lags >= n
        throw(ArgumentError("n_lags must be less than length of series"))
    end
    if m < 1
        throw(ArgumentError("frequency m must be at least 1"))
    end

    y_centered = demean ? y .- mean(y) : y
    variance = sum(y_centered .^ 2) / n

    if variance == 0
        values = ones(Float64, n_lags + 1)
    else
        values = zeros(Float64, n_lags + 1)
        values[1] = 1.0

        for k in 1:n_lags
            autocovariance = sum(y_centered[1:n-k] .* y_centered[k+1:n]) / n
            values[k + 1] = autocovariance / variance
        end
    end

    lags = collect(0:n_lags)
    ci = 1.96 / sqrt(n)

    return ACFResult(values, lags, n, m, ci, :acf)
end

"""
    pacf(y, m, n_lags=nothing) -> PACFResult

Compute the sample partial autocorrelation function (PACF) of a time series
using the Durbin-Levinson algorithm.

# Arguments
- `y::AbstractVector`: Input time series
- `m::Int`: Frequency/seasonal period of the data
- `n_lags::Union{Int,Nothing}=nothing`: Number of lags to compute. If `nothing`,
  defaults to `min(10*log10(n), n-1)` following R's convention.

# Returns
`PACFResult` containing partial autocorrelations and metadata. Use `plot(result)` to visualize.

# Example
```julia
y = randn(100)
result = pacf(y, 12)
result.values
result.lags
plot(result)
```

# References
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
  Time Series Analysis: Forecasting and Control. Wiley.
- Durbin, J. (1960). The fitting of time-series models. *Revue de l'Institut
  International de Statistique*, 28(3), 233-244.
"""
function pacf(y::AbstractVector{T}, m::Int, n_lags::Union{Int,Nothing}=nothing) where T<:Real
    n = length(y)

    if isnothing(n_lags)
        n_lags = min(floor(Int, 10 * log10(n)), n - 1)
    end

    if n_lags < 1
        throw(ArgumentError("n_lags must be at least 1"))
    end
    if n_lags >= n
        throw(ArgumentError("n_lags must be less than length of series"))
    end
    if m < 1
        throw(ArgumentError("frequency m must be at least 1"))
    end

    acf_result = acf(y, m, n_lags)
    acf_values = acf_result.values

    pacf_vals = zeros(Float64, n_lags)
    coefficients = zeros(Float64, n_lags, n_lags)

    coefficients[1, 1] = acf_values[2]
    pacf_vals[1] = coefficients[1, 1]

    for k in 2:n_lags
        numerator = acf_values[k + 1]
        denominator = 1.0
        for j in 1:k-1
            numerator -= coefficients[k-1, j] * acf_values[k - j + 1]
            denominator -= coefficients[k-1, j] * acf_values[j + 1]
        end

        if abs(denominator) < 1e-10
            coefficients[k, k] = 0.0
        else
            coefficients[k, k] = numerator / denominator
        end
        pacf_vals[k] = coefficients[k, k]

        for j in 1:k-1
            coefficients[k, j] = coefficients[k-1, j] - coefficients[k, k] * coefficients[k-1, k-j]
        end
    end

    lags = collect(1:n_lags)
    ci = 1.96 / sqrt(n)

    return PACFResult(pacf_vals, lags, n, m, ci, :pacf)
end

"""
    Base.show(io::IO, result::ACFResult)

Pretty print an `ACFResult`.
"""
function Base.show(io::IO, result::ACFResult)
    println(io, "ACF Result")
    println(io, "  Series length: ", result.n)
    println(io, "  Frequency (m): ", result.m)
    println(io, "  Number of lags: ", length(result.lags) - 1)
    println(io, "  95% CI: ±", round(result.ci, digits=4))
    println(io, "  ACF values (first 10): ", round.(result.values[1:min(10, end)], digits=4))
    return
end

"""
    Base.show(io::IO, result::PACFResult)

Pretty print a `PACFResult`.
"""
function Base.show(io::IO, result::PACFResult)
    println(io, "PACF Result")
    println(io, "  Series length: ", result.n)
    println(io, "  Frequency (m): ", result.m)
    println(io, "  Number of lags: ", length(result.lags))
    println(io, "  95% CI: ±", round(result.ci, digits=4))
    println(io, "  PACF values (first 10): ", round.(result.values[1:min(10, end)], digits=4))
    return
end

"""
    plot(result::ACFResult; kwargs...)
    plot(result::PACFResult; kwargs...)

Plot ACF or PACF with confidence bands.

This function is implemented in the DurbynPlotsExt extension module.
Load Plots.jl to enable plotting: `using Plots`

# Example
```julia
using Plots
result = acf(y, 12)
plot(result)
```
"""
function plot end
