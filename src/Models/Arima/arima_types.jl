"""
    ArimaStateSpace

A struct representing a univariate ARIMA state-space model, including AR and MA parameters, differencing, and state-space matrices.

# Fields
- `phi::AbstractVector`   : AR coefficients (length ≥ 0)
- `theta::AbstractVector` : MA coefficients (length ≥ 0)
- `Delta::AbstractVector` : Differencing coefficients (for seasonal/nonseasonal differences)
- `Z::AbstractVector`     : Observation coefficients
- `a::AbstractVector`     : Current state estimate
- `P::AbstractMatrix`     : Current state covariance matrix
- `T::AbstractMatrix`     : Transition/state evolution matrix
- `V::Any`                : Innovations or 'RQR', for process covariance
- `h::Real`               : Observation variance
- `Pn::AbstractMatrix`    : Prior state covariance at time t-1 (not updated by KalmanForecast)
"""
mutable struct ArimaStateSpace
    phi::AbstractVector
    theta::AbstractVector
    Delta::AbstractVector
    Z::AbstractVector
    a::AbstractVector
    P::AbstractMatrix
    T::AbstractMatrix
    V::Matrix{Float64}
    h::Real
    Pn::AbstractMatrix
end

"""
    KalmanWorkspace

Pre-allocated workspace arrays for Kalman filter computations in compute_arima_likelihood.
Reusing this workspace across multiple likelihood evaluations avoids repeated allocations
during optimization, significantly improving performance.

# Fields
- `anew::Vector{Float64}`: State prediction workspace (length rd)
- `M::Vector{Float64}`: Kalman gain workspace (length rd)
- `mm::Union{Matrix{Float64},Nothing}`: Covariance workspace (rd × rd), or nothing if d == 0
- `rsResid::Union{Vector{Float64},Nothing}`: Residuals workspace (length n), or nothing if not needed

# Constructor
    KalmanWorkspace(rd::Int, n::Int, d::Int, give_resid::Bool)

Creates a workspace with appropriately sized arrays.
"""
mutable struct KalmanWorkspace
    anew::Vector{Float64}
    M::Vector{Float64}
    mm::Union{Matrix{Float64},Nothing}
    rsResid::Union{Vector{Float64},Nothing}
end

function KalmanWorkspace(rd::Int, n::Int, d::Int, give_resid::Bool)
    anew = zeros(rd)
    M = zeros(rd)
    mm = d > 0 ? zeros(rd, rd) : nothing
    rsResid = give_resid ? zeros(n) : nothing
    return KalmanWorkspace(anew, M, mm, rsResid)
end

"""
    reset!(ws::KalmanWorkspace)

Reset workspace arrays to zero for reuse.
"""
function reset!(ws::KalmanWorkspace)
    fill!(ws.anew, 0.0)
    fill!(ws.M, 0.0)
    if !isnothing(ws.mm)
        fill!(ws.mm, 0.0)
    end
    if !isnothing(ws.rsResid)
        fill!(ws.rsResid, 0.0)
    end
    return ws
end

function show(io::IO, s::ArimaStateSpace)
    println(io, "ArimaStateSpace:")
    println(io, "  phi   (AR coefficients):         ", s.phi)
    println(io, "  theta (MA coefficients):         ", s.theta)
    println(io, "  Delta (Differencing coeffs):     ", s.Delta)
    println(io, "  Z     (Observation coeffs):      ", s.Z)
    println(io, "  a     (Current state estimate):  ", s.a)
    println(io, "  P     (Current state covariance):")
    show(io, "text/plain", s.P)
    println(io, "\n  T     (Transition matrix):")
    show(io, "text/plain", s.T)
    println(io, "\n  V     (Innovations or 'RQR'):    ", s.V)
    println(io, "  h     (Observation variance):    ", s.h)
    println(io, "  Pn    (Prior state covariance):")
    show(io, "text/plain", s.Pn)
end

"""
    ArimaFit

Holds the results of an ARIMA/SARIMA fit (including ARIMAX). This struct stores the data,
estimated parameters, likelihood and information criteria, residuals, and model metadata.

# Fields
- `y::AbstractArray`  
  The observed time-series data provided to the model.

- `fitted::AbstractArray`  
  In-sample fitted values.

- `coef::NamedMatrix`  
  Estimated coefficients (AR, MA, seasonal AR/MA, regressors). See `model.names` (or
  `model[:names]` if applicable) for parameter names and ordering.

- `sigma2::Float64`  
  Estimated innovation variance.

- `var_coef::Matrix{Float64}`  
  Variance-covariance matrix of the estimated coefficients.

- `mask::Vector{Bool}`  
  Indicates which parameters were estimated (vs. fixed/excluded).

- `loglik::Float64`  
  Maximized log-likelihood.

- `aic::Union{Float64,Nothing}`  
- `bic::Union{Float64,Nothing}`  
- `aicc::Union{Float64,Nothing}`  
  Information criteria computed from the fitted model (may be `nothing` if not applicable).

- `ic::Union{Float64,Nothing}`  
  The value of the information criterion selected for model comparison.

- `arma::Vector{Int}`  
  Compact model specification: `[p, q, P, Q, s, d, D]`.

- `residuals::Vector{Float64}`  
  Model residuals (estimated innovations).

- `convergence_code::Bool`  
  `true` if the optimizer reported successful convergence, `false` otherwise.

- `n_cond::Int`  
  Number of initial observations excluded due to conditioning (e.g., differencing).

- `nobs::Int`  
  Number of observations used in estimation (after differencing/trimming).

- `model::ArimaStateSpace`  
  State-space representation and model metadata.

- `xreg::Any`  
  Exogenous regressors matrix used in fitting (if any).

- `method::String`  
  Estimation method (e.g., `"ML"`, `"CSS"`). Stored as a descriptive string.

- `lambda::Union{Real,Nothing}`  
  Box-Cox transformation parameter used (if any).

- `biasadj::Bool`  
  Whether bias adjustment was applied when back-transforming from Box-Cox scale.

- `offset::Float64`  
  Constant offset applied internally (e.g., for transformed models).

# Usage
```julia
fit = auto_arima(y, 12)
fit.ic               # selected IC value
fit.arma             # [p, q, P, Q, s, d, D]
fit.coef             # parameter table with names
fit.residuals        # innovations
fit.model            # state-space representation
```

"""
mutable struct ArimaFit
    y::AbstractArray
    fitted::AbstractArray
    coef::NamedMatrix
    sigma2::Float64
    var_coef::Matrix{Float64}
    mask::Vector{Bool}
    loglik::Float64
    aic::Union{Float64,Nothing}
    bic::Union{Float64,Nothing}
    aicc::Union{Float64,Nothing}
    ic::Union{Float64,Nothing}
    arma::Vector{Int}
    residuals::Vector{Float64}
    convergence_code::Bool
    n_cond::Int
    nobs::Int
    model::ArimaStateSpace
    xreg::Union{NamedMatrix, Nothing}
    method::String
    lambda::Union{Real, Nothing}
    biasadj::Union{Bool, Nothing}
    offset::Union{Float64, Nothing}
end


function Base.show(io::IO, fit::ArimaFit)
    println(io, "ARIMA Fit Summary")
    println(io, "-----------------")

    println(io, "Coefficients:")
    show(io, fit.coef)  # use NamedMatrix's show for aligned table

    println(io, "\nSigma²: ", fit.sigma2)
    println(io, "Log-likelihood: ", fit.loglik)
    if !isnothing(fit.aic) && !isnan(fit.aic)
        println(io, "AIC: ", fit.aic)
    end
end


"""
A struct representing the parameters of an ARIMA model: autoregressive (`p`), differencing (`d`), and moving average (`q`) terms.

### Fields
- `p::Int`: Number of autoregressive (AR) terms.
- `d::Int`: Degree of differencing.
- `q::Int`: Number of moving average (MA) terms.

### Example
```julia
pdq_instance = PDQ(1, 0, 1)
println(pdq_instance)  # Output: PDQ(1, 0, 1)
```
"""
struct PDQ
    p::Int
    d::Int
    q::Int

    function PDQ(p::Int, d::Int, q::Int)
        if p < 0 || d < 0 || q < 0
            throw(
                ArgumentError(
                    "All PDQ parameters must be non-negative integers. Got: p=$p, d=$d, q=$q",
                ),
            )
        end
        new(p, d, q)
    end
end
