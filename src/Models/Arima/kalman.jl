# helper for compute_arima_likelihood
function state_prediction!(anew::AbstractArray, a::AbstractArray, p::Int, r::Int, d::Int, rd::Int, phi::AbstractArray, delta::AbstractArray)
     @inbounds for i in 1:r
        tmp = (i < r) ? a[i + 1] : 0.0
        if i <= p
            tmp += phi[i] * a[1]
        end
        anew[i] = tmp
    end

    if d > 0
        @inbounds for i in (r + 2):(rd)
            anew[i] = a[i - 1]
        end
        tmp = a[1]
        @inbounds for i in 1:d
            tmp += delta[i] * a[r + i]
        end
        anew[r + 1] = tmp
    end
end
# helper for compute_arima_likelihood
function predict_covariance_nodiff!(Pnew::Matrix{Float64}, P::Matrix{Float64},
    r::Int, p::Int, q::Int,
    phi::Vector{Float64}, theta::Vector{Float64})
    @inbounds for i in 1:r

        if i == 1
            vi = 1.0
        elseif i - 1 <= q
            vi = theta[i-1]
        else
            vi = 0.0
        end

        for j in 1:r

            if j == 1
                tmp = vi
            elseif j - 1 <= q
                tmp = vi * theta[j-1]
            else
                tmp = 0.0
            end

            if i <= p && j <= p
                tmp = tmp + phi[i] * phi[j] * P[1, 1]
            end

            if i <= r - 1 && j <= r - 1
                tmp = tmp + P[i+1, j+1]
            end

            if i <= p && j <= r - 1
                tmp = tmp + phi[i] * P[j+1, 1]
            end

            if j <= p && i <= r - 1
                tmp = tmp + phi[j] * P[i+1, 1]
            end

            Pnew[i, j] = tmp
        end
    end
end
# helper for compute_arima_likelihood
function predict_covariance_with_diff!(Pnew::Matrix{Float64}, P::Matrix{Float64},
    r::Int, d::Int, p::Int, q::Int, rd::Int,
    phi::Vector{Float64}, delta::Vector{Float64},
    theta::Vector{Float64}, mm::Matrix{Float64})
    # Step 1: mm = T * P
    @inbounds for i in 1:r
        for j in 1:rd
            tmp = 0.0
            if i <= p
                tmp = tmp + phi[i] * P[1, j]
            end
            if i < r
                tmp = tmp + P[i+1, j]
            end
            mm[i, j] = tmp
        end
    end

    @inbounds for j in 1:rd
        tmp = P[1, j]
        for k in 1:d
            tmp = tmp + delta[k] * P[r+k, j]
        end
        mm[r+1, j] = tmp
    end

    @inbounds for i in 2:d
        for j in 1:rd
            mm[r+i, j] = P[r+i-1, j]
        end
    end

    # Step 2: Pnew = mm * Tᵀ
    # Column-major traversal: indices are transposed compared to row-major reading
    @inbounds for i in 1:r
        for j in 1:rd
            tmp = 0.0
            if i <= p
                tmp = tmp + phi[i] * mm[j, 1]
            end
            if i < r
                tmp = tmp + mm[j, i+1]
            end
            Pnew[j, i] = tmp
        end
    end

    @inbounds for j in 1:rd
        tmp = mm[j, 1]
        for k in 1:d
            tmp = tmp + delta[k] * mm[j, r+k]
        end
        Pnew[j, r+1] = tmp
    end

    @inbounds for i in 2:d
        for j in 1:rd
            Pnew[j, r+i] = mm[j, r+i-1]
        end
    end

    # Step 3: Add noise (MA(q))
    @inbounds for i in 1:(q+1)
        if i == 1
            vi = 1.0
        else
            vi = theta[i-1]
        end

        for j in 1:(q+1)
            if j == 1
                vj = 1.0
            else
                vj = theta[j-1]
            end
            Pnew[i, j] = Pnew[i, j] + vi * vj
        end
    end
end
# helper for compute_arima_likelihood
# This is a bit confusing: C code uses row major operations. Pnew[i + r * j]
function kalman_update!(y_obs, anew, delta, Pnew, M, d, r, rd, a, P, useResid, rsResid, l, ssq, sumlog, nu,)

    # 1) residual
    resid = y_obs - anew[1]
    @inbounds for i in 1:d
        resid = resid - delta[i] * anew[r+i]
    end

    # 2) build M = Pnew * [1; delta]
    @inbounds for i in 1:rd
        tmp = Pnew[i, 1]
        for j in 1:d
            tmp += Pnew[i, r+j] * delta[j]
        end
        M[i] = tmp
    end

    # 3) compute gain = H* M
    gain = M[1]
    @inbounds for j in 1:d
        gain += delta[j] * M[r+j]
    end

    # 4) update ssq, sumlog, nu if gain is "safe"
    #    Guard gain > 0: C's log(negative) returns NaN silently;
    #    Julia throws DomainError. Use NaN to match C behaviour.
    if gain < 1e4
        nu[] += 1
        ssq[] += resid^2 / gain
        sumlog[] += gain > 0 ? log(gain) : NaN
    end

    # 5) store standardized innovation: resid / sqrt(gain)
    if useResid
        rsResid[l] = gain > 0 ? resid / sqrt(gain) : NaN
    end

    # 6) state update: a = anew + (M * resid)/gain
    @inbounds for i in 1:rd
        a[i] = anew[i] + M[i] * resid / gain
    end

    # 7) covariance update: P = Pnew - (M Mᵀ)/gain
    @inbounds for i = 1:rd
        for j = 1:rd
            P[i, j] = Pnew[i, j] - (M[i] * M[j]) / gain
        end
    end
end

"""
    compute_arima_likelihood(y::Vector{Float64},
                             model::ArimaStateSpace,
                             update_start::Int,
                             give_resid::Bool)

Compute the Gaussian log-likelihood and related quantities for a univariate ARIMA model using the Kalman filter.

It runs a Kalman filter on the observed time series `y`, using the state-space representation stored in `model`.
It accumulates the innovation sum of squares and the log-determinant contributions required for the Gaussian likelihood.
If `give_resid` is true, the function also computes and returns the raw residuals (innovations).

# Arguments
- `y::Vector{Float64}`: Observed time series (univariate).
- `model::ArimaStateSpace`: State-space model, as returned by `initialize_arima_state`.
- `update_start::Int`: The time index at which to begin updating the likelihood and residuals (typically 1).
- `give_resid::Bool`: If true, also compute and return residuals (raw innovations).

# Returns
A `Dict` with keys:
- `"ssq"`: Sum of squared innovations.
- `"sumlog"`: Accumulated log-determinants of the prediction error variances.
- `"nu"`: Innovations (prediction errors).
- `"resid"`: (only if `give_resid` is true) Raw residuals (innovations, i.e., prediction errors).

# Notes
- The arguments and behavior closely follow the C implementation in R's base ARIMA code.
- For details on the state-space representation, see [`initialize_arima_state`](@ref).

# References
- Durbin, J. & Koopman, S. J. (2001). *Time Series Analysis by State Space Methods*. Oxford University Press.
- Gardner, G., Harvey, A. C. & Phillips, G. D. A. (1980). Algorithm AS 154. *Applied Statistics*, 29, 311-322.

"""
# Tested and it is safe. Possible improvement potatial.
function compute_arima_likelihood(
    y::Vector{Float64},
    model::ArimaStateSpace,
    update_start::Int,
    give_resid::Bool;
    workspace::Union{KalmanWorkspace,Nothing}=nothing
)

    phi = model.phi
    theta = model.theta
    delta = model.Delta
    # Use references (not copies) for in-place modification of model state.
    a = model.a
    P = model.P
    Pnew = model.Pn

    n = length(y)
    rd = length(a)
    p = length(phi)
    q = length(theta)
    d = length(delta)
    r = rd - d

    ssq = Ref(0.0)
    sumlog = Ref(0.0)
    nu = Ref(0)

    if isnothing(workspace)
        anew = zeros(rd)
        M = zeros(rd)
        mm = d > 0 ? zeros(rd, rd) : nothing
        rsResid = give_resid ? zeros(n) : nothing
    else
        # Reset workspace arrays for reuse
        reset!(workspace)
        anew = workspace.anew
        M = workspace.M
        mm = workspace.mm
        rsResid = workspace.rsResid
    end
    @inbounds for l = 1:n
        state_prediction!(anew, a, p, r, d, rd, phi, delta)

        if l > update_start + 1
            if d == 0
                predict_covariance_nodiff!(Pnew, P, r, p, q, phi, theta)
            else
                predict_covariance_with_diff!(Pnew, P, r, d, p, q, rd, phi, delta, theta, mm)
            end
        end

        if !isnan(y[l])
            
            kalman_update!(y[l], anew, delta, Pnew, M, d, r, rd, a, P, give_resid, rsResid, l, ssq, sumlog, nu)
        else
            a .= anew
            copyto!(P, Pnew)
            if give_resid
                rsResid[l] = NaN
            end
        end
    end

    result_stats = [ssq[], sumlog[], nu[]]

    if give_resid
        return (stats = result_stats, residuals = rsResid)
    else
        return result_stats
    end
end
