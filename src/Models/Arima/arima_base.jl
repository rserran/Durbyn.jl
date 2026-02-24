"""
    compute_css_residuals(y, arma, phi, theta, ncond)

Compute the conditional sum of squares (CSS) and residuals for an ARIMA model.

This routine mirrors the behaviour of the R function `ARIMA_CSS`.  It first
applies the appropriate differencing specified by the ARIMA order and then
computes the residuals of the ARMA model defined by `phi` and `theta`.  The
sum of squared residuals and the number of non-missing residuals are used
to compute the innovation variance estimate.  When called from the
high-level `arima` function, these residuals provide a fast approximate
estimate of the parameters prior to full maximum likelihood estimation.

Arguments
---------
- `y::AbstractArray`: the observed series (may contain missing values).
- `arma::Vector{Int}`: model specification `[p, q, P, Q, s, d, D]`.
- `phi::AbstractArray`: vector of non-seasonal AR coefficients.
- `theta::AbstractArray`: vector of non-seasonal MA coefficients.
- `ncond::Int`: number of initial observations to condition on.

Returns
-------
A `Dict` with keys `"sigma2"` giving the variance estimate and
`"resid"` giving the vector of residuals.
"""
# The function is tested works as expected
function compute_css_residuals(
    y::AbstractArray,
    arma::Vector{Int},
    phi::AbstractArray,
    theta::AbstractArray,
    ncond::Int,
)
    n = length(y)
    p = length(phi)
    q = length(theta)

    w = copy(y)

    for _ = 1:arma[6]
        for l = n:-1:2
            w[l] -= w[l-1]
        end
    end

    ns = arma[5]
    for _ = 1:arma[7]
        for l = n:-1:(ns+1)
            w[l] -= w[l-ns]
        end
    end

    resid = Vector{Float64}(undef, n)
    for i = 1:ncond
        resid[i] = 0.0
    end

    ssq = 0.0
    nu = 0

    for l = (ncond+1):n
        tmp = w[l]
        for j = 1:p
            if (l - j) < 1
                continue
            end
            tmp -= phi[j] * w[l-j]
        end

        jmax = min(l - ncond, q)
        for j = 1:jmax
            if (l - j) < 1
                continue
            end
            tmp -= theta[j] * resid[l-j]
        end

        resid[l] = tmp

        if !isnan(tmp)
            nu += 1
            ssq += tmp^2
        end
    end

    return (sigma2 = ssq / nu, residuals = resid)
end

"""
    initialize_arima_state(phi, theta, Delta; kappa=1e6, SSinit=:gardner1980, tol=eps(Float64))

Create and initialize the state-space representation of an ARIMA model.

Given vectors of AR coefficients `phi`, MA coefficients `theta`, and the differencing polynomial `Delta`, 
this function constructs all state-space matrices required for Kalman filtering and smoothing.  
This function mirrors the structure and logic of the corresponding C function used in R, and is used internally 
by high-level ARIMA fitting routines.

The initial state covariance matrix `Pn` is computed either by `compute_q0_covariance_matrix` (for `SSinit=:gardner1980`)
or by `compute_q0_bis_covariance_matrix` (for `SSinit=:rossignol2011`).

# Arguments
- `phi::Vector{Float64}`: Non-seasonal AR coefficients.
- `theta::Vector{Float64}`: Non-seasonal MA coefficients.
- `Delta::Vector{Float64}`: Differencing polynomial coefficients.
- `kappa::Float64`: Prior variance used to initialize the differenced states (default: `1e6`).
- `SSinit::Symbol`: Method for computing the initial covariance matrix (`:gardner1980` or `:rossignol2011`).
- `tol::Float64`: Tolerance parameter used by the Rossignol method.

# Returns
- An [`ArimaStateSpace`](@ref) struct containing the fields:
    - `phi`, `theta`, `Delta`, `Z`, `a`, `P`, `T`, `V`, `h`, `Pn`  
      (see [`ArimaStateSpace`](@ref) for field descriptions).

# Notes
- This function is intended for internal use, typically by higher-level ARIMA fitting routines.
- The returned struct can be used directly with Kalman filtering and smoothing algorithms.

# References
- Gardner, G., Harvey, A. C. & Phillips, G. D. A. (1980). Algorithm AS 154: An algorithm for exact maximum likelihood estimation of autoregressive-moving average models by means of Kalman filtering. *Applied Statistics*, 29, 311-322.
- Durbin, J. & Koopman, S. J. (2001). *Time Series Analysis by State Space Methods*. Oxford University Press.

"""
# The function is tested works as expected
function initialize_arima_state(phi::Vector{Float64}, theta::Vector{Float64}, Delta::Vector{Float64}; kappa::Float64=1e6, SSinit::Symbol=:gardner1980, tol::Float64=eps(Float64))
    p = length(phi)
    q = length(theta)
    r = max(p, q + 1)
    d = length(Delta)
    rd = r + d
    Z = vcat([1.0], zeros(r - 1), Delta)
    T = zeros(Float64, rd, rd)
    if p > 0
        for i = 1:p
            T[i, 1] = phi[i]
        end
    end
    if r > 1
        for i = 2:r
            T[i-1, i] = 1.0
        end
    end
    if d > 0
        T[r+1, :] = Z'
        if d > 1
            for i = 2:d
                T[r+i, r+i-1] = 1.0
            end
        end
    end
    if q < r - 1
        theta = vcat(theta, zeros(r - 1 - q))
    end
    R = vcat([1.0], theta, zeros(d))
    V = R * R'
    h = 0.0
    a = zeros(Float64, rd)
    P = zeros(Float64, rd, rd)
    Pn = zeros(Float64, rd, rd)
    if r > 1
        if SSinit === :gardner1980
            Pn[1:r, 1:r] = compute_q0_covariance_matrix(phi, theta)
        elseif SSinit === :rossignol2011
            Pn[1:r, 1:r] = compute_q0_bis_covariance_matrix(phi, theta, tol)
        else
            throw(ArgumentError("Invalid value for SSinit: :$SSinit"))
        end
    else
        if p > 0
            Pn[1, 1] = 1.0 / (1.0 - phi[1]^2)
        else
            Pn[1, 1] = 1.0
        end
    end
    if d > 0
        for i = r+1:r+d
            Pn[i, i] = kappa
        end
    end
    return ArimaStateSpace(phi, theta, Delta, Z, a, P, T, V, h, Pn)
end


"""
    process_xreg(xreg::Union{NamedMatrix, Nothing}, n::Int)

Process an exogenous regressor `xreg` (which may be `nothing` or a `NamedMatrix`).

Returns:
- `xreg::Matrix{Float64}`: the data matrix (guaranteed Float64 type)
- `ncxreg::Int`: number of columns in `xreg`
- `nmxreg::Vector{String}`: column names

# Arguments
- `xreg`: either `nothing` or a `NamedMatrix`
- `n`: number of rows expected

# Throws
- `ArgumentError` if the number of rows in `xreg` does not match `n`
"""
function process_xreg(xreg::Union{NamedMatrix,Nothing}, n::Int)
    if isnothing(xreg)
        xreg_mat = Matrix{Float64}(undef, n, 0)
        ncxreg = 0
        nmxreg = String[]
    else
        if size(xreg.data, 1) != n
            throw(ArgumentError("Lengths of x and xreg do not match!"))
        end
        xreg_mat = xreg.data
        # Ensure Float64 for integer inputs (e.g. dummy variables).
        if !(eltype(xreg_mat) <: Float64)
            xreg_mat = Float64.(xreg_mat)
        end
        ncxreg = size(xreg_mat, 2)
        nmxreg = xreg.colnames
    end
    return xreg_mat, ncxreg, nmxreg
end

"""
    regress_and_update!(x, xreg, narma, ncxreg, order_d, seasonal_d, m, Delta)

Regression block for exogenous regressors with missing value handling and coefficient scaling.
SVD rotation of xreg (when applicable) is handled by the caller in `arima()`.

# Arguments
- `x::AbstractArray`: Target variable (can contain NaN for missing values).
- `xreg::Matrix`: Regressor matrix (possibly SVD-rotated, can contain NaN for missing).
- `narma::Int`: Number of ARMA params.
- `ncxreg::Int`: Number of exogenous regressors.
- `order_d::Int`: Order of nonseasonal differencing.
- `seasonal_d::Int`: Order of seasonal differencing.
- `m::Int`: Seasonal period.
- `Delta::AbstractArray`: Differencing polynomial (used for n_used).

# Returns
Tuple: (`init0`, `parscale`, `n_used`)
"""
function regress_and_update!(
    x::AbstractArray,
    xreg::Matrix,
    narma::Int,
    ncxreg::Int,
    order_d::Int,
    seasonal_d::Int,
    m::Int,
    Delta::AbstractArray,
)

    init0 = zeros(narma)
    parscale = ones(narma)

    dx = copy(x)
    dxreg = copy(xreg)
    if order_d > 0
        dx = diff(dx; lag = 1, differences = order_d)
        dxreg = diff(dxreg; lag = 1, differences = order_d)
        dx, dxreg = dropmissing(dx, dxreg)
    end
    if m > 1 && seasonal_d > 0
        dx = diff(dx; lag = m, differences = seasonal_d)
        dxreg = diff(dxreg; lag = m, differences = seasonal_d)
        dx, dxreg = dropmissing(dx, dxreg)
    end

    if length(dx) > size(dxreg, 2)
        try
            fit = Stats.ols(dx, dxreg)
            fit_rank = rank(dxreg)
        catch e
            @warn "Fitting OLS to difference data failed: $e"
            fit = nothing
            fit_rank = 0
        end
    else
        @debug "Not enough observations to fit OLS" length_dx=length(dx) predictors=size(dxreg, 2)
        fit = nothing
        fit_rank = 0
    end

    if fit_rank == 0
        x_clean, xreg_clean = dropmissing(x, xreg)
        fit = Stats.ols(x_clean, xreg_clean)
    end

    has_na = isnan.(x) .| [any(isnan, row) for row in eachrow(xreg)]
    n_used = sum(.!has_na) - length(Delta)
    model_coefs = Stats.coefficients(fit)
    init0 = append!(init0, model_coefs)
    ses = fit.se
    parscale = append!(parscale, 10 * ses)

    return init0, parscale, n_used
end

"""
    prep_coefs(
        arma::Vector{Int}, 
        coef::AbstractArray, 
        cn::Vector{String}, 
        ncxreg::Int
    ) -> NamedMatrix

Construct a `NamedMatrix` representing model coefficients, assigning appropriate names to each coefficient 
according to AR, MA, seasonal, and exogenous (xreg) components.

# Arguments

- `arma::Vector{Int}`: A vector of length 4 specifying the orders of the model in the form 
  `[AR, MA, SAR, SMA]`, where:
    - `AR`: Number of non-seasonal autoregressive coefficients.
    - `MA`: Number of non-seasonal moving average coefficients.
    - `SAR`: Number of seasonal autoregressive coefficients.
    - `SMA`: Number of seasonal moving average coefficients.

- `coef::AbstractArray`: A one-dimensional array of coefficient values, ordered as specified by the model.

- `cn::Vector{String}`: Names of exogenous regressors (if any).

- `ncxreg::Int`: Number of exogenous regressors.

# Returns

- A `NamedMatrix` object containing the coefficients as a 1-row matrix, with column names 
  corresponding to parameter names such as `"ar1"`, `"ma1"`, `"sar1"`, `"sma1"`, and any 
  exogenous regressor names.

# Example

```julia
arma = [2, 1, 0, 0]                  # AR(2), MA(1), no seasonal
coef = [0.8, -0.3, 0.4, 1.2]         # AR1, AR2, MA1, xreg1
cn = ["xreg1"]                       # exogenous name(s)
ncxreg = 1

nm = prep_coefs(arma, coef, cn, ncxreg)
```
# Output: NamedMatrix with columns ["ar1", "ar2", "ma1", "xreg1"]
"""
function prep_coefs(arma::Vector{Int}, coef::AbstractArray, cn::Vector{String}, ncxreg::Int)
    names = String[]
    if arma[1] > 0
        append!(names, ["ar$(i)" for i in 1:arma[1]])
    end
    if arma[2] > 0
        append!(names, ["ma$(i)" for i in 1:arma[2]])
    end
    if arma[3] > 0
        append!(names, ["sar$(i)" for i in 1:arma[3]])
    end
    if arma[4] > 0
        append!(names, ["sma$(i)" for i in 1:arma[4]])
    end
    if ncxreg > 0
        append!(names, cn)
    end
    mat = reshape(coef, 1, :)
    return NamedMatrix(mat, names)
end

function update_arima(mod::ArimaStateSpace, phi, theta; ss_g=true)
    p = length(phi)
    q = length(theta)
    r = max(p, q + 1)

    mod.phi = phi
    mod.theta = theta

    if p > 0
        mod.T[1:p, 1] .= phi
    end

    if r > 1
        if ss_g
            mod.Pn[1:r, 1:r] .= compute_q0_covariance_matrix(phi, theta)
        else
            mod.Pn[1:r, 1:r] .= compute_q0_bis_covariance_matrix(phi, theta, 0.0)
        end
    else
        if p > 0
            mod.Pn[1, 1] = 1 / (1 - phi[1]^2)
        else
            mod.Pn[1, 1] = 1.0
        end
    end

    mod.a .= 0.0
    return mod
end

# Check AR polynomial stationarity
function ar_check(ar)
    v = vcat(1.0, -ar...)
    last_nz = findlast(x -> x != 0.0, v)
    p = isnothing(last_nz) ? 0 : last_nz - 1
    if p == 0
        return true
    end

    coeffs = vcat(1.0, -ar[1:p]...)
    rts = roots(Polynomial(coeffs))

    return all(abs.(rts) .> 1.0)
end

# Invert MA polynomial
function ma_invert(ma)
    q = length(ma)
    cdesc = vcat(1.0, ma...)
    nz = findall(x -> x != 0.0, cdesc)
    q0 = isempty(nz) ? 0 : maximum(nz) - 1
    if q0 == 0
        return ma
    end
    cdesc_q = cdesc[1:q0+1]
    rts = roots(Polynomial(cdesc_q))
    ind = abs.(rts) .< 1.0
    if all(.!ind)
        return ma
    end
    if q0 == 1
        return vcat(1.0 / ma[1], zeros(q - q0))
    end
    rts[ind] .= 1.0 ./ rts[ind]
    x = [1.0]
    for r in rts
        x = vcat(x, 0.0) .- (vcat(0.0, x) ./ r)
    end
    return vcat(real.(x[2:end]), zeros(q - q0))
end

function arima(
    x::AbstractArray,
    m::Int;
    order::PDQ = PDQ(0, 0, 0),
    seasonal::PDQ = PDQ(0, 0, 0),
    xreg::Union{Nothing, NamedMatrix} = nothing,
    include_mean::Bool = true,
    transform_pars::Bool = true,
    fixed::Union{Nothing, AbstractArray} = nothing,
    init::Union{Nothing, AbstractArray}= nothing,
    method::Symbol = :css_ml,
    n_cond::Union{Nothing, AbstractArray} = nothing,
    SSinit::Symbol = :gardner1980,
    optim_method::Symbol = :bfgs,
    optim_control::Dict = Dict(),
    kappa::Real = 1e6,)

    _check_arg(SSinit, (:gardner1980, :rossignol2011), "SSinit")
    _check_arg(method, (:css_ml, :ml, :css), "method")
    SS_G = SSinit === :gardner1980

    kalman_ws = Ref{Union{KalmanWorkspace,Nothing}}(nothing)

    # State-space likelihood
    function compute_state_space_likelihood(y, model)
        # Delegate to the Kalman filter based likelihood computation.
        return compute_arima_likelihood(y, model, 0, true)
    end

    # Objective for ML optimization
    function armafn(p, trans)
        par = copy(coef)
        par[mask] = p
        trarma = transform_arima_parameters(par, arma, trans)
        xxi = copy(x)

        # Update state-space model with new parameters (protected by try-catch)
        Z = try
            update_arima(mod, trarma[1], trarma[2]; ss_g=SS_G)
        catch e
            @warn "Updating arima failed $e"
            return typemax(Float64)
        end

        if ncxreg > 0
            xxi = xxi .- xreg * par[narma+1 : narma+ncxreg]
        end
        resss = compute_arima_likelihood(xxi, Z, 0, false; workspace=kalman_ws[])

        # Safety checks to avoid NaN/Inf in optimization
        nu = resss[3]
        if nu <= 0
            return typemax(Float64)
        end

        s2 = resss[1] / nu
        if s2 < 0 || isnan(s2) || s2 == Inf
            return typemax(Float64)
        end

        result = 0.5 * (log(s2) + resss[2] / nu)
        # NaN/+Inf → bad parameters; -Inf → perfect fit (s2=0)
        return isnan(result) || result == Inf ? typemax(Float64) : result
    end
    # Conditional sum of squares objective
    function armaCSS(p)
        par = copy(fixed)
        par[mask] .= p
        trarma = transform_arima_parameters(par, arma, false)
        x_in = copy(x)

        if ncxreg > 0
            x_in = x_in .- xreg * par[narma+1 : narma+ncxreg]
        end

        ross = compute_css_residuals(x_in, arma, trarma[1], trarma[2], ncond)
        sigma2 = ross[:sigma2]

        # sigma2 < 0 or NaN/Inf → bad parameters
        # sigma2 == 0 → perfect fit (log returns -Inf)
        if sigma2 < 0 || isnan(sigma2) || sigma2 == Inf
            return typemax(Float64)
        end

        result = 0.5 * log(sigma2)
        # NaN/+Inf → bad parameters; -Inf → perfect fit (sigma2=0)
        return isnan(result) || result == Inf ? typemax(Float64) : result
    end
    
    n = length(x)
    y = copy(x)

    arma = [order.p, order.q, seasonal.p, seasonal.q, m, order.d, seasonal.d]

    narma = sum(arma[1:4])

    # Build Delta
    Delta = [1.0]

    for _ = 1:order.d
        Delta = time_series_convolution(Delta, [1.0, -1.0])
    end

    for _ = 1:seasonal.d
        seasonal_filter = [1.0; zeros(m - 1); -1.0]
        Delta = time_series_convolution(Delta, seasonal_filter)
    end

    Delta = -Delta[2:end]

    nd = order.d + seasonal.d
    n_used = length(dropmissing(x)) - length(Delta)

    xreg_original = xreg

    if include_mean && (nd == 0)
        if isnothing(xreg)
            xreg = NamedMatrix(zeros(n, 0), String[])
        end
        xreg = add_drift_term(xreg, ones(n), "intercept")
    end

    xreg, ncxreg, nmxreg = process_xreg(xreg, n)

    if method === :css_ml
        has_missing = xi -> (ismissing(xi) || isnan(xi))
        anyna = any(has_missing, x)
        if ncxreg > 0
            anyna |= any(has_missing, xreg)
        end
        if anyna
            method = :ml
        end
    end

    if method in (:css, :css_ml)
        ncond = order.d + seasonal.d * m
        ncond1 = order.p + seasonal.p * m

        if isnothing(n_cond)
            ncond += ncond1
        else
            ncond += max(n_cond, ncond1)
        end
    else
        ncond = 0
    end

    # Handle fixed
    if isnothing(fixed)
        fixed = fill(NaN, narma + ncxreg)
    elseif length(fixed) != narma + ncxreg
        throw(ArgumentError("Wrong length for 'fixed'"))
    end
    mask = isnan.(fixed)
    no_optim = !any(mask)

    if no_optim
        transform_pars = false
    end

    if transform_pars
        ind = arma[1] + arma[2] .+ (1:arma[3])

        if any(.!mask[1:arma[1]]) || any(.!mask[ind])
            @warn "Some AR parameters were fixed: Setting transform_pars = false"
            transform_pars = false
        end
    end

    # SVD rotation for multi-regressor numerical stability (must happen in
    # arima() scope so that armafn/armaCSS closures capture the rotated xreg).
    orig_xreg = true
    S = nothing
    if ncxreg > 0
        orig_xreg = (ncxreg == 1) || any(.!mask[(narma+1):(narma+ncxreg)])
        if !orig_xreg
            rows_good = [all(isfinite, row) for row in eachrow(xreg)]
            S = svd(xreg[rows_good, :])
            xreg = xreg * S.V
        end
    end

    # estimate init and scale
    if ncxreg > 0
        init0, parscale, n_used =
        regress_and_update!(x, xreg, narma, ncxreg, order.d, seasonal.d, m, Delta)
    else
        init0 = zeros(narma)
        parscale = ones(narma)
    end

    if n_used <= 0
        throw(ArgumentError("Too few non-missing observations"))
    end

    if !isnothing(init)
        if length(init) != length(init0)
            throw(ArgumentError("'init' is of the wrong length"))
        end

        ind = map(x -> isnan(x) || ismissing(x), init)
        if any(ind)
           init[ind] .= init0[ind] 
        end

        if method === :ml
            p = arma[1]  # non-seasonal AR order
            P = arma[3]  # seasonal AR order
            if p > 0
                if !ar_check(init[1:p])
                    error("non-stationary AR part")
                end
            end
            if P > 0
                # Seasonal AR params are at positions (p + q + 1) : (p + q + P)
                sa_start = arma[1] + arma[2] + 1
                sa_stop = arma[1] + arma[2] + P
                if !ar_check(init[sa_start:sa_stop])
                    error("non-stationary seasonal AR part")
                end
            end
        end
    else
        init = copy(init0)
    end

    coef = copy(Float64.(fixed))

    if method === :css
        if no_optim
            res = (converged = true, minimizer = zeros(0), minimum = armaCSS(zeros(0)))
        else
            ctrl = copy(optim_control)
            ctrl["parscale"] = parscale[mask]
            # CSS needs larger finite-difference step and more iterations
            if !haskey(ctrl, "ndeps")
                ctrl["ndeps"] = fill(1e-2, sum(mask))
            end
            if !haskey(ctrl, "maxit")
                ctrl["maxit"] = 500
            end

            opt = optimize(
                init[mask],
                p -> armaCSS(p);
                method = optim_method,
                control = ctrl,
            )
            res = (
                converged = opt.convergence == 0,
                minimizer = opt.par,
                minimum = opt.value,
            )
        end

        if !res.converged
            @warn "CSS optimization convergence issue: convergence code $(opt.convergence)"
        end

        coef[mask] .= res.minimizer

        trarma = transform_arima_parameters(coef, arma, false)
        mod = initialize_arima_state(
            trarma[1],
            trarma[2],
            Delta;
            kappa = kappa,
            SSinit = SSinit,
        )
        
        if ncxreg > 0
            x = x - xreg * coef[narma+1 : narma+ncxreg]
        end
        # Change a in mod
        compute_state_space_likelihood(x, mod)

        val = compute_css_residuals(x, arma, trarma[1], trarma[2], ncond)
        sigma2 = val[:sigma2]


        if no_optim
            var = zeros(0)
        else
            hessian = numerical_hessian(p -> armaCSS(p), res.minimizer)
            var = inv(hessian * n_used)
        end

    else
        if method in (:css_ml, :ml)
            # CSS pre-initialization for better starting values.
            # For ML, temporarily set ncond for CSS computation.
            if method === :ml
                ncond = order.d + seasonal.d * m
                ncond1 = order.p + seasonal.p * m
                ncond += isnothing(n_cond) ? ncond1 : max(n_cond, ncond1)
            end

            if no_optim
                res = (
                    converged = true,
                    minimizer = zeros(sum(mask)),
                    minimum = armaCSS(zeros(0)),
                )
            else
                ctrl = copy(optim_control)
                ctrl["parscale"] = parscale[mask]
                # CSS pre-init needs larger finite-difference step and more iterations
                if !haskey(ctrl, "ndeps")
                    ctrl["ndeps"] = fill(1e-2, sum(mask))
                end
                if !haskey(ctrl, "maxit")
                    ctrl["maxit"] = 500
                end

                opt = optimize(
                    init[mask],
                    p -> armaCSS(p);
                    method = optim_method,
                    control = ctrl,
                )
                res = (
                    converged = opt.convergence == 0,
                    minimizer = opt.par,
                    minimum = opt.value,
                )
            end

            if res.converged
                init[mask] .= res.minimizer
            end

            if arma[1] > 0 && !ar_check(init[1:arma[1]])
                error("Non-stationary AR part from CSS")
            end

            if arma[3] > 0 && !ar_check(init[(sum(arma[1:2]) + 1):(sum(arma[1:2]) + arma[3])])
                error("Non-stationary seasonal AR part from CSS")
            end

            ncond = 0
        end

        if transform_pars
            init = inverse_arima_parameter_transform(init, arma)

            if arma[2] > 0
                ind = (arma[1]+1):(arma[1]+arma[2])
                init[ind] .= ma_invert(init[ind])
            end

            if arma[4] > 0
                ind = (sum(arma[1:3]) + 1) : (sum(arma[1:3]) + arma[4])
                init[ind] .= ma_invert(init[ind])
            end
        end

        trarma = transform_arima_parameters(init, arma, transform_pars)
        mod = initialize_arima_state(
            trarma[1],
            trarma[2],
            Delta;
            kappa = kappa,
            SSinit = SSinit,
        )

        # Initialize Kalman workspace after mod is created (for optimization reuse)
        rd = length(mod.a)
        d_len = length(mod.Delta)
        kalman_ws[] = KalmanWorkspace(rd, n, d_len, false)

        if no_optim

            res = (
                converged = true,
                minimizer = zeros(0),
                minimum = armafn(zeros(0), transform_pars),
            )
        else
            ctrl = copy(optim_control)
            ctrl["parscale"] = parscale[mask]

            opt = optimize(
                init[mask],
                p -> armafn(p, transform_pars);
                method = optim_method,
                control = ctrl,
            )
            res = (
                converged = opt.convergence == 0,
                minimizer = opt.par,
                minimum = opt.value,
            )
        end

        if !res.converged
            @warn "Possible convergence problem: convergence code $(opt.convergence)"
        end

        coef[mask] .= res.minimizer

        if transform_pars
            if arma[2] > 0
                ind = (arma[1]+1):(arma[1]+arma[2])
                if all(mask[ind])
                    coef[ind] .= ma_invert(coef[ind])
                end
            end

            if arma[4] > 0
                ind = (sum(arma[1:3]) + 1) : (sum(arma[1:3]) + arma[4])
                if all(mask[ind])
                    coef[ind] .= ma_invert(coef[ind])
                end
            end

            if any(coef[mask] .!= res.minimizer)
                _old_convergence = res.converged  # Preserved for potential future use

                ctrl = copy(optim_control)
                ctrl["parscale"] = parscale[mask]
                ctrl["maxit"] = 0

                opt = optimize(
                    coef[mask],
                    p -> armafn(p, true);
                    method = optim_method,
                    control = ctrl,
                )
                res = (
                    converged = opt.convergence == 0,
                    minimizer = opt.par,
                    minimum = opt.value,
                )

                hessian = numerical_hessian(p -> armafn(p, true), res.minimizer)

                coef[mask] .= res.minimizer
            else
                hessian = numerical_hessian(p -> armafn(p, true), res.minimizer)
            end

            A = compute_arima_transform_gradient(coef, arma)
            A = A[mask, mask]
            var = A' * ((hessian * n_used) \ A)
            coef = undo_arima_parameter_transform(coef, arma)
        else
            if no_optim
                var = zeros(0)
            else
                hessian = numerical_hessian(p -> armafn(p, true), res.minimizer)
                var = inv(hessian * n_used)
            end
        end

        trarma = transform_arima_parameters(coef, arma, false)
        mod = initialize_arima_state(
            trarma[1],
            trarma[2],
            Delta;
            kappa = kappa,
            SSinit = SSinit,
        )

        val = if ncxreg > 0
            compute_state_space_likelihood(x - xreg * coef[narma+1 : narma+ncxreg], mod)
        else
            compute_state_space_likelihood(x, mod)
        end
        sigma2 = val[1][1] / n_used
    end

    # # Final steps
    value = 2 * n_used * res.minimum + n_used + n_used * log(2 * π)
    
    if method !== :css
        aic = value + 2 * sum(mask) + 2
    else
        aic = nothing
    end
    loglik = -0.5 * value

    if ncxreg > 0 && !orig_xreg
        ind = narma .+ (1:ncxreg)
        coef[ind] = S.V * coef[ind]
        A = Matrix{Float64}(I, narma + ncxreg, narma + ncxreg)
        A[ind, ind] = S.V
        A = A[mask, mask]
        var = A * var * transpose(A)
    end

    arima_coef = prep_coefs(arma, coef, nmxreg, ncxreg)
    resid = val[:residuals]
    fitted = y .- resid

    if ncxreg > 0
        fit_method = "Regression with ARIMA($(order.p),$(order.d),$(order.q))(" * 
        "$(seasonal.p),$(seasonal.d),$(seasonal.q))[$m]" * 
        " errors"
    else
        fit_method = "ARIMA($(order.p),$(order.d),$(order.q))(" * 
        "$(seasonal.p),$(seasonal.d),$(seasonal.q))[$m]"
    end
    
    if size(var) == (0, )
        var = reshape(var, 0, 0)
    end
    result = ArimaFit(
        y,
        fitted,
        arima_coef,
        sigma2,
        var,
        mask,
        loglik,
        aic,
        nothing,
        nothing,
        nothing,
        arma,
        resid,
        res.converged,
        ncond,
        n_used,
        mod,
        xreg_original,
        fit_method,
        nothing,
        nothing,
        nothing,
    )
    return result

end

"""
    kalman_forecast(n_ahead::Int, mod::ArimaStateSpace; update::Bool=false)

Forecast n steps ahead from the current state of the ARIMA state-space model `mod`.
Returns a NamedTuple with fields:
- `pred`: Vector of n_ahead predictions.
- `var`: Vector of corresponding (unscaled) prediction variances.
If `update` is true, the updated model is also returned in the NamedTuple as `mod`.
"""
function kalman_forecast(n_ahead::Int, mod::ArimaStateSpace; update::Bool=false)
    phi = mod.phi
    theta = mod.theta
    delta = mod.Delta
    Z = mod.Z
    a = copy(mod.a)
    P = copy(mod.P)
    Pnew = copy(mod.Pn)
    h = mod.h

    p = length(phi)
    q = length(theta)
    d = length(delta)
    rd = length(a)
    r = rd - d

    #a[1:r] .= 0.0

    forecasts = Vector{Float64}(undef, n_ahead)
    variances = Vector{Float64}(undef, n_ahead)

    anew = similar(a)
    mm = d > 0 ? zeros(rd, rd) : nothing

    for l in 1:n_ahead
        # 1. State prediction: a = T * a
        state_prediction!(anew, a, p, r, d, rd, phi, delta)
        a .= anew

        # 2. Forecast value
        fc = dot(Z, a)
        forecasts[l] = fc

        # 3. Covariance prediction: Pnew = T * P * T' + V (before computing variance)
        if d == 0
            predict_covariance_nodiff!(Pnew, P, r, p, q, phi, theta)
        else
            predict_covariance_with_diff!(Pnew, P, r, d, p, q, rd, phi, delta, theta, mm)
        end

        # 4. Compute variance: h + Z' * Pnew * Z
        tmpvar = h + dot(Z, Pnew, Z)
        variances[l] = tmpvar

        # 5. Update P for next iteration
        P .= Pnew
    end

    result = (pred = forecasts, var = variances)
    if update
        updated_mod = deepcopy(mod)
        updated_mod.a .= a
        updated_mod.P .= P
        result = merge(result, (; mod = updated_mod))
    end
    return result
end

struct ArimaPredictions
    prediction::Vector{Float64}
    se::Vector{Float64}
    y::AbstractVector
    fitted::Vector{Float64}
    residuals::Vector{Float64}
    method::String
end


function predict_arima(model::ArimaFit, n_ahead::Int=1;
    newxreg::Union{Nothing, NamedMatrix}= nothing, se_fit::Bool=true)

    myncol(x) = isnothing(x) ? 0 : size(x, 2)

    # Validate xreg consistency
    if !isnothing(newxreg) && isnothing(model.xreg)
        throw(ArgumentError("newxreg provided but model was fit without exogenous regressors"))
    end

    if newxreg isa NamedMatrix
        newxreg = align_columns(newxreg, model.xreg.colnames)
        newxreg = newxreg.data
    end

    arma = model.arma
    coefs = vec(model.coef.data)
    coef_names = model.coef.colnames
    narma = sum(arma[1:4])
    ncoefs = length(coefs)

    intercept_idx = findfirst(==("intercept"), coef_names)
    has_intercept = !isnothing(intercept_idx)

    ncxreg = model.xreg isa NamedMatrix ? size(model.xreg.data, 2) : 0
    if myncol(newxreg) != ncxreg
        throw(ArgumentError("`xreg` and `newxreg` have different numbers of columns"))
    end
    xm = zeros(n_ahead)
    if ncoefs > narma
        if has_intercept && coef_names[narma+1] == "intercept"
            intercept_col = ones(n_ahead, 1)
            usexreg = isnothing(newxreg) ? intercept_col : hcat(intercept_col, newxreg)
            reg_coef_inds = (narma+1):ncoefs
        else
            usexreg = newxreg
            reg_coef_inds = (narma+1):ncoefs
        end
        if narma == 0
            xm = vec(usexreg * coefs)
        else
            xm = vec(usexreg * coefs[reg_coef_inds])
        end
    end

    pred, se = kalman_forecast(n_ahead, model.model, update=false)
    
    pred = pred .+ xm
    if se_fit
        se = sqrt.(se .* model.sigma2)
    else
        se = fill(NaN, length(pred))
    end

    return ArimaPredictions(pred, se, model.y, model.fitted, model.residuals, model.method)

end


function fitted(model::ArimaFit)
    return model.fitted
end

function residuals(model::ArimaFit)
    return model.residuals
end
