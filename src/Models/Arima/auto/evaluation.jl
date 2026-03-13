# ─── Candidate model evaluation for automatic selection ───────────────────────
#
# Mirrors core/fit.jl: standalone functions with explicit typed arguments for
# fitting a single candidate, computing information criteria, and checking
# polynomial root stability. The SearchConfig struct replaces kwargs forwarding.

"""
    format_trace(order, seasonal, m, constant; ic_value) -> String

Format a trace string for logging model evaluation progress.
"""
function format_trace(order::PDQ, seasonal::PDQ, m::Int, constant::Bool;
                      ic_value::Real = Inf)
    p, d, q = order.p, order.d, order.q
    P, D, Q = seasonal.p, seasonal.d, seasonal.q
    seasonal_part = (P + D + Q) > 0 && m > 0 ? "($P,$D,$Q)[$m]" : ""
    mean_str = if constant && (d + D == 0)
        " with non-zero mean"
    elseif constant && (d + D == 1)
        " with drift        "
    elseif !constant && (d + D == 0)
        " with zero mean    "
    else
        "                   "
    end
    ic_str = isnan(ic_value) ? "NaN" : (isfinite(ic_value) ? string(round(ic_value, digits = 3)) : "Inf")
    " ARIMA($(p),$(d),$(q))$(seasonal_part)$(mean_str) : $(ic_str)"
end

# Keep old name as alias for external callers
const arima_trace_str = format_trace

"""
    evaluate_candidate(x, m, candidate, config; nstar) -> ArimaFit

Fit a single ARIMA model, compute information criteria, and check polynomial
root stability. Takes a `SearchConfig` struct instead of kwargs.
Returns an ArimaFit (with ic=Inf if fitting fails or roots are near-unit).
"""
function evaluate_candidate(
    x::AbstractVector,
    m::Int,
    candidate::ModelCandidate,
    config::SearchConfig;
    nstar::Union{Nothing,Int} = nothing,
)
    order = PDQ(candidate.p, candidate.d, candidate.q)
    seasonal = PDQ(candidate.P, candidate.D, candidate.Q)

    # ── Effective series length ──
    n = _effective_series_length(x, candidate, m, nstar)

    # ── Choose estimation method ──
    fit_method = if !isnothing(config.method)
        config.method
    elseif config.approximation
        :css
    else
        :css_ml
    end

    # ── Build xreg with drift column if needed ──
    diffs = candidate.d + candidate.D
    drift_case = (diffs == 1) && candidate.constant
    fit_xreg = config.xreg

    if drift_case
        drift_vec = collect(1:length(x))
        if isnothing(fit_xreg)
            fit_xreg = NamedMatrix(reshape(drift_vec, :, 1), ["drift"])
        else
            fit_xreg = add_drift_term(fit_xreg, drift_vec, "drift")
        end
    end

    # ── Fit model ──
    use_season = (candidate.P + candidate.D + candidate.Q) > 0 && m > 0
    seasonal_arg = use_season ? seasonal : PDQ(0, 0, 0)

    fit = try
        if drift_case
            arima(x, m; order = order, seasonal = seasonal_arg,
                  xreg = fit_xreg, method = fit_method, config.arima_kwargs...)
        else
            arima(x, m; order = order, seasonal = seasonal_arg,
                  xreg = fit_xreg, method = fit_method,
                  include_mean = candidate.constant, config.arima_kwargs...)
        end
    catch err
        err
    end

    # ── Handle fitting failure ──
    if !(fit isa ArimaFit)
        errtxt = sprint(showerror, fit)
        if occursin("unused argument", errtxt)
            error(first(split(errtxt, '\n')))
        end
        if config.trace
            println()
            println(format_trace(order, seasonal, m, candidate.constant))
        end
        return _error_arimafit()
    end

    # ── Compute information criteria ──
    nstar_eff = n - candidate.d - candidate.D * m
    if drift_case
        fit.xreg = fit_xreg
    end
    npar = sum(fit.mask) + 1

    _compute_ic!(fit, config.ic, config.offset, fit_method, nstar_eff, npar)

    # ── Root stability check ──
    _check_root_stability!(fit, candidate)

    fit.xreg = fit_xreg

    if config.trace
        println()
        println(format_trace(order, seasonal, m, candidate.constant; ic_value = fit.ic))
    end

    return fit
end

"""Compute effective non-missing series length for IC calculations."""
function _effective_series_length(x::AbstractVector, candidate::ModelCandidate,
                                  m::Int, nstar::Union{Nothing,Int})
    if !isnothing(nstar)
        return nstar + candidate.d + candidate.D * m
    end
    first_idx = findfirst(xi -> !(ismissing(xi) || isnan(xi)), x)
    last_idx = findlast(xi -> !(ismissing(xi) || isnan(xi)), x)
    if isnothing(first_idx) || isnothing(last_idx)
        return 0
    end
    count(xi -> !(ismissing(xi) || isnan(xi)), @view x[first_idx:last_idx])
end

"""Compute AIC, BIC, AICc and set the selected IC on the fit."""
function _compute_ic!(fit::ArimaFit, ic::Symbol, offset::Float64,
                      method::Symbol, nstar::Int, npar::Int)
    if method === :css
        fit.aic = offset + nstar * log(fit.sigma2) + 2 * npar
    end

    if !isnan(fit.aic)
        fit.bic = fit.aic + npar * (log(nstar) - 2)
        if nstar <= npar + 1
            fit.aicc = Inf
            @warn "AICc not computable: insufficient observations (n=$nstar, npar=$npar)"
        else
            fit.aicc = fit.aic + 2 * npar * (npar + 1) / (nstar - npar - 1)
        end
        fit.ic = ic === :bic ? fit.bic : (ic === :aicc ? fit.aicc : fit.aic)
    else
        fit.aic, fit.bic, fit.aicc, fit.ic = Inf, Inf, Inf, Inf
    end
end

"""
Check that AR and MA polynomial roots are well outside the unit circle.
Sets fit.ic = Inf if any root has modulus < 1.01.
"""
function _check_root_stability!(fit::ArimaFit, candidate::ModelCandidate)
    minroot = 2.0

    if (candidate.p + candidate.P) > 0
        ar_root = _min_polynomial_root(fit.model.phi, negate = true)
        if isnothing(ar_root)
            fit.ic = Inf
            return
        end
        minroot = min(minroot, ar_root)
    end

    if (candidate.q + candidate.Q) > 0 && fit.ic < Inf
        ma_root = _min_polynomial_root(fit.model.theta, negate = false)
        if isnothing(ma_root)
            fit.ic = Inf
            return
        end
        minroot = min(minroot, ma_root)
    end

    if minroot < 1 + 1e-2
        fit.ic = Inf
    end
end

"""
Compute the minimum absolute root of a polynomial [1, ±coefs...].
Returns `nothing` on computation failure (caller must reject the model).
"""
function _min_polynomial_root(coefs::AbstractVector; negate::Bool)
    lastnz = 0
    @inbounds for i in length(coefs):-1:1
        if abs(coefs[i]) > 1e-8
            lastnz = i
            break
        end
    end
    lastnz == 0 && return 2.0

    sign_mult = negate ? -1.0 : 1.0
    poly = vcat(1.0, sign_mult .* @view(coefs[1:lastnz]))

    roots = try
        arima_polynomial_roots(poly)
    catch
        return nothing
    end

    isnothing(roots) && return nothing
    minimum(abs.(roots))
end
