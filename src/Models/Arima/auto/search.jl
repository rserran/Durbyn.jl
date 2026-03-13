# ─── Search algorithms for automatic ARIMA model selection ───────────────────
#
# Mirrors core/kalman.jl pattern: computational inner loops in their own file,
# operating on typed structs (SearchConfig, SearchBounds, ModelHistory).
# All functions are standalone with explicit arguments — no closures.

# ─── Neighbor generation ─────────────────────────────────────────────────────

"""
    _apply_move(base, move, bounds) -> Union{ModelCandidate, Nothing}

Apply a (Δp, Δq, ΔP, ΔQ) move to a candidate, returning `nothing` if the
result violates bounds.
"""
function _apply_move(base::ModelCandidate, move::NTuple{4,Int}, bounds::SearchBounds)
    new_p = base.p + move[1]
    new_q = base.q + move[2]
    new_P = base.P + move[3]
    new_Q = base.Q + move[4]

    (new_p < 0 || new_p > bounds.max_p) && return nothing
    (new_q < 0 || new_q > bounds.max_q) && return nothing
    (new_P < 0 || new_P > bounds.max_P) && return nothing
    (new_Q < 0 || new_Q > bounds.max_Q) && return nothing

    ModelCandidate(new_p, base.d, new_q, new_P, base.D, new_Q, base.constant)
end

# ─── Single candidate trial ──────────────────────────────────────────────────

"""
    _try_candidate(x, m, candidate, history, config, results; nstar) -> Union{ArimaFit, Nothing}

Register a candidate in history, evaluate it if new, record the result,
return nothing if already seen or budget exceeded.
Standalone — no closure, no captured state.
"""
function _try_candidate(
    x::AbstractVector,
    m::Int,
    candidate::ModelCandidate,
    history::ModelHistory,
    config::SearchConfig,
    results::SearchResults;
    nstar::Union{Nothing,Int} = nothing,
)
    try_register!(history, candidate) || return nothing
    n_evaluated(history) > config.nmodels && return nothing
    fit = evaluate_candidate(x, m, candidate, config; nstar = nstar)
    record!(results, candidate, fit.ic)
    return fit
end

# ─── Stepwise search ─────────────────────────────────────────────────────────

"""
    stepwise_search(x, m, d, D, start_p, start_q, start_P, start_Q,
                    constant, bounds, config) -> ArimaFit

Hyndman & Khandakar (2008) stepwise model selection. Hill-climbing over
neighbors in (p,q,P,Q,constant) space, accepting the first improvement
in each sweep.

Takes `SearchBounds` and `SearchConfig` structs — no kwargs forwarding.
"""
function stepwise_search(
    x::AbstractVector,
    m::Int,
    d::Int, D::Int,
    start_p::Int, start_q::Int, start_P::Int, start_Q::Int,
    constant::Bool,
    bounds::SearchBounds,
    config::SearchConfig,
)
    history = ModelHistory()
    results = SearchResults()

    # ── Phase 1: Initial models ──────────────────────────────────────────
    p, q, P, Q = start_p, start_q, start_P, start_Q

    bestfit = _try_candidate(x, m, ModelCandidate(p, d, q, P, D, Q, constant),
                             history, config, results)
    best_ic = isnothing(bestfit) ? Inf : bestfit.ic

    # Null model
    fit = _try_candidate(x, m, ModelCandidate(0, d, 0, 0, D, 0, constant),
                         history, config, results)
    if !isnothing(fit) && fit.ic < best_ic
        bestfit, best_ic = fit, fit.ic
        p, q, P, Q = 0, 0, 0, 0
    end

    # Pure AR(1)
    if bounds.max_p > 0 || bounds.max_P > 0
        ar_p = bounds.max_p > 0 ? 1 : 0
        ar_P = m > 1 && bounds.max_P > 0 ? 1 : 0
        fit = _try_candidate(x, m, ModelCandidate(ar_p, d, 0, ar_P, D, 0, constant),
                             history, config, results)
        if !isnothing(fit) && fit.ic < best_ic
            bestfit, best_ic = fit, fit.ic
            p, q, P, Q = ar_p, 0, ar_P, 0
        end
    end

    # Pure MA(1)
    if bounds.max_q > 0 || bounds.max_Q > 0
        ma_q = bounds.max_q > 0 ? 1 : 0
        ma_Q = m > 1 && bounds.max_Q > 0 ? 1 : 0
        fit = _try_candidate(x, m, ModelCandidate(0, d, ma_q, 0, D, ma_Q, constant),
                             history, config, results)
        if !isnothing(fit) && fit.ic < best_ic
            bestfit, best_ic = fit, fit.ic
            p, q, P, Q = 0, ma_q, 0, ma_Q
        end
    end

    # No-constant variant of null
    if constant
        fit = _try_candidate(x, m, ModelCandidate(0, d, 0, 0, D, 0, false),
                             history, config, results)
        if !isnothing(fit) && fit.ic < best_ic
            bestfit, best_ic = fit, fit.ic
            p, q, P, Q = 0, 0, 0, 0
        end
    end

    # ── Phase 2: Hill-climbing ───────────────────────────────────────────
    prev_count = 0
    while prev_count < n_evaluated(history) && n_evaluated(history) < config.nmodels
        prev_count = n_evaluated(history)
        current = ModelCandidate(p, d, q, P, D, Q, constant)
        improved = false

        for move in Iterators.flatten((_SEASONAL_MOVES, _NONSEASONAL_MOVES))
            neighbor = _apply_move(current, move, bounds)
            isnothing(neighbor) && continue

            fit = _try_candidate(x, m, neighbor, history, config, results)
            if !isnothing(fit) && fit.ic < best_ic
                bestfit, best_ic = fit, fit.ic
                p, q, P, Q = neighbor.p, neighbor.q, neighbor.P, neighbor.Q
                improved = true
                break  # restart sweep from new best
            end
        end
        improved && continue

        # Toggle constant term
        if config.allowdrift || config.allowmean
            toggled = ModelCandidate(p, d, q, P, D, Q, !constant)
            fit = _try_candidate(x, m, toggled, history, config, results)
            if !isnothing(fit) && fit.ic < best_ic
                bestfit, best_ic = fit, fit.ic
                constant = !constant
            end
        end
    end

    if n_evaluated(history) > config.nmodels
        @warn "Stepwise search was stopped early due to reaching the model number limit: $(config.nmodels)"
    end

    return (bestfit = bestfit, results = results)
end

# ─── Grid (exhaustive) search ────────────────────────────────────────────────

"""
    grid_search(x, m, d, D, bounds, config) -> ArimaFit

Exhaustive search over all (p,q,P,Q) combinations within bounds, optionally
with and without a constant term. If approximation was used, re-fits the best
model with ML.
"""
function grid_search(
    x::AbstractVector,
    m::Int,
    d::Int, D::Int,
    bounds::SearchBounds,
    config::SearchConfig,
)
    try_constant = config.allowdrift || config.allowmean

    best_ic = Inf
    bestfit = nothing
    best_constant = nothing

    @inbounds for i = 0:bounds.max_p, j = 0:bounds.max_q,
                  I = 0:bounds.max_P, J = 0:bounds.max_Q
        i + j + I + J <= bounds.max_order || continue

        for use_constant in (false, true)
            (!try_constant && use_constant) && continue

            candidate = ModelCandidate(i, d, j, I, D, J, use_constant)
            fit = evaluate_candidate(x, m, candidate, config)

            if fit isa ArimaFit && fit.ic < best_ic
                best_ic = fit.ic
                bestfit = fit
                best_constant = use_constant
            end
        end
    end

    if isnothing(bestfit)
        error("No ARIMA model able to be estimated")
    end

    # Re-fit without approximation
    if config.approximation
        if config.trace
            println("\n\n Now re-fitting the best model(s) without approximations...\n")
        end
        bestfit = _refit_grid_best(x, m, d, D, bestfit, best_constant, bounds, config)
    end

    return bestfit
end

"""Re-fit a grid search best model without CSS approximation."""
function _refit_grid_best(
    x::AbstractVector, m::Int, d::Int, D::Int,
    bestfit::ArimaFit, constant::Bool,
    bounds::SearchBounds, config::SearchConfig,
)
    arma = bestfit.arma
    candidate = ModelCandidate(arma[1], arma[6], arma[2], arma[3], arma[7], arma[4], constant)

    # Build a non-approximate config for the refit (preserve caller's method)
    refit_config = SearchConfig(
        config.ic, false, false, 0.0,
        config.xreg, config.method,
        config.allowdrift, config.allowmean,
        config.nmodels, config.arima_kwargs,
    )

    newfit = evaluate_candidate(x, m, candidate, refit_config)

    if newfit.ic == Inf
        # Fallback: full grid without approximation
        return grid_search(x, m, d, D, bounds, refit_config)
    end

    return newfit
end

# ─── Post-search refit (stepwise path) ───────────────────────────────────────

"""
    refit_stepwise_best(bestfit, results, x, m, d, D, serieslength, config) -> ArimaFit

After stepwise search with approximation, re-fit explored candidates in
IC-ranked order until one succeeds with ML estimation. This matches the
original behavior of iterating all explored models, not just the single best.
"""
function refit_stepwise_best(
    bestfit::ArimaFit,
    results::SearchResults,
    x::AbstractVector,
    m::Int,
    d::Int, D::Int,
    serieslength::Int,
    config::SearchConfig,
)
    if config.trace
        println("Now re-fitting the best model(s) without approximations...")
    end

    refit_config = SearchConfig(
        config.ic, config.trace, false, 0.0,
        config.xreg, config.method,
        config.allowdrift, config.allowmean,
        config.nmodels, config.arima_kwargs,
    )

    # Sort explored models by approximate IC (best first), skip missing/NaN
    valid_mask = [!(ismissing(v) || isnan(v)) for v in results.ic_values]
    ranked = sortperm(results.ic_values)

    for idx in ranked
        valid_mask[idx] || continue
        candidate = results.candidates[idx]

        fit = evaluate_candidate(x, m, candidate, refit_config; nstar = serieslength)
        if fit.ic < Inf
            return fit
        end
    end

    return bestfit
end
