# ─── Single ETS model evaluation ──────────────────────────────────────────────
#
# Functions for fitting a single ETS model specification: objective function,
# optimizer wrapper, and the etsmodel() orchestrator that ties them together.
# Analogous to auto/evaluation.jl in the ARIMA module: standalone functions
# with explicit typed arguments for fitting one candidate model.

function create_params(
    optimized_params::AbstractArray,
    opt_alpha::Bool,
    opt_beta::Bool,
    opt_gamma::Bool,
    opt_phi::Bool,
)
    j = 1
    pars = Dict()

    if opt_alpha
        pars["alpha"] = optimized_params[j]
        j += 1
    end

    if opt_beta
        pars["beta"] = optimized_params[j]
        j += 1
    end

    if opt_gamma
        pars["gamma"] = optimized_params[j]
        j += 1
    end

    if opt_phi
        pars["phi"] = optimized_params[j]
        j += 1
    end

    result = optimized_params[j:end]
    return merge(pars, Dict("initstate" => result))
end


function objective_fun(
    par,
    y,
    nstate,
    errortype::Int,
    trendtype::Int,
    seasontype::Int,
    damped,
    lower,
    upper,
    opt_crit::Int,
    nmse,
    bounds::Int,
    m,
    init_alpha,
    init_beta,
    init_gamma,
    init_phi,
    opt_alpha,
    opt_beta,
    opt_gamma,
    opt_phi,
    workspace::ETSWorkspace,
)

    j = 1

    alpha = opt_alpha ? par[j] : init_alpha
    j += opt_alpha

    beta = nothing
    if trendtype != 0
        beta = opt_beta ? par[j] : init_beta
        j += opt_beta
    end

    gamma = nothing
    if seasontype != 0
        gamma = opt_gamma ? par[j] : init_gamma
        j += opt_gamma
    end

    phi = nothing
    if damped
        phi = opt_phi ? par[j] : init_phi
        j += opt_phi
    end

    if isnan(alpha)
        throw(ArgumentError("alpha must be numeric"))
    elseif !isnothing(beta) && isnan(beta)
        throw(ArgumentError("beta must be numeric"))
    elseif !isnothing(gamma) && isnan(gamma)
        throw(ArgumentError("gamma must be numeric"))
    elseif !isnothing(phi) && isnan(phi)
        throw(ArgumentError("phi must be numeric"))
    end

    if !check_param(alpha, beta, gamma, phi, lower, upper, bounds, m)
        return Inf
    end

    init_state = view(par, (length(par)-nstate+1):length(par))
    init_state_eval = init_state
    trend_slots = trendtype == 0 ? 0 : 1

    if seasontype != 0
        workspace_init = workspace.init_state
        @inbounds copyto!(workspace_init, 1, init_state, 1, nstate)
        seasonal_start = 2 + trend_slots
        seasonal_sum = sum(view(workspace_init, seasonal_start:nstate))
        workspace_init[nstate+1] = (seasontype == 2 ? m : 0.0) - seasonal_sum
        init_state_eval = view(workspace_init, 1:(nstate+1))
    end

    if seasontype == 2
        if minimum(view(init_state_eval, (2+trend_slots):length(init_state_eval))) < 0.0
            return Inf
        end
    end

    lik, _ = calculate_residuals!(
        workspace,
        y,
        m,
        init_state_eval,
        errortype,
        trendtype,
        seasontype,
        damped,
        alpha,
        beta,
        gamma,
        phi,
        nmse,
    )

    if isnan(lik) || abs(lik + 99999) < 1e-7
        lik = Inf
    elseif lik < -1e10
        # Clamp near-perfect fits to a finite floor.
        lik = -1e10
    end

    if opt_crit == OPT_CRIT_LIK
        return lik
    elseif opt_crit == OPT_CRIT_MSE
        return workspace.amse[1]
    elseif opt_crit == OPT_CRIT_AMSE
        return sum(view(workspace.amse, 1:nmse)) / nmse
    elseif opt_crit == OPT_CRIT_SIGMA
        return sum(abs2, view(workspace.e, 1:length(y))) / length(y)
    elseif opt_crit == OPT_CRIT_MAE
        return sum(abs, view(workspace.e, 1:length(y))) / length(y)
    else
        throw(ArgumentError("Unknown optimization criterion"))
    end
end

function optim_ets_base(
    opt_params,
    y,
    nstate,
    errortype,
    trendtype,
    seasontype,
    damped,
    lower,
    upper,
    opt_crit,
    nmse,
    bounds,
    m,
    initial_params,
    options)

    init_alpha = initial_params.alpha
    init_beta = initial_params.beta
    init_gamma = initial_params.gamma
    init_phi = initial_params.phi
    opt_alpha = !isnan(init_alpha)
    opt_beta = !isnan(init_beta)
    opt_gamma = !isnan(init_gamma)
    opt_phi = !isnan(init_phi)
    errortype_code = ets_model_type_code(errortype)
    trendtype_code = ets_model_type_code(trendtype)
    seasontype_code = ets_model_type_code(seasontype)
    opt_crit_int = opt_crit_code(opt_crit)
    bounds_int = bounds_code(bounds)
    max_state_len = nstate + (seasontype_code != 0 ? 1 : 0)
    workspace = ETSWorkspace(length(y), m, nmse, max_state_len)

    result = nelder_mead(par -> objective_fun(
            par,
            y,
            nstate,
            errortype_code,
            trendtype_code,
            seasontype_code,
            damped,
            lower,
            upper,
            opt_crit_int,
            nmse,
            bounds_int,
            m,
            init_alpha,
            init_beta,
            init_gamma,
            init_phi,
            opt_alpha,
            opt_beta,
            opt_gamma,
            opt_phi,
            workspace,
        ), opt_params,
        options)

    optimized_params = result.x_opt
    optimized_value = result.f_opt
    number_of_iterations = result.fncount

    optimized_params =
        create_params(optimized_params, opt_alpha, opt_beta, opt_gamma, opt_phi)

    return Dict(
        "optimized_params" => optimized_params,
        "optimized_value" => optimized_value,
        "number_of_iterations" => number_of_iterations,
    )
end

function etsmodel(
    y::Vector{Float64},
    m::Int,
    errortype::String,
    trendtype::String,
    seasontype::String,
    damped::Bool,
    alpha::Union{Float64,Nothing,Bool},
    beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool},
    phi::Union{Float64,Nothing,Bool},
    lower::Vector{Float64},
    upper::Vector{Float64},
    opt_crit::Symbol,
    nmse::Int,
    bounds::Symbol,
    options::NelderMeadOptions)

    if seasontype == "N"
        m = 1
    end

    if isa(alpha, Bool)
        if alpha
            alpha = 1.0 - 1e-10
        else
            alpha = 0.0
        end
    end

    if isa(beta, Bool)
        if beta
            beta = 1.0
        else
            beta = 0.0
        end
    end

    if isa(gamma, Bool)
        if gamma
            gamma = 1.0
        else
            gamma = 0.0
        end
    end

    if isa(phi, Bool)
        if phi
            phi = 1.0
        else
            phi = 0.0
        end
    end

    if !(isnothing(alpha) || (!isnothing(alpha) && isnan(alpha)))
        upper[2] = min(alpha, upper[2])
        upper[3] = min(1 - alpha, upper[3])
    end


    if !(isnothing(beta) || (!isnothing(beta) && isnan(beta)))
        lower[1] = max(beta, lower[1])
    end

    if !(isnothing(gamma) || (!isnothing(gamma) && isnan(gamma)))
        upper[1] = min(1 - gamma, upper[1])
    end

    par = initparam(
        alpha,
        beta,
        gamma,
        phi,
        trendtype,
        seasontype,
        damped,
        lower,
        upper,
        m,
        bounds,
        nothing_as_nan = true,
    )

    if !isnan(par.alpha)
        alpha = par.alpha
    end

    if !isnan(par.beta)
        beta = par.beta
    end

    if !isnan(par.gamma)
        gamma = par.gamma
    end

    if !isnan(par.phi)
        phi = par.phi
    end

    if !check_param(alpha, beta, gamma, phi, lower, upper, bounds, m)
        damped_str = damped ? "d" : ""
        throw(ArgumentError("For model `ETS($(errortype),\
         $(trendtype)$(damped_str), $(seasontype))` \
         parameters are out of range!"))
    end

    init_state = initialize_states(y, m, trendtype, seasontype)
    nstate = length(init_state)
    initial_params = par
    par = [par.alpha, par.beta, par.gamma, par.phi]
    par = dropmissing(par)
    par = vcat(par, init_state)

    lower = vcat(lower, fill(-Inf, nstate))
    upper = vcat(upper, fill(Inf, nstate))

    np = length(par)
    if np >= length(y) - 1
        return Dict(
            :aic => Inf,
            :bic => Inf,
            :aicc => Inf,
            :mse => Inf,
            :amse => Inf,
            :fit => nothing,
            :par => par,
            :states => init_state,
        )
    end

    init_state = nothing

    optimized_fit = optim_ets_base(
        par,
        y,
        nstate,
        errortype,
        trendtype,
        seasontype,
        damped,
        lower,
        upper,
        opt_crit,
        nmse,
        bounds,
        m,
        initial_params,
        options)

    fit_par = optimized_fit["optimized_params"]

    states = fit_par["initstate"]

    if seasontype != "N"
        states =
            vcat(states, m * (seasontype == "M") - sum(states[(2+(trendtype!="N")):nstate]))
    end

    if !isnan(initial_params.alpha)
        alpha = fit_par["alpha"]
    end
    if !isnan(initial_params.beta)
        beta = fit_par["beta"]
    end
    if !isnan(initial_params.gamma)
        gamma = fit_par["gamma"]
    end
    if !isnan(initial_params.phi)
        phi = fit_par["phi"]
    end

    lik, amse, e, states = calculate_residuals(
        y,
        m,
        states,
        errortype,
        trendtype,
        seasontype,
        damped,
        alpha,
        beta,
        gamma,
        phi,
        nmse,
    )

    np += 1
    ny = length(y)
    aic = lik + 2 * np
    bic = lik + log(ny) * np
    aicc = aic + 2 * np * (np + 1) / (ny - np - 1)
    mse = amse[1]
    amse = mean(amse)

    if errortype == "A"
        fits = y .- e
    else
        fits = y ./ (1 .+ e)
    end

    out = Dict(
        "loglik" => -0.5 * lik,
        "aic" => aic,
        "bic" => bic,
        "aicc" => aicc,
        "mse" => mse,
        "amse" => amse,
        "fit" => optimized_fit,
        "residuals" => e,
        "fitted" => fits,
        "states" => states,
        "par" => fit_par,
    )
    return out
end
