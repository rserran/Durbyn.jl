# ─── ETS model selection ──────────────────────────────────────────────────────
#
# Grid-based search over ETS model specifications, analogous to auto/search.jl
# in the ARIMA module. Generates candidate models, fits each via etsmodel(),
# and selects the best by information criterion.

function validate_and_set_model_params(model, y, m, damped, restrict, additive_only)

    errortype = string(model[1])
    trendtype = string(model[2])
    seasontype = string(model[3])

    if !(errortype in ["M", "A", "Z"])
        throw(ArgumentError("Invalid error type"))
    end
    if !(trendtype in ["N", "A", "M", "Z"])
        throw(ArgumentError("Invalid trend type"))
    end
    if !(seasontype in ["N", "A", "M", "Z"])
        throw(ArgumentError("Invalid season type"))
    end

    if m < 1 || length(y) <= m
        seasontype = "N"
    end
    if m == 1
        if seasontype in ["A", "M"]
            throw(ArgumentError("Nonseasonal data"))
        else
            seasontype = "N"
        end
    end
    if m > 24
        if seasontype in ["A", "M"]
            throw(ArgumentError("Frequency too high"))
        elseif seasontype == "Z"
            @warn "I can't handle data with frequency greater than 24. Seasonality will be ignored. Try stlf() if you need seasonal forecasts."
            seasontype = "N"
        end
    end

    if restrict
        if (errortype == "A" && (trendtype == "M" || seasontype == "M")) ||
           (errortype == "M" && trendtype == "M" && seasontype == "A") ||
           (additive_only && (errortype == "M" || trendtype == "M" || seasontype == "M"))
            throw(ArgumentError("Forbidden model combination"))
        end
    end

    data_positive = minimum(y) > 0
    if !data_positive
        if errortype == "M"
            throw(ArgumentError(
                "Multiplicative error models require strictly positive data. " *
                "Data contains zero or negative values."
            ))
        end
        if trendtype == "M"
            throw(ArgumentError(
                "Multiplicative trend models require strictly positive data. " *
                "Data contains zero or negative values."
            ))
        end
        if seasontype == "M"
            throw(ArgumentError(
                "Multiplicative seasonal models require strictly positive data. " *
                "Data contains zero or negative values."
            ))
        end
    end

    if !isnothing(damped)
        if damped && trendtype == "N"
            throw(
                ArgumentError(
                    "Forbidden model combination: Damped trend with no trend component",
                ),
            )
        end
    end

    n = length(y)
    npars = 2

    if trendtype in ["A", "M"]
        npars += 2
    end
    if seasontype in ["A", "M"]
        npars += m
    end
    if !isnothing(damped)
        npars += damped ? 1 : 0
    end

    return errortype, trendtype, seasontype, npars, data_positive
end

function get_ic(fit, ic)
    if ic === :aic
        return fit["aic"]
    elseif ic === :bic
        return fit["bic"]
    elseif ic === :aicc
        return fit["aicc"]
    else
        return Inf
    end
end

function generate_ets_grid_fixed(
    errortype::String,
    trendtype::String,
    seasontype::String,
    allow_multiplicative_trend::Bool,
    restrict::Bool,
    additive_only::Bool,
    data_positive::Bool,
    damped::Union{Bool,Nothing},
)
    errortype_vals = (errortype == "Z") ? ["A", "M"] : [errortype]
    trendtype_vals = if trendtype == "Z"
        allow_multiplicative_trend ? ["N", "A", "M"] : ["N", "A"]
    else
        [trendtype]
    end
    seasontype_vals = (seasontype == "Z") ? ["N", "A", "M"] : [seasontype]
    damped_vals = isnothing(damped) ? [true, false] : [damped]

    grid = []

    for e in errortype_vals
        for t in trendtype_vals
            for s in seasontype_vals
                for d in damped_vals
                    if t == "N" && d
                        continue
                    end
                    if restrict
                        if e == "A" && (t == "M" || s == "M")
                            continue
                        end
                        if e == "M" && t == "M" && s == "A"
                            continue
                        end
                        if additive_only && (e == "M" || t == "M" || s == "M")
                            continue
                        end
                    end
                    if !data_positive && e == "M"
                        continue
                    end
                    push!(grid, (e, t, s, d))
                end
            end
        end
    end

    return grid
end

function fit_ets_models(
    grid,
    y,
    m,
    alpha,
    beta,
    gamma,
    phi,
    lower,
    upper,
    opt_crit,
    nmse,
    bounds,
    ic,
    options)

    best_ic = Inf
    best_model = nothing
    best_params = ()
    lower_scratch = similar(lower)
    upper_scratch = similar(upper)

    for combo in grid
        et, t, s, d = combo
        try
            copyto!(lower_scratch, lower)
            copyto!(upper_scratch, upper)
            the_fit_model = etsmodel(
                y,
                m,
                et,
                t,
                s,
                d,
                alpha,
                beta,
                gamma,
                phi,
                lower_scratch,
                upper_scratch,
                opt_crit,
                nmse,
                bounds,
                options)
            fit_ic = get_ic(the_fit_model, ic)
            if fit_ic < best_ic
                best_ic = fit_ic
                best_model = the_fit_model
                best_params = combo
            end
        catch e
            @warn "Error fitting model with combination: $combo" exception=(e, catch_backtrace())
            continue
        end
    end
    return Dict(
        "best_model" => best_model,
        "best_params" => best_params,
        "best_ic" => best_ic,
    )
end

function fit_best_ets_model(
    y,
    m,
    errortype,
    trendtype,
    seasontype,
    damped,
    alpha,
    beta,
    gamma,
    phi,
    lower,
    upper,
    opt_crit,
    nmse,
    bounds,
    ic,
    data_positive;
    restrict = true,
    additive_only = false,
    allow_multiplicative_trend = true,
    options)

    grid = generate_ets_grid_fixed(
        errortype,
        trendtype,
        seasontype,
        allow_multiplicative_trend,
        restrict,
        additive_only,
        data_positive,
        damped,
    )

    result = fit_ets_models(
        grid,
        y,
        m,
        alpha,
        beta,
        gamma,
        phi,
        lower,
        upper,
        opt_crit,
        nmse,
        bounds,
        ic,
        options)

    best_model = result["best_model"]
    best_params = result["best_params"]
    best_ic = result["best_ic"]
    result = nothing

    best_e, best_t, best_s, best_d = best_params

    if best_ic == Inf
        throw(ModelFitError("No model able to be fitted"))
    end

    method = "ETS($(best_e),$(best_t)$(best_d ? "d" : ""),$(best_s))"
    components = [best_e, best_t, best_s, string(best_d)]
    return Dict("model" => best_model, "method" => method, "components" => components)
end

function fit_small_dataset(
    y,
    m,
    alpha,
    beta,
    gamma,
    phi,
    trendtype,
    seasontype,
    lambda,
    biasadj,
    options
)

    if seasontype in ["A", "M"]
        try
            fit = holt_winters_conventional(
                y,
                m,
                alpha = alpha,
                beta = beta,
                gamma = gamma,
                seasonal = (seasontype == "M" ? :multiplicative : :additive),
                exponential = (trendtype == "M"),
                phi = phi,
                lambda = lambda,
                biasadj = biasadj,
                warnings = false,
                options = options
            )
            return fit
        catch e
            @warn "Seasonal component could not be estimated: $e"
        end
    end

    if trendtype in ["A", "M"]
        try
            fit = holt_winters_conventional(
                y,
                m,
                alpha = alpha,
                beta = beta,
                gamma = false,
                seasonal = :additive,
                exponential = (trendtype == "M"),
                phi = phi,
                lambda = lambda,
                biasadj = biasadj,
                warnings = false,
                options = options
            )
            return fit
        catch e
            @warn "Trend component could not be estimated: $e"
            return nothing
        end
    end

    if trendtype == "N" && seasontype == "N"
        try
            fit = holt_winters_conventional(
                y,
                m,
                alpha = alpha,
                beta = false,
                gamma = false,
                seasonal = :additive,
                exponential = false,
                phi = nothing,
                lambda = lambda,
                biasadj = biasadj,
                warnings = false,
                options = options
            )
            return fit
        catch e
            @warn "Model without trend and seasonality could not be estimated: $e"
            return nothing
        end
    end

    fit1 = try
        holt_winters_conventional(
            y,
            m,
            alpha = alpha,
            beta = beta,
            gamma = false,
            seasonal = :additive,
            exponential = (trendtype == "M"),
            phi = phi,
            lambda = lambda,
            biasadj = biasadj,
            warnings = false,
            options = options
        )
    catch e
        nothing
    end

    fit2 = try
        holt_winters_conventional(
            y,
            m,
            alpha = alpha,
            beta = false,
            gamma = false,
            seasonal = :additive,
            exponential = (trendtype == "M"),
            phi = phi,
            lambda = lambda,
            biasadj = biasadj,
            warnings = false,
            options = options
        )
    catch e
        nothing
    end

    fit =
        isnothing(fit1) ? fit2 :
        (isnothing(fit2) ? fit1 : (fit1.sigma2 < fit2.sigma2 ? fit1 : fit2))

    if isnothing(fit)
        error("Unable to estimate a model.")
    end

    return fit
end
