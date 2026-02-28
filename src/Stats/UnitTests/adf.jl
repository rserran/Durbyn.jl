
"""
    struct ADF
        model::Symbol
        cval::Matrix{Float64}
        clevels::Vector{Float64}
        lag::Int
        teststat::NamedMatrix{Float64}
        testreg::NamedTuple
        res::Vector{Float64}
        testnames::Vector{Symbol}
    end

Augmented Dickey-Fuller (ADF) unit-root test result.

# Fields
- `model::Symbol`: Deterministic component used in the test regression.
  One of `:none` (no constant, no trend), `:drift` (intercept only),
  or `:trend` (intercept and linear trend).
- `cval::Matrix{Float64}`: Matrix of critical values. Rows typically
  map to `testnames`, columns to the significance levels in `clevels`.
- `clevels::Vector{Float64}`: Significance levels (e.g. `[0.01, 0.05, 0.10]`)
  corresponding to the columns of `cval`.
- `lag::Int`: Number of lagged differences included (augmentation order).
- `teststat::NamedMatrix{Float64}`: A named matrix of test statistics
  (e.g., τ-statistics) with row/column names.
- `testreg::NamedTuple`: Auxiliary regression output (e.g., coefficient
  table, residual variance). The exact content is implementation-defined.
- `res::Vector{Float64}`: Regression residuals.
- `testnames::Vector{Symbol}`: Names for the statistics reported in
  `teststat` and for the rows of `cval` (when aligned).

# See also
[`adf`](@ref), pretty-printing via `show(::IO, ::ADF)` and `summary(::ADF)`.
"""
struct ADF
    model::Symbol
    cval::Matrix{Float64}
    clevels::Vector{Float64}
    lag::Int
    teststat::NamedMatrix{Float64}
    testreg::NamedTuple
    res::Vector{Float64}
    testnames::Vector{Symbol}
end

"""
    adf(y; type::Symbol = :none, lags::Int = 1, selectlags::Symbol = :fixed) -> ADF

Perform the Augmented Dickey-Fuller (ADF) unit-root test.

The ADF test augments the basic Dickey-Fuller regression with lagged
differences of the series to correct for serial correlation. The test
can be run without deterministic terms, with an intercept ("drift"),
or with an intercept and linear trend.

# Arguments
- `y`: Univariate series (e.g. `AbstractVector{<:Real}`) to be tested
  for a unit root.

# Keyword Arguments
- `type::Symbol = :none`: Deterministic component in the test regression.
  One of `:none` (no constant), `:drift` (intercept only),
  `:trend` (intercept and linear trend).
- `lags::Int = 1`: Maximum number of lagged differences to include in
  the augmentation. If `selectlags == :fixed`, this exact number is used.
- `selectlags::Symbol = :fixed`: Lag-selection rule. One of:
  - `:fixed`: use the provided `lags` as-is;
  - `:aic`: select the lag length ≤ `lags` minimizing aic;
  - `:bic`: select the lag length ≤ `lags` minimizing bic.

# Details
If `type == :none`, neither an intercept nor a trend is included.
If `type == :drift`, an intercept is included. If `type == :trend`,
both an intercept and a linear trend are included. When
`selectlags` is `:aic` or `:bic`, the test evaluates specifications
with augmentation orders `0:lags` and selects the information-criterion
minimizer.

Critical values for the ADF test statistics follow the tabulations of
Hamilton (1994) and Dickey & Fuller (1981). Reported statistics and
their names are provided via the `NamedMatrix` `teststat` and
`testnames`. Additional regression information (coefficients, standard
errors, etc.) may be exposed in `testreg`.

# Returns
An [`ADF`](@ref) object containing:
- the chosen `model` (`:none`, `:drift`, `:trend`);
- the selected `lag` (augmentation order);
- `teststat` (named matrix of ADF statistics);
- `cval` and `clevels` (critical values and significance levels);
- `res` (residuals) and `testreg` (auxiliary regression results);
- `testnames` (labels for the reported statistics).

# References
- Dickey, D. A. and Fuller, W. A. (1979).
  *Distribution of the Estimators for Autoregressive Time Series with a Unit Root*,
  **Journal of the American Statistical Association**, 75, 427-431.
- Dickey, D. A. and Fuller, W. A. (1981).
  *Likelihood Ratio Statistics for Autoregressive Time Series with a Unit Root*,
  **Econometrica**, 49, 1057-1072.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.

# Examples
```julia
julia> using Durbyn
julia> using Durbyn.Stats
julia> ap = air_passengers();

julia>  adf(ap)
julia> adf(y = ap)
```
"""
function adf(y; type::Symbol=:none, lags::Int=1, selectlags::Symbol=:fixed)

    type ∈ (:none, :drift, :trend) || throw(ArgumentError("type must be :none, :drift, or :trend"))
    selectlags ∈ (:fixed, :aic, :bic) || throw(ArgumentError("selectlags must be :fixed, :aic, or :bic"))

    yv = _skipmissing_to_vec(y)
    (n_data,) = size(yv)
    n_data > 1 || throw(ArgumentError("y is not a vector or too short"))
    any(isnan.(yv)) && throw(ArgumentError("NAs in y"))

    lag = Int(lags)
    lag >= 0 || throw(ArgumentError("lags must be a nonnegative integer"))

    augmentation_order = lag + 1
    z = diff(yv)
    z = _skipmissing_to_vec(z)
    n = length(z)
    n >= augmentation_order || throw(ArgumentError("Not enough observations for the requested lags"))

    x = time_delay_embed(z, augmentation_order)
    n_obs = size(x, 1)
    delta_y  = x[:, 1]
    y_lagged = yv[augmentation_order:n]
    time_index = collect(augmentation_order:(augmentation_order + n_obs - 1))

    function _run_adf(chosen_L::Int)
        if chosen_L > 1
            z_diff_lag = x[:, 2:chosen_L]
            n_lag_cols = size(z_diff_lag, 2)
        else
            z_diff_lag = Array{Float64}(undef, n_obs, 0)
            n_lag_cols = 0
        end

        if type == :none
            X = hcat(y_lagged, z_diff_lag)
            fit = _ols(delta_y, X)
            β, se, res = fit.β, fit.se, fit.residuals
            tau = β[1] / se[1]

            TS = NamedMatrix(1, ["tau1"]; T=Float64, rownames=["statistic"])
            TS.data[1, 1] = tau
            return (fit=fit, res=res, teststat=TS, p=length(β), X=X)

        elseif type == :drift
            X = hcat(ones(Float64, n_obs), y_lagged, z_diff_lag)
            fit = _ols(delta_y, X)
            β, se, res = fit.β, fit.se, fit.residuals
            tau = β[2] / se[2]

            Xr = z_diff_lag
            if size(Xr,2) == 0
                Xr = ones(Float64, n_obs, 0)
            end
            fitR = _ols(delta_y, Xr)
            RSSr, dfr = sum(fitR.residuals.^2), fitR.df_residual
            RSSf, dff = sum(res.^2), fit.df_residual
            phi1 = _f_test_restricted_vs_full(RSSr, dfr, RSSf, dff)

            TS = NamedMatrix(1, ["tau2","phi1"]; T=Float64, rownames=["statistic"])
            TS.data[1, :] .= (tau, phi1)

            return (fit=fit, res=res, teststat=TS, p=length(β), X=X)

        else
            X = hcat(ones(Float64, n_obs), y_lagged, Float64.(time_index), z_diff_lag)
            fit = _ols(delta_y, X)
            β, se, res = fit.β, fit.se, fit.residuals
            tau = β[2] / se[2]

            Xr2 = z_diff_lag
            if size(Xr2,2) == 0
                Xr2 = ones(Float64, n_obs, 0)
            end
            fitR2 = _ols(delta_y, Xr2)
            phi2 = _f_test_restricted_vs_full(sum(fitR2.residuals.^2), fitR2.df_residual,
                                  sum(res.^2), fit.df_residual)

            Xr3 = hcat(ones(Float64, n_obs), z_diff_lag)
            fitR3 = _ols(delta_y, Xr3)
            phi3 = _f_test_restricted_vs_full(sum(fitR3.residuals.^2), fitR3.df_residual,
                                  sum(res.^2), fit.df_residual)

            TS = NamedMatrix(1, ["tau3","phi2","phi3"]; T=Float64, rownames=["statistic"])
            TS.data[1, :] .= (tau, phi2, phi3)

            return (fit=fit, res=res, teststat=TS, p=length(β), X=X)
        end
    end

    chosen_L = augmentation_order
    if augmentation_order > 1 && selectlags != :fixed
        critRes = fill(Inf, augmentation_order)
        kpen = selectlags == :aic ? 2.0 : log(length(delta_y))
        for i in 2:augmentation_order
            out = _run_adf(i)
            RSS = sum(out.res .^ 2)
            nobs = length(out.res)
            p = out.p
            critRes[i] = _information_criterion(RSS, nobs, p, kpen)
        end
        imin = argmin(critRes)
        if imin ≥ 2 && isfinite(critRes[imin])
            chosen_L = imin
        end
    end

    out = _run_adf(chosen_L)

    cv_row = n_obs < 25 ? 1 :
             n_obs < 50 ? 2 :
             n_obs < 100 ? 3 :
             n_obs < 250 ? 4 :
             n_obs < 500 ? 5 : 6

    function _cvals(type::Symbol, row::Int)
        if type == :none
            cval_tau1 = [
                -2.66 -1.95 -1.60;
                -2.62 -1.95 -1.61;
                -2.60 -1.95 -1.61;
                -2.58 -1.95 -1.62;
                -2.58 -1.95 -1.62;
                -2.58 -1.95 -1.62
            ]
            return reshape(cval_tau1[row, :], 1, 3), [:tau1]
        elseif type == :drift
            cval_tau2 = [
                -3.75 -3.00 -2.63;
                -3.58 -2.93 -2.60;
                -3.51 -2.89 -2.58;
                -3.46 -2.88 -2.57;
                -3.44 -2.87 -2.57;
                -3.43 -2.86 -2.57
            ]
            cval_phi1 = [
                7.88 5.18 4.12;
                7.06 4.86 3.94;
                6.70 4.71 3.86;
                6.52 4.63 3.81;
                6.47 4.61 3.79;
                6.43 4.59 3.78
            ]
            C = vcat(reshape(cval_tau2[row, :], 1, 3),
                     reshape(cval_phi1[row, :], 1, 3))
            return C, [:tau2, :phi1]
        else
            cval_tau3 = [
                -4.38 -3.60 -3.24;
                -4.15 -3.50 -3.18;
                -4.04 -3.45 -3.15;
                -3.99 -3.43 -3.13;
                -3.98 -3.42 -3.13;
                -3.96 -3.41 -3.12
            ]
            cval_phi2 = [
                8.21 5.68 4.67;
                7.02 5.13 4.31;
                6.50 4.88 4.16;
                6.22 4.75 4.07;
                6.15 4.71 4.05;
                6.09 4.68 4.03
            ]
            cval_phi3 = [
                10.61 7.24 5.91;
                 9.31 6.73 5.61;
                 8.73 6.49 5.47;
                 8.43 6.49 5.47;
                 8.34 6.30 5.36;
                 8.27 6.25 5.34
            ]
            C = vcat(reshape(cval_tau3[row, :], 1, 3),
                     reshape(cval_phi2[row, :], 1, 3),
                     reshape(cval_phi3[row, :], 1, 3))
            return C, [:tau3, :phi2, :phi3]
        end
    end

    cvals, names = _cvals(type, cv_row)
    clevels = [0.01, 0.05, 0.10]

    return ADF(type, cvals, clevels, chosen_L - 1, out.teststat, out.fit, out.res, names)
end


function summary(x::ADF)
    vals = join(string.(round.(collect(x.teststat); digits=4)), ", ")
    "ADF(model=$(x.model), lag=$(x.lag), teststat=[$(vals)])"
end


function show(io::IO, ::MIME"text/plain", x::ADF)
    println(io, "ADF (Augmented Dickey–Fuller) Unit Root Test")

    desc = x.model === :none ? "none (no constant)" :
           x.model === :drift ? "drift (intercept only)" :
           x.model === :trend ? "trend (intercept + linear trend)" :
           string(x.model)
    println(io, "Deterministic component (model): ", desc)
    println(io, "Lag truncation (bandwidth): ", x.lag)

    println(io, "\nTest statistic(s):")
    show(io, MIME("text/plain"), x.teststat)
    println(io)

    if !isempty(x.cval) && !isempty(x.clevels) && size(x.cval, 2) == length(x.clevels)
        println(io, "\nCritical values:")
        levels_str = ["$(Int(round(100α)))%" for α in x.clevels]

        rowlabels = (length(x.testnames) == size(x.cval, 1) && !isempty(x.testnames)) ?
                    string.(x.testnames) :
                    ["test$(i)" for i in 1:size(x.cval, 1)]

        colw = maximum(length, ["level"; levels_str])
        valw = maximum(length, string.(round.(vec(x.cval); digits=4)))

        print(io, "  ", lpad("level", colw))
        for L in levels_str
            print(io, "  ", lpad(L, max(valw, length(L))))
        end
        println(io)

        for (i, rlbl) in enumerate(rowlabels)
            print(io, "  ", lpad(rlbl, colw))
            for j in 1:length(levels_str)
                print(io, "  ", lpad(string(round(x.cval[i, j]; digits=4)), max(valw, length(levels_str[j]))))
            end
            println(io)
        end
    end

    if hasproperty(x, :testreg) && !isnothing(x.testreg)

        println(io, "\nRegression fields available in `x.testreg` (not shown).")
    end
end

function show(io::IO, x::ADF)
    if get(io, :compact, false)

        v = try
            x.teststat[1]
        catch
            NaN
        end
        print(io, "ADF($(round(v; digits=4)))")
    else
        show(io, MIME("text/plain"), x)
    end
end
