module Arima
export arima, ArimaFit, PDQ, predict_arima, ArimaPredictions
export ArimaRJHFit, arima_rjh, auto_arima

using LinearAlgebra
import Statistics: mean
import LinearAlgebra: rank

using Polynomials
using Distributions
import Tables

using ..Stats
using ..Grammar
import ..Utils: is_constant, _check_arg, dropmissing, NamedMatrix, align_columns, ismissingish
import ..Stats: handle_missing
import ..Utils: is_constant_all, drop_constant_columns, is_rank_deficient, row_sums
import ..Utils: cbind, add_drift_term, setrow!, get_elements, select_rows, as_vector, as_integer
import ..Utils: mean2
import ..Stats: box_cox_lambda, box_cox, inv_box_cox, decompose, DecomposedTimeSeries, diff, fourier
import ..Generics: Forecast, forecast, plot, fitted, residuals
import ..Optimize: optimize, numerical_hessian
import ..Grammar: ModelFormula, ArimaOrderTerm, VarTerm, compile_arima_formula

import Base: show
include("types.jl")
include("covariance.jl")
include("hyperparameters.jl")
include("order.jl")
include("kalman.jl")
include("system.jl")
include("compat.jl")
include("fit.jl")
include("arima_rjh.jl")
include("auto_arima_utils.jl")
include("auto_arima.jl")
include("formula_interface.jl")
include("simulate.jl")
include("forecast.jl")
include("show.jl")

end
