module IntermittentDemand
using Statistics
import Base: show
import Statistics: mean

using Optim
using ..Generics
import ..Utils: evaluation_metrics
import ..Utils: _check_arg
import ..Generics: plot
import ..Generics: forecast, fitted, residuals

include("crost_utils.jl")
include("crost.jl")

export croston_classic, croston_sba, croston_sbj
export IntermittentDemandForecast
export IntermittentDemandCrostonFit

end