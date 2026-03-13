# ─── Types for automatic ARIMA model selection ───────────────────────────────
#
# Mirrors the core pattern: types.jl defines SARIMAOrder, KalmanWorkspace,
# ObjectiveWorkspace — all data structures separate from logic.

"""
    ModelCandidate

Immutable description of a candidate ARIMA model's orders and constant term.
Analogous to `SARIMAOrder` in core — a value type that fully specifies what
to fit, with no mutable state.
"""
struct ModelCandidate
    p::Int
    d::Int
    q::Int
    P::Int
    D::Int
    Q::Int
    constant::Bool
end

"""
    SearchBounds

Upper limits for the order search space. Constructed once from user arguments
and passed through the search pipeline.
"""
struct SearchBounds
    max_p::Int
    max_q::Int
    max_P::Int
    max_Q::Int
    max_order::Int
end

"""
    SearchConfig

All parameters that are constant throughout a single auto_arima search.
Analogous to the `SARIMA` model struct in core — captures configuration
upfront so that downstream functions take `(data, config)` instead of
forwarding 15+ keyword arguments.

The `arima_kwargs` field holds any extra keyword arguments to forward to
the underlying `arima()` fitter (e.g. optimizer settings).
"""
struct SearchConfig
    ic::Symbol
    trace::Bool
    approximation::Bool
    offset::Float64
    xreg::Union{Nothing,NamedMatrix}
    method::Union{Nothing,Symbol}
    allowdrift::Bool
    allowmean::Bool
    nmodels::Int
    arima_kwargs::Base.Pairs
end

"""
    ModelHistory

Tracks which (p,d,q,P,D,Q,constant) tuples have been evaluated, using O(1)
hash-based lookup. Analogous to `KalmanWorkspace` — pre-allocated mutable
state for the inner loop.
"""
struct ModelHistory
    seen::Set{NTuple{7,Int}}
end

ModelHistory() = ModelHistory(Set{NTuple{7,Int}}())

function _candidate_key(c::ModelCandidate)
    (c.p, c.d, c.q, c.P, c.D, c.Q, Int(c.constant))
end

"""Return true if this candidate has not been tried before, and mark it as seen."""
function try_register!(history::ModelHistory, candidate::ModelCandidate)
    key = _candidate_key(candidate)
    key in history.seen && return false
    push!(history.seen, key)
    return true
end

n_evaluated(history::ModelHistory) = length(history.seen)

"""
    SearchResults

Collects all evaluated candidates and their IC values during search,
so that the post-search refit can iterate them in IC-ranked order.
"""
struct SearchResults
    candidates::Vector{ModelCandidate}
    ic_values::Vector{Float64}
end

SearchResults() = SearchResults(ModelCandidate[], Float64[])

function record!(results::SearchResults, candidate::ModelCandidate, ic_value::Float64)
    push!(results.candidates, candidate)
    push!(results.ic_values, ic_value)
end

# ─── Neighbor move tables ────────────────────────────────────────────────────
#
# Data-driven neighbor generation for the stepwise search. Each move is a
# (Δp, Δq, ΔP, ΔQ) tuple. The search tries seasonal moves first (matching
# Hyndman & Khandakar 2008), then non-seasonal, accepting the first improvement.

const _SEASONAL_MOVES = (
    ( 0,  0, -1,  0),   # decrease P
    ( 0,  0,  0, -1),   # decrease Q
    ( 0,  0, +1,  0),   # increase P
    ( 0,  0,  0, +1),   # increase Q
    ( 0,  0, -1, -1),   # decrease both
    ( 0,  0, -1, +1),   # trade P for Q
    ( 0,  0, +1, -1),   # trade Q for P
    ( 0,  0, +1, +1),   # increase both
)

const _NONSEASONAL_MOVES = (
    (-1,  0,  0,  0),   # decrease p
    ( 0, -1,  0,  0),   # decrease q
    (+1,  0,  0,  0),   # increase p
    ( 0, +1,  0,  0),   # increase q
    (-1, -1,  0,  0),   # decrease both
    (-1, +1,  0,  0),   # trade p for q
    (+1, -1,  0,  0),   # trade q for p
    (+1, +1,  0,  0),   # increase both
)
