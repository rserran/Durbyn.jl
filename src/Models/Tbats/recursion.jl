# ─── TBATS state space recursion ──────────────────────────────────────────────
#
# Core computational kernel for the TBATS state space model.

function calc_model_tbats(
    y::Vector{Float64},
    x_nought::Vector{Float64},
    F::Matrix{Float64},
    g::Vector{Float64},
    w::NamedTuple,
)
    n = length(y)
    dim = length(x_nought)

    x = zeros(dim, n)
    y_hat = zeros(n)
    e = zeros(n)

    y_hat[1] = (w.w_transpose*x_nought)[1]
    e[1] = y[1] - y_hat[1]
    x[:, 1] = F * x_nought + g * e[1]

    for t = 2:n
        y_hat[t] = (w.w_transpose*x[:, t-1])[1]
        e[t] = y[t] - y_hat[t]
        x[:, t] = F * x[:, t-1] + g * e[t]
    end

    return (y_hat = y_hat, e = e, x = x)
end
