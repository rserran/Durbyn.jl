# ─── ETS state initialization ─────────────────────────────────────────────────
#
# Functions for computing initial level, trend, and seasonal states from the
# observed series. Separated from the core recursion kernels to mirror the
# ARIMA core pattern (covariance.jl / hyperparameters.jl for initial state setup).

function handle_seasonality(y::AbstractVector, m::Integer, seasontype::String)
    n = length(y)
    n < 4 && throw(ArgumentError("y is too short!"))

    if n < 3 * m
        F = fourier(y; m = m, K = 1)
        t = collect(1:n)
        X = hcat(ones(n), t, reduce(hcat, values(F)))

        β = X \ y
        α, φ = β[1], β[2]

        seasonal = if seasontype == "A"
            y .- (α .+ φ .* t)
        else
            y ./ (α .+ φ .* t)
        end

        return Dict(:seasonal => seasonal)
    else
        decomp = decompose(
            x = y,
            m = m,
            type = seasontype == "A" ? :additive : :multiplicative,
        )
        return Dict(:seasonal => decomp.seasonal)
    end
end

function initialize_seasonal_components(y_d, m, seasontype)
    seasonal = y_d[:seasonal]
    init_seas = reverse(seasonal[2:m])
    if seasontype != "A"
        init_seas = [max(val, 0.01) for val in init_seas]
        if sum(init_seas) > m
            factor = sum(init_seas .+ 0.01)
            init_seas .= init_seas ./ factor
        end
    end

    return init_seas
end

function adjust_y_sa(y, y_d, seasontype)
    seasonal = y_d[:seasonal]
    if seasontype == "A"
        return y .- seasonal
    else
        return y ./ max.(seasonal, 0.01)
    end
end

function lsfit_ets(x::Matrix{Float64}, y::Vector{Float64})
    n, _ = size(x)
    X = hcat(ones(n), x)
    return X \ y
end

function calculate_initial_values(y_sa::Vector{Float64}, trendtype::String, maxn::Int)

    if trendtype == "N"
        l0 = mean(y_sa[1:maxn])
        b0 = nothing
    else

        x = reshape(collect(1.0:maxn), maxn, 1)
        β = lsfit_ets(x, y_sa[1:maxn])


        if trendtype == "A"
            l0, b0 = β[1], β[2]

            if abs(l0 + b0) < 1e-8
                l0 *= (1 + 1e-3)
                b0 *= (1 - 1e-3)
            end

        else
            l0 = β[1] + β[2]
            if abs(l0) < 1e-8
                l0 = 1e-7
            end

            b0 = (β[1] + 2 * β[2]) / l0
            l0 = l0 / b0

            if abs(b0) > 1e10
                b0 = sign(b0) * 1e10
            end

            if l0 < 1e-8 || b0 < 1e-8
                l0 = max(y_sa[1], 1e-3)
                denom = isapprox(y_sa[1], 0.0; atol = 1e-8) ? y_sa[1] + 1e-10 : y_sa[1]
                b0 = max(y_sa[2] / denom, 1e-3)
            end
        end
    end

    return Dict(:l0 => l0, :b0 => b0)
end

function initialize_states(y, m, trendtype, seasontype)
    if seasontype != "N"

        y_d = handle_seasonality(y, m, seasontype)
        init_seas = initialize_seasonal_components(y_d, m, seasontype)
        y_sa = adjust_y_sa(y, y_d, seasontype)
    else
        m = 1
        init_seas = nothing
        y_sa = y
    end

    maxn = min(max(10, 2 * m), length(y_sa))
    initial_values = calculate_initial_values(y_sa, trendtype, maxn)

    l0 = initial_values[:l0]
    b0 = initial_values[:b0]
    out = vcat([l0, b0], init_seas)
    out = [x for x in out if !isnothing(x)]
    return out
end
