# ─── ETS core state space recursion kernels ───────────────────────────────────

function construct_states(
    level::AbstractArray,
    trend::AbstractArray,
    season::AbstractArray,
    trendtype::String,
    seasontype::String,
    m::Int,
)

    states = hcat(level)
    state_names = ["l"]

    if trendtype != "N"
        states = hcat(states, trend)
        push!(state_names, "b")
    end

    if seasontype != "N"
        nr = size(states, 1)
        @inbounds for i = 1:m

            seasonal_column = season[(m-i).+(1:nr)]
            states = hcat(states, seasonal_column)
        end

        seas_names = ["s$i" for i = 1:m]
        append!(state_names, seas_names)
    end

    initstates = states[1, :]

    return states, state_names, initstates
end

function ets_base(y, n, x, m, error, trend, season, alpha, beta, gamma, phi, e, amse, nmse)
    olds = zeros(Float64, max(24, m))
    s = zeros(Float64, max(24, m))
    f = zeros(Float64, 30)
    denom = zeros(Float64, 30)
    return ets_base(
        y,
        n,
        x,
        m,
        error,
        trend,
        season,
        alpha,
        beta,
        gamma,
        phi,
        e,
        amse,
        nmse,
        olds,
        s,
        f,
        denom,
    )
end

function ets_base(
    y,
    n,
    x,
    m,
    error,
    trend,
    season,
    alpha,
    beta,
    gamma,
    phi,
    e,
    amse,
    nmse,
    olds::AbstractVector{Float64},
    s::AbstractVector{Float64},
    f::AbstractVector{Float64},
    denom::AbstractVector{Float64},
)
    oldb = 0.0

    if m < 1
        m = 1
    end
    nmse_cap = min(nmse, 30)

    nstates = m * (season > 0) + 1 + (trend > 0)
    trend_offset = trend > 0 ? 1 : 0

    # Copy initial state components
    l = x[1]
    if trend > 0
        b = x[2]
    else
        b = 0.0
    end

    if season > 0
        @inbounds for j = 1:m
            s[j] = x[trend_offset+j+1]
        end
    end

    lik = 0.0
    lik2 = 0.0
    @inbounds for j = 1:nmse_cap
        amse[j] = 0.0
        denom[j] = 0.0
    end

    @inbounds for i = 1:n
        # Copy previous state
        oldl = l
        if trend > 0
            oldb = b
        end
        if season > 0
            for j = 1:m
                olds[j] = s[j]
            end
        end

        # One step forecast
        forecast_ets_base(oldl, oldb, olds, m, trend, season, phi, f, nmse_cap)

        f1 = f[1]
        if abs(f1 - -99999.0) < 1.0e-10
            lik = -99999.0
            return lik
        end

        if error == 1
            e[i] = y[i] - f1
        else
            if abs(f1) < 1.0e-10
                f_0 = f1 + 1.0e-10
            else
                f_0 = f1
            end
            e[i] = (y[i] - f1) / f_0
        end

        for j = 1:nmse_cap
            if (i + j - 1) <= n
                denom[j] += 1.0
                tmp = y[i+j-1] - f[j]
                amse[j] = (amse[j] * (denom[j] - 1.0) + (tmp * tmp)) / denom[j]
            end
        end

        # Update state
        l, b, s = update_ets_base(
            oldl,
            l,
            oldb,
            b,
            olds,
            s,
            m,
            trend,
            season,
            alpha,
            beta,
            gamma,
            phi,
            y[i],
        )

        # Store new state
        x[nstates*i+1] = l
        if trend > 0
            x[nstates*i+2] = b
        end
        if season > 0
            for j = 1:m
                x[nstates*i+trend_offset+j+1] = s[j]
            end
        end

        lik += e[i] * e[i]
        val = abs(f1)
        if val > 0.0
            lik2 += log(val)
        else
            lik2 += log(val + 1e-8)
        end
    end

    if lik > 0.0
        lik = n * log(lik)
    else
        lik = n * log(lik + 1e-8)
    end

    if error == 2
        lik += 2 * lik2
    end

    return lik
end

function forecast_ets_base(l, b, s, m, trend, season, phi, f, h)
    TOL = 1.0e-10
    phistar = phi
    @inbounds for i = 1:h
        if trend == 0
            f[i] = l
        elseif trend == 1
            f[i] = l + phistar * b
        elseif b < 0
            f[i] = -99999.0
        else
            f[i] = l * (b ^ phistar)
        end

        j = mod1(m - i + 1, m)

        if season == 1
            f[i] += s[j]
        elseif season == 2
            f[i] *= s[j]
        end

        if i < h
            if abs(phi - 1.0) < TOL
                phistar += 1.0
            else
                phistar += phi^(i + 1)
            end
        end
    end
end

function update_ets_base(
    oldl,
    l,
    oldb,
    b,
    olds,
    s,
    m,
    trend,
    season,
    alpha,
    beta,
    gamma,
    phi,
    y,
)
    # New Level
    if trend == 0
        q = oldl
        phib = 0
    elseif trend == 1
        phib = phi * oldb
        q = oldl + phib
    elseif abs(phi - 1.0) < 1.0e-10
        phib = oldb
        q = oldl * oldb
    else
        phib = oldb^phi
        q = oldl * phib
    end

    # Season
    if season == 0
        p = y
    elseif season == 1
        p = y - olds[m]
    else
        if abs(olds[m]) < 1.0e-10
            p = 1.0e10
        else
            p = y / olds[m]
        end
    end

    l = q + alpha * (p - q)

    # New Growth
    if trend > 0
        if trend == 1
            r = l - oldl
        else
            if abs(oldl) < 1.0e-10
                r = 1.0e10
            else
                r = l / oldl
            end
        end
        b = phib + (beta / alpha) * (r - phib)
    end

    # New Seasonal
    if season > 0
        if season == 1
            t = y - q
        else # if season == 2
            if abs(q) < 1.0e-10
                t = 1.0e10
            else
                t = y / q
            end
        end
        @inbounds s[1] = olds[m] + gamma * (t - olds[m]) # s[t] = s[t - m] + gamma * (t - s[t - m])
        @inbounds for j = 2:m
            s[j] = olds[j-1] # s[t] = s[t]
        end
    end

    return l, b, s
end

function simulate_ets_base(x, m, error, trend, season, alpha, beta, gamma, phi, h, y, e)
    oldb = 0.0
    olds = zeros(24)
    s = zeros(24)
    f = zeros(10)

    if m > 24 && season > 0
        return
    elseif m < 1
        m = 1
    end

    l = x[1]
    b = 0.0
    if trend > 0
        b = x[2]
    end

    if season > 0
        @inbounds for j = 1:m
            s[j] = x[(trend>0)+j+1]
        end
    end

    @inbounds for i = 1:h
        oldl = l
        if trend > 0
            oldb = b
        end
        if season > 0
            for j = 1:m
                olds[j] = s[j]
            end
        end

        forecast_ets_base(oldl, oldb, olds, m, trend, season, phi, f, 1)

        if abs(f[1] - -99999.0) < 1.0e-10
            y[1] = -99999.0
            return
        end

        if error == 1
            y[i] = f[1] + e[i]
        else
            y[i] = f[1] * (1.0 + e[i])
        end

        # Update state
        l, b, s = update_ets_base(
            oldl,
            l,
            oldb,
            b,
            olds,
            s,
            m,
            trend,
            season,
            alpha,
            beta,
            gamma,
            phi,
            y[i],
        )
    end
end

function forecast(
    x::AbstractVector,
    m::Int,
    trend::Int,
    season::Int,
    phi::Float64,
    h::Int,
    f::AbstractVector,
)

    if (m > 24) && (season > 0)
        return
    elseif m < 1
        m = 1
    end

    l = Float64(x[1])
    b = trend > 0 ? Float64(x[2]) : 0.0
    s = zeros(Float64, 24)

    if season > 0
        offset = trend > 0 ? 2 : 1
        @inbounds for j = 1:m
            s[j] = Float64(x[offset+j])
        end
    end

    forecast_ets_base(l, b, s, m, trend, season, phi, f, h)
end

# ─── Residual calculation wrappers ────────────────────────────────────────────

function calculate_residuals(
    y::AbstractArray,
    m::Int,
    init_state::AbstractArray,
    errortype::String,
    trendtype::String,
    seasontype::String,
    damped::Bool,
    alpha::Union{Float64,Nothing,Bool},
    beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool},
    phi::Union{Float64,Nothing,Bool},
    nmse::Int,
)
    err = ets_model_type_code(errortype)
    trend = ets_model_type_code(trendtype)
    season = ets_model_type_code(seasontype)
    workspace = ETSWorkspace(length(y), m, nmse, length(init_state))
    likelihood, amse, e, x = calculate_residuals(
        y,
        m,
        init_state,
        err,
        trend,
        season,
        damped,
        alpha,
        beta,
        gamma,
        phi,
        nmse,
        workspace,
    )
    return likelihood, copy(amse), copy(e), copy(x)
end

function calculate_residuals!(
    workspace::ETSWorkspace,
    y::AbstractArray,
    m::Int,
    init_state::AbstractArray,
    errortype::Int,
    trendtype::Int,
    seasontype::Int,
    damped::Bool,
    alpha::Union{Float64,Nothing,Bool},
    beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool},
    phi::Union{Float64,Nothing,Bool},
    nmse::Int,
)
    n = length(y)
    p = length(init_state)
    x_len = p * (n + 1)
    if length(workspace.x) < x_len
        throw(ArgumentError("ETSWorkspace.x is too small for state dimension $p and series length $n"))
    end
    if length(workspace.e) < n
        throw(ArgumentError("ETSWorkspace.e is too small for series length $n"))
    end
    if length(workspace.amse) < nmse
        throw(ArgumentError("ETSWorkspace.amse is too small for nmse=$nmse"))
    end

    x = workspace.x
    e = workspace.e
    amse = workspace.amse
    @inbounds x[1:p] .= init_state
    if nmse > 0
        fill!(view(amse, 1:nmse), 0.0)
    end

    if !damped
        phi = 1.0
    end
    if trendtype == 0
        beta = 0.0
    end
    if seasontype == 0
        gamma = 0.0
    end

    alpha_f = Float64(alpha)
    beta_f = Float64(beta)
    gamma_f = Float64(gamma)
    phi_f = Float64(phi)

    likelihood = ets_base(
        y,
        n,
        x,
        m,
        errortype,
        trendtype,
        seasontype,
        alpha_f,
        beta_f,
        gamma_f,
        phi_f,
        e,
        amse,
        nmse,
        workspace.olds,
        workspace.s,
        workspace.f,
        workspace.denom,
    )

    if abs(likelihood + 99999.0) < 1e-7
        likelihood = NaN
    end

    return likelihood, p
end

function calculate_residuals(
    y::AbstractArray,
    m::Int,
    init_state::AbstractArray,
    errortype::Int,
    trendtype::Int,
    seasontype::Int,
    damped::Bool,
    alpha::Union{Float64,Nothing,Bool},
    beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool},
    phi::Union{Float64,Nothing,Bool},
    nmse::Int,
    workspace::ETSWorkspace,
)
    likelihood, p = calculate_residuals!(
        workspace,
        y,
        m,
        init_state,
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
    n = length(y)
    x_len = p * (n + 1)
    x = reshape(view(workspace.x, 1:x_len), p, n + 1)'
    amse = view(workspace.amse, 1:nmse)
    e = view(workspace.e, 1:n)
    return likelihood, amse, e, x
end
