# ─── ETS parameter initialization, validation, and admissibility ──────────────

function normalize_parameter(param)
    if isnothing(param)
        return nothing
    elseif param isa Bool
        return param ? 1.0 : 0.0
    elseif isnan(param)
        return nothing
    else
        return param
    end
end

function initparam(
    alpha::Union{Float64,Bool,Nothing},
    beta::Union{Float64,Bool,Nothing},
    gamma::Union{Float64,Bool,Nothing},
    phi::Union{Float64,Bool,Nothing},
    trendtype::String,
    seasontype::String,
    damped::Bool,
    lower::Vector{Float64},
    upper::Vector{Float64},
    m::Int,
    bounds::Symbol;
    nothing_as_nan::Bool = false,)


    if bounds === :admissible
        lower[1] = 0.0; lower[2] = 0.0; lower[3] = 0.0
        upper[1] = 1e-3; upper[2] = 1e-3; upper[3] = 1e-3
    elseif any(lower .> upper)
        throw(ArgumentError("Inconsistent parameter boundaries"))
    end

    # Select alpha
    if isnothing(alpha)
        m_eff = (seasontype == "N") ? 1 : m
        alpha = lower[1] + 0.2 * (upper[1] - lower[1]) / m_eff
        if alpha > 1 || alpha < 0
            alpha = lower[1] + 2e-3
        end
    end
    # Select beta
    if trendtype != "N" && (isnothing(beta))
        upper[2] = min(upper[2], alpha)
        beta = lower[2] + 0.1 * (upper[2] - lower[2])
        if beta < 0 || beta > alpha
            beta = alpha - 1e-3
        end
    end

    # Select gamma
    if seasontype != "N" && (isnothing(gamma))
        upper[3] = min(upper[3], 1 - alpha)
        gamma = lower[3] + 0.05 * (upper[3] - lower[3])
        if gamma < 0 || gamma > 1 - alpha
            gamma = 1 - alpha - 1e-3
        end
    end

    # Select phi
    if damped && isnothing(phi)
        phi = lower[4] + 0.99 * (upper[4] - lower[4])
        if phi < 0 || phi > 1
            phi = upper[4] - 1e-3
        end
    end

    if nothing_as_nan
        if isnothing(alpha)
            alpha = NaN
        end
        if isnothing(beta)
            beta = NaN
        end
        if isnothing(gamma)
            gamma = NaN
        end
        if isnothing(phi)
            phi = NaN
        end
    end

    return (alpha=alpha, beta=beta, gamma=gamma, phi=phi)
end

function admissible(
    alpha::Union{Float64,Nothing,Bool},
    beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool},
    phi::Union{Float64,Nothing,Bool},
    m::Int,
)

    alpha = normalize_parameter(alpha)
    beta = normalize_parameter(beta)
    gamma = normalize_parameter(gamma)
    phi = normalize_parameter(phi)

    if isnothing(phi)
        phi = 1.0
    elseif phi isa Bool
        phi = phi ? 1.0 : 0.0
    end

    if phi < 0 || phi > 1 + 1e-8
        return false
    end

    if isnothing(gamma)
        if alpha isa Bool
            alpha = alpha ? 1.0 : 0.0
        end
        if alpha < 1 - 1 / phi || alpha > 1 + 1 / phi
            return false
        end

        if !isnothing(beta)
            if beta isa Bool
                beta = beta ? 1.0 : 0.0
            end
            if beta < alpha * (phi - 1) || beta > (1 + phi) * (2 - alpha)
                return false
            end
        end
    elseif m > 1  # Seasonal model
        if isnothing(beta)
            beta = 0.0
        elseif beta isa Bool
            beta = beta ? 1.0 : 0.0
        end

        if gamma isa Bool
            gamma = gamma ? 1.0 : 0.0
        end
        if gamma < max(1 - 1 / phi - alpha, 0.0) || gamma > 1 + 1 / phi - alpha
            return false
        end

        if alpha isa Bool
            alpha = alpha ? 1.0 : 0.0
        end
        if alpha < 1 - 1 / phi - gamma * (1 - m + phi + phi * m) / (2 * phi * m)
            return false
        end

        if beta < -(1 - phi) * (gamma / m + alpha)
            return false
        end

        a = phi * (1 - alpha - gamma)
        b = alpha + beta - alpha * phi + gamma - 1
        c = repeat([alpha + beta - alpha * phi], m - 2)
        d = alpha + beta - phi
        P = vcat([a, b], c, [d, 1])

        poly = Polynomial(P)
        poly_roots = roots(poly)

        if maximum(abs, poly_roots) > 1 + 1e-10
            return false
        end
    end

    # Passed all tests
    return true
end

function check_param(
    alpha::Union{Float64,Nothing,Bool},
    beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool},
    phi::Union{Float64,Nothing,Bool},
    lower::Vector{Float64},
    upper::Vector{Float64},
    bounds::Symbol,
    m::Int,
)

    alpha = normalize_parameter(alpha)
    beta = normalize_parameter(beta)
    gamma = normalize_parameter(gamma)
    phi = normalize_parameter(phi)

    if bounds !== :admissible
        if !isnothing(alpha) && !isnan(alpha)
            if alpha < lower[1] || alpha > upper[1]
                return false
            end
        end
        if !isnothing(beta) && !isnan(beta)
            if beta < lower[2] || beta > alpha || beta > upper[2]
                return false
            end
        end
        if !isnothing(phi) && !isnan(phi)
            if phi < lower[4] || phi > upper[4]
                return false
            end
        end
        if !isnothing(gamma) && !isnan(gamma)
            if gamma < lower[3] || gamma > 1 - alpha || gamma > upper[3]
                return false
            end
        end
    end

    if bounds !== :usual
        if !admissible(alpha, beta, gamma, phi, m)
            return false
        end
    end
    return true
end

# Hot-path overload using int bounds code (avoids string comparison per NM iteration)
function check_param(
    alpha::Union{Float64,Nothing,Bool},
    beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool},
    phi::Union{Float64,Nothing,Bool},
    lower::Vector{Float64},
    upper::Vector{Float64},
    bounds::Int,
    m::Int,
)
    alpha = normalize_parameter(alpha)
    beta = normalize_parameter(beta)
    gamma = normalize_parameter(gamma)
    phi = normalize_parameter(phi)

    if bounds != BOUNDS_ADMISSIBLE
        if !isnothing(alpha) && !isnan(alpha)
            if alpha < lower[1] || alpha > upper[1]
                return false
            end
        end
        if !isnothing(beta) && !isnan(beta)
            if beta < lower[2] || beta > alpha || beta > upper[2]
                return false
            end
        end
        if !isnothing(phi) && !isnan(phi)
            if phi < lower[4] || phi > upper[4]
                return false
            end
        end
        if !isnothing(gamma) && !isnan(gamma)
            if gamma < lower[3] || gamma > 1 - alpha || gamma > upper[3]
                return false
            end
        end
    end

    if bounds != BOUNDS_USUAL
        if !admissible(alpha, beta, gamma, phi, m)
            return false
        end
    end
    return true
end
