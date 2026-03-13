# ─── BATS state space recursion ────────────────────────────────────────────────
#
# Core computational kernels for the BATS state space model. calc_model is
# the readable reference implementation; calc_bats_faster is the optimized
# version with hand-unrolled seasonal/ARMA indexing for zero-allocation loops.

function calc_model(
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

@inline function calc_bats_faster(
    y::AbstractMatrix,
    y_hat::AbstractMatrix,
    w_transpose::AbstractMatrix,
    F::AbstractMatrix,
    x::AbstractMatrix,
    g::AbstractMatrix,
    e::AbstractMatrix,
    x_nought::AbstractMatrix;
    seasonal_periods::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    beta_v::Union{Nothing,Any} = nothing,
    tau::Int = 0,
    p::Int = 0,
    q::Int = 0,
    Fx_buffer::Union{Nothing,AbstractVector} = nothing,
)

    adjBeta = (beta_v === nothing) ? 0 : 1
    lengthArma = p + q

    hasSeasonal = (seasonal_periods !== nothing)
    lengthSeasonal = hasSeasonal ? length(seasonal_periods) : 0

    nT = size(y, 2)
    nStates = size(x_nought, 1)
    Fcols = size(F, 2)

    if hasSeasonal
        @views y_hat[:, 1] = w_transpose[:, 1:(adjBeta+1)] * x_nought[1:(adjBeta+1), 1]

        previousS = 0
        for i = 1:lengthSeasonal
            sp = seasonal_periods[i]
            @views y_hat[1, 1] += x_nought[previousS+sp+adjBeta+1, 1]
            previousS += sp
        end
        if lengthArma > 0
            jstart = tau + adjBeta + 2
            @views y_hat[:, 1] .+=
                w_transpose[:, jstart:nStates] * x_nought[jstart:nStates, 1]
        end

        e[1, 1] = y[1, 1] - y_hat[1, 1]

        @views x[1:(adjBeta+1), 1] =
            F[1:(adjBeta+1), 1:(adjBeta+1)] * x_nought[1:(adjBeta+1), 1]

        if lengthArma > 0
            jstart = adjBeta + tau + 2
            @views x[1:(adjBeta+1), 1] .+=
                F[1:(adjBeta+1), jstart:Fcols] * x_nought[jstart:Fcols, 1]
        end

        previousS = 0
        for i = 1:lengthSeasonal
            sp = seasonal_periods[i]
            row_head = adjBeta + previousS + 2
            row_from = adjBeta + previousS + sp + 1
            x[row_head, 1] = x_nought[row_from, 1]

            if lengthArma > 0
                jstart = adjBeta + tau + 2
                @views x[row_head, 1] +=
                    dot(F[row_head, jstart:Fcols], x_nought[jstart:Fcols, 1])
            end

            rowDestStart = adjBeta + previousS + 3
            rowDestEnd = adjBeta + previousS + sp + 1
            rowSrcStart = adjBeta + previousS + 2
            rowSrcEnd = adjBeta + previousS + sp
            if rowDestStart <= rowDestEnd
                @views x[rowDestStart:rowDestEnd, 1] .= x_nought[rowSrcStart:rowSrcEnd, 1]
            end

            previousS += sp
        end

        if p > 0
            idx_p1 = adjBeta + tau + 2
            @views x[idx_p1, 1] = dot(F[idx_p1, idx_p1:Fcols], x_nought[idx_p1:Fcols, 1])

            if p > 1
                rowDestStart = adjBeta + tau + 3
                rowDestEnd = adjBeta + tau + p + 1
                rowSrcStart = adjBeta + tau + 2
                rowSrcEnd = adjBeta + tau + p
                if rowDestStart <= rowDestEnd
                    @views x[rowDestStart:rowDestEnd, 1] .=
                        x_nought[rowSrcStart:rowSrcEnd, 1]
                end
            end
        end

        if q > 0
            idx_q1 = adjBeta + tau + p + 2
            x[idx_q1, 1] = 0.0
            if q > 1
                rowDestStart = adjBeta + tau + p + 3
                rowDestEnd = adjBeta + tau + p + q + 1
                rowSrcStart = adjBeta + tau + p + 2
                rowSrcEnd = adjBeta + tau + p + q
                if rowDestStart <= rowDestEnd
                    @views x[rowDestStart:rowDestEnd, 1] .=
                        x_nought[rowSrcStart:rowSrcEnd, 1]
                end
            end
        end

        x[1, 1] += g[1, 1] * e[1, 1]

        if adjBeta == 1
            x[2, 1] += g[2, 1] * e[1, 1]
        end

        previousS = 0
        for i = 1:lengthSeasonal
            sp = seasonal_periods[i]
            rowS = adjBeta + previousS + 2
            x[rowS, 1] += g[rowS, 1] * e[1, 1]
            previousS += sp
        end

        if p > 0
            idx_p1 = adjBeta + tau + 2
            x[idx_p1, 1] += e[1, 1]
            if q > 0
                idx_q1 = adjBeta + tau + p + 2
                x[idx_q1, 1] += e[1, 1]
            end
        elseif q > 0
            idx_q1 = adjBeta + tau + 2
            x[idx_q1, 1] += e[1, 1]
        end

        @inbounds for t = 1:(nT-1)
            col_t = t + 1

            @views y_hat[:, col_t] .= w_transpose[:, 1:(adjBeta+1)] * x[1:(adjBeta+1), t]

            previousS = 0
            for i = 1:lengthSeasonal
                sp = seasonal_periods[i]
                @views y_hat[1, col_t] += x[previousS+sp+adjBeta+1, t]
                previousS += sp
            end

            if lengthArma > 0
                jstart = tau + adjBeta + 2
                @views y_hat[:, col_t] .+=
                    w_transpose[:, jstart:nStates] * x[jstart:nStates, t]
            end

            e[1, col_t] = y[1, col_t] - y_hat[1, col_t]

            @views x[1:(adjBeta+1), col_t] .=
                F[1:(adjBeta+1), 1:(adjBeta+1)] * x[1:(adjBeta+1), t]

            if lengthArma > 0
                jstart = adjBeta + tau + 2
                @views x[1:(adjBeta+1), col_t] .+=
                    F[1:(adjBeta+1), jstart:Fcols] * x[jstart:Fcols, t]
            end

            previousS = 0
            for i = 1:lengthSeasonal
                sp = seasonal_periods[i]

                row_head = adjBeta + previousS + 2
                row_from = adjBeta + previousS + sp + 1

                x[row_head, col_t] = x[row_from, t]

                if lengthArma > 0
                    jstart = adjBeta + tau + 2
                    @views x[row_head, col_t] +=
                        dot(F[row_head, jstart:Fcols], x[jstart:Fcols, t])
                end

                rowDestStart = adjBeta + previousS + 3
                rowDestEnd = adjBeta + previousS + sp + 1
                rowSrcStart = adjBeta + previousS + 2
                rowSrcEnd = adjBeta + previousS + sp
                if rowDestStart <= rowDestEnd
                    @views x[rowDestStart:rowDestEnd, col_t] .= x[rowSrcStart:rowSrcEnd, t]
                end

                previousS += sp
            end

            if p > 0
                idx_p1 = adjBeta + tau + 2
                @views x[idx_p1, col_t] = dot(F[idx_p1, idx_p1:Fcols], x[idx_p1:Fcols, t])

                if p > 1
                    rowDestStart = adjBeta + tau + 3
                    rowDestEnd = adjBeta + tau + p + 1
                    rowSrcStart = adjBeta + tau + 2
                    rowSrcEnd = adjBeta + tau + p
                    if rowDestStart <= rowDestEnd
                        @views x[rowDestStart:rowDestEnd, col_t] .=
                            x[rowSrcStart:rowSrcEnd, t]
                    end
                end
            end

            if q > 0
                idx_q1 = adjBeta + tau + p + 2
                x[idx_q1, col_t] = 0.0
                if q > 1
                    rowDestStart = adjBeta + tau + p + 3
                    rowDestEnd = adjBeta + tau + p + q + 1
                    rowSrcStart = adjBeta + tau + p + 2
                    rowSrcEnd = adjBeta + tau + p + q
                    if rowDestStart <= rowDestEnd
                        @views x[rowDestStart:rowDestEnd, col_t] .=
                            x[rowSrcStart:rowSrcEnd, t]
                    end
                end
            end

            x[1, col_t] += g[1, 1] * e[1, col_t]
            if adjBeta == 1
                x[2, col_t] += g[2, 1] * e[1, col_t]
            end

            previousS = 0
            for i = 1:lengthSeasonal
                sp = seasonal_periods[i]
                rowS = adjBeta + previousS + 2
                x[rowS, col_t] += g[rowS, 1] * e[1, col_t]
                previousS += sp
            end

            if p > 0
                idx_p1 = adjBeta + tau + 2
                x[idx_p1, col_t] += e[1, col_t]
                if q > 0
                    idx_q1 = adjBeta + tau + p + 2
                    x[idx_q1, col_t] += e[1, col_t]
                end
            elseif q > 0
                idx_q1 = adjBeta + tau + 2
                x[idx_q1, col_t] += e[1, col_t]
            end
        end

    else
        if Fx_buffer !== nothing
            y_hat[1, 1] = dot(view(w_transpose, 1, :), view(x_nought, :, 1))
            e[1, 1] = y[1, 1] - y_hat[1, 1]
            mul!(Fx_buffer, F, view(x_nought, :, 1))
            @. x[:, 1] = Fx_buffer + g * e[1, 1]

            @inbounds for t = 1:(nT-1)
                col_t = t + 1
                y_hat[1, col_t] = dot(view(w_transpose, 1, :), view(x, :, t))
                e[1, col_t] = y[1, col_t] - y_hat[1, col_t]
                mul!(Fx_buffer, F, view(x, :, t))
                @. x[:, col_t] = Fx_buffer + g * e[1, col_t]
            end
        else
            # Fallback path without buffer
            @views y_hat[:, 1] .= w_transpose * x_nought[:, 1]
            e[1, 1] = y[1, 1] - y_hat[1, 1]
            @views x[:, 1] .= F * x_nought[:, 1] .+ g .* e[1, 1]

            @inbounds for t = 1:(nT-1)
                col_t = t + 1
                @views y_hat[:, col_t] .= w_transpose * x[:, t]
                e[1, col_t] = y[1, col_t] - y_hat[1, col_t]
                @views x[:, col_t] .= F * x[:, t] .+ g .* e[1, col_t]
            end
        end
    end

    return nothing
end
