function loess_estimate!(y::AbstractVector{Float64}, n::Int, bandwidth::Int, degree::Int,
                         eval_point::Float64, left_bound::Int, right_bound::Int,
                         w::AbstractVector{Float64}, use_weights::Bool, robustness_weights::AbstractVector{Float64})

    range = float(n) - 1.0
    half_width = max(eval_point - float(left_bound), float(right_bound) - eval_point)
    if bandwidth > n
        half_width += float((bandwidth - n) ÷ 2)
    end
    upper_threshold = 0.999 * half_width
    lower_threshold = 0.001 * half_width

    weight_sum = 0.0
    for j in left_bound:right_bound
        r = abs(float(j) - eval_point)
        if r <= upper_threshold
            if r <= lower_threshold || half_width == 0.0
                w[j] = 1.0
            else
                rr = r / half_width
                w[j] = (1.0 - rr^3)^3
            end
            if use_weights
                w[j] *= robustness_weights[j]
            end
            weight_sum += w[j]
        else
            w[j] = 0.0
        end
    end

    if weight_sum <= 0.0
        return 0.0, false
    end

    inv_weight_sum = 1.0 / weight_sum
    for j in left_bound:right_bound
        w[j] *= inv_weight_sum
    end

    if half_width > 0.0 && degree > 0

        a_mean = 0.0
        for j in left_bound:right_bound
            a_mean += w[j] * float(j)
        end
        b = eval_point - a_mean
        c = 0.0
        for j in left_bound:right_bound
            d = float(j) - a_mean
            c += w[j] * d^2
        end

        if sqrt(c) > 0.001 * range
            b /= c
            for j in left_bound:right_bound
                w[j] = w[j] * (b * (float(j) - a_mean) + 1.0)
            end
        end
    end

    ys = 0.0
    for j in left_bound:right_bound
        ys += w[j] * y[j]
    end
    return ys, true
end

function loess_smooth!(y::AbstractVector{Float64}, n::Int, bandwidth::Int, degree::Int, jump::Int,
                       use_weights::Bool, robustness_weights::AbstractVector{Float64},
                       ys::AbstractVector{Float64}, res::AbstractVector{Float64})

    if n < 2

        ys[firstindex(ys)] = y[1]
        return
    end

    new_jump = min(jump, n - 1)

    left_bound = 1
    right_bound = min(bandwidth, n)

    if bandwidth >= n
        left_bound = 1
        right_bound = n
        i = 1
        while i <= n
            eval_point = float(i)
            ysi, ok = loess_estimate!(y, n, bandwidth, degree, eval_point, left_bound, right_bound, res, use_weights, robustness_weights)
            if ok
                ys[firstindex(ys) - 1 + i] = ysi
            else
                ys[firstindex(ys) - 1 + i] = y[i]
            end
            i += new_jump
        end
    else
        if new_jump == 1
            half_bandwidth = (bandwidth + 1) ÷ 2
            left_bound = 1
            right_bound = bandwidth
            for i in 1:n
                if (i > half_bandwidth) && (right_bound != n)
                    left_bound += 1
                    right_bound += 1
                end
                eval_point = float(i)
                ysi, ok = loess_estimate!(y, n, bandwidth, degree, eval_point, left_bound, right_bound, res, use_weights, robustness_weights)
                if ok
                    ys[firstindex(ys) - 1 + i] = ysi
                else
                    ys[firstindex(ys) - 1 + i] = y[i]
                end
            end
        else
            half_bandwidth = (bandwidth + 1) ÷ 2
            i = 1
            while i <= n
                if i < half_bandwidth
                    left_bound = 1
                    right_bound = bandwidth
                elseif i >= n - half_bandwidth + 1
                    left_bound = n - bandwidth + 1
                    right_bound = n
                else
                    left_bound = i - half_bandwidth + 1
                    right_bound = bandwidth + i - half_bandwidth
                end
                eval_point = float(i)
                ysi, ok = loess_estimate!(y, n, bandwidth, degree, eval_point, left_bound, right_bound, res, use_weights, robustness_weights)
                if ok
                    ys[firstindex(ys) - 1 + i] = ysi
                else
                    ys[firstindex(ys) - 1 + i] = y[i]
                end
                i += new_jump
            end
        end
    end

    if new_jump != 1
        i = 1
        while i <= n - new_jump
            ysi = ys[firstindex(ys) - 1 + i]
            ysj = ys[firstindex(ys) - 1 + i + new_jump]
            delta = (ysj - ysi) / float(new_jump)
            for j in (i + 1):(i + new_jump - 1)
                ys[firstindex(ys) - 1 + j] = ysi + delta * float(j - i)
            end
            i += new_jump
        end

        k = ((n - 1) ÷ new_jump) * new_jump + 1
        if k != n

            eval_point = float(n)
            ysn, ok = loess_estimate!(y, n, bandwidth, degree, eval_point, left_bound, right_bound, res, use_weights, robustness_weights)
            if ok
                ys[firstindex(ys) - 1 + n] = ysn
            else
                ys[firstindex(ys) - 1 + n] = y[n]
            end
            if k != n - 1

                valk = ys[firstindex(ys) - 1 + k]
                valn = ys[firstindex(ys) - 1 + n]
                delta2 = (valn - valk) / float(n - k)
                for j in (k + 1):(n - 1)
                    ys[firstindex(ys) - 1 + j] = valk + delta2 * float(j - k)
                end
            end
        end
    end
    return
end
