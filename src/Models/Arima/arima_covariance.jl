

"""
    transform_unconstrained_to_ar_params!(p, raw, dest)

Convert a vector of unconstrained real numbers to autoregressive (AR) coefficients.

This routine implements the two-step transformation used in the R reference
implementation: it first maps the unconstrained `raw` values into the open
interval `(-1, 1)` via the hyperbolic tangent to obtain partial autocorrelation
coefficients (PACF), and then applies the Durbin-Levinson recursion to obtain
the corresponding AR parameters.  The result is written in-place to `dest`.

Arguments
---------
- `p::Int`: the number of parameters to transform.  Must satisfy `p ≤ 100`.
- `raw::AbstractVector`: a vector of length at least `p` containing the
  unconstrained parameters.
- `new::AbstractVector`: a preallocated vector of length at least `p` into
  which the AR coefficients will be written.  On entry its contents are
  ignored; on exit it contains the transformed values.

Throws
------
`ArgumentError` if `p > 100`.

Notes
-----
This function mutates its `new` argument.  A working copy of the first
`p` elements is used internally to avoid aliasing.
"""

function transform_unconstrained_to_ar_params!(
    p::Int,
    raw::AbstractVector,
    new::AbstractVector,
)
    if p > 100
        throw(ArgumentError("The function can only transform 100 parameters in arima0"))
    end

    @inbounds new[1:p] .= tanh.(raw[1:p])
    work = copy(new[1:p])

    @inbounds for j = 2:p
        a = new[j]
        for k = 1:(j-1)
            work[k] -= a * new[j-k]
        end
        new[1:(j-1)] .= work[1:(j-1)]
    end

end


"""
    compute_arima_transform_gradient(x, arma)

Compute the Jacobian matrix of the ARIMA parameter transformation.

This function numerically approximates the gradient (Jacobian) of the
transformation that maps unconstrained parameter vectors to the ARIMA
parameter space.  It mirrors the behaviour of the R/C implementation
`ARIMA_Gradtrans` by perturbing each parameter individually by a small
epsilon (`1e-3`) and computing the resulting change in the transformed
coefficients.  The result is returned as a square matrix where each
row corresponds to the gradient with respect to one parameter.

Arguments
---------
- `x::AbstractArray`: the vector of input parameters (potentially
  partially constrained).  Its length determines the dimension of the
  Jacobian.
- `arma::AbstractArray`: a vector encoding the ARIMA order.  The first
  three entries correspond to the number of non-seasonal AR terms (`p`),
  the number of non-seasonal MA terms (`q`), and the number of
  seasonal AR terms (`P`), respectively.  Only these three values are
  used by this function.

Returns
-------
A dense `n*n` matrix of `Float64` where `n = length(x)`.  Elements
outside the blocks corresponding to AR parameters are zero.
"""

function compute_arima_transform_gradient(x::AbstractArray, arma::AbstractArray)
    eps = 1e-3
    mp, mq, msp = arma[1:3]
    if mp > 100 || msp > 100
        throw(ArgumentError("AR order > 100 not supported (p=$mp, P=$msp)"))
    end
    n = length(x)
    y = Matrix{Float64}(I, n, n)

    w1 = Vector{Float64}(undef, 100)
    w2 = Vector{Float64}(undef, 100)
    w3 = Vector{Float64}(undef, 100)

    if mp > 0

        for i = 1:mp
            w1[i] = x[i]
        end

        transform_unconstrained_to_ar_params!(mp, w1, w2)

        for i = 1:mp
            w1[i] += eps
            transform_unconstrained_to_ar_params!(mp, w1, w3)
            for j = 1:mp
                y[i, j] = (w3[j] - w2[j]) / eps
            end
            w1[i] -= eps
        end
    end

    if msp > 0
        v = mp + mq
        for i = 1:msp
            w1[i] = x[i+v]
        end
        transform_unconstrained_to_ar_params!(msp, w1, w2)
        for i = 1:msp
            w1[i] += eps
            transform_unconstrained_to_ar_params!(msp, w1, w3)
            for j = 1:msp
                y[i+v, j+v] = (w3[j] - w2[j]) / eps
            end
            w1[i] -= eps
        end
    end
    return y
end

"""
    undo_arima_parameter_transform(x, arma)

Undo the ARIMA parameter transformations applied to the AR coefficients.

Given a vector of transformed parameters `x` and the ARIMA specification `arma`,
this function applies the inverse of the parameter transformation used in
`transform_unconstrained_to_ar_params!` to restore the original (unconstrained)
parameters.  It mirrors the behaviour of the C function `ARIMA_undoPars`.
The result is returned as a copy of `x` with the AR terms replaced by
their inverse-transformed values.

Arguments
---------
- `x::AbstractArray`: a vector containing the transformed parameters.
- `arma::AbstractArray`: a vector encoding the ARIMA order.  Only the
  first three elements (`p`, `q`, `P`) are used here.

Returns
-------
A new vector of the same length as `x` containing the untransformed
parameters.
"""
# The function is tested works as expected
function undo_arima_parameter_transform(x::AbstractArray, arma::AbstractArray)
    mp, mq, msp = arma[1:3]
    res = copy(x)
    if mp > 0
        transform_unconstrained_to_ar_params!(mp, x, res)
    end
    v = mp + mq
    if msp > 0
        transform_unconstrained_to_ar_params!(msp, @view(x[v+1:end]), @view(res[v+1:end]))
    end
    return res
end

"""
    time_series_convolution(a, b)

Perform a discrete convolution between two numeric sequences.

This function computes the convolution of vectors `a` and `b`, returning
an array whose length is `length(a) + length(b) - 1`.  It corresponds to
the helper `TSconv` in the R/C source and is used to construct
difference operators for the ARIMA model.

Arguments
---------
- `a::AbstractArray`: the first sequence.
- `b::AbstractArray`: the second sequence.

Returns
-------
A vector containing the convolution of `a` and `b`.
"""
# The function is tested works as expected
function time_series_convolution(a::AbstractArray, b::AbstractArray)
    na = length(a)
    nb = length(b)
    nab = na + nb - 1
    ab = zeros(Float64, nab)

    for i = 1:na
        for j = 1:nb
            ab[i+j-1] += a[i] * b[j]
        end
    end
    return ab
end

"""
    update_least_squares!(n_parameters, xnext, xrow, ynext, d, rbar, thetab)

Internal helper used by `compute_q0_covariance_matrix` to update the
least-squares regression quantities when processing autocovariances.  This
function closely follows the Fortran routine used in the R implementation.
It updates the arrays `d`, `rbar` and `thetab` in place, based on the
incoming observation `xnext` and response `ynext`.

Arguments
---------
- `n_parameters::Int`: the number of parameters in the regression.
- `xnext::AbstractArray`: the new predictor values.
- `xrow::AbstractArray`: a working array to hold modified predictor values.
- `ynext::Float64`: the new response value.
- `d::AbstractArray`: diagonal of the regression matrix to be updated.
- `rbar::AbstractArray`: upper triangular portion of the regression matrix.
- `thetab::AbstractArray`: regression coefficients to be updated.

This function mutates `d`, `rbar` and `thetab` and returns nothing.
"""
# The function is tested works as expected
function update_least_squares!(
    n_parameters::Int,
    xnext::AbstractArray,
    xrow::AbstractArray,
    ynext::Float64,
    d::AbstractArray,
    rbar::AbstractArray,
    thetab::AbstractArray,
)

for i = 1:n_parameters
        xrow[i] = xnext[i]
    end

    ithisr = 1
    for i = 1:n_parameters
        if xrow[i] != 0.0
            xi = xrow[i]
            di = d[i]
            dpi = di + xi * xi
            d[i] = dpi
            cbar = dpi != 0.0 ? di / dpi : Inf
            sbar = dpi != 0.0 ? xi / dpi : Inf

            for k = (i+1):n_parameters
                xk = xrow[k]
                rbthis = rbar[ithisr]
                xrow[k] = xk - xi * rbthis
                rbar[ithisr] = cbar * rbthis + sbar * xk
                ithisr += 1
            end

            xk = ynext
            ynext = xk - xi * thetab[i]
            thetab[i] = cbar * thetab[i] + sbar * xk

            if di == 0.0
                return
            end
        else
            ithisr = ithisr + n_parameters - i
        end
    end

    return
end

"""
    inverse_ar_parameter_transform(ϕ)

Compute the inverse transformation from AR coefficients to unconstrained
parameters.

This function reverses the Durbin-Levinson transformation applied by
`transform_unconstrained_to_ar_params!`.  Given a vector of AR
coefficients `ϕ`, it returns the corresponding unconstrained parameters
on the real line by running the recursion backwards and applying the
inverse hyperbolic tangent.

Arguments
---------
- `ϕ::AbstractVector`: vector of AR coefficients.

Returns
-------
A vector of the same length as `ϕ` containing the unconstrained
parameters.
"""
# The function is tested works as expected
function inverse_ar_parameter_transform(ϕ::AbstractVector)
    p = length(ϕ)
    new = Array{Float64}(undef, p)
    copy!(new, ϕ)
    work = similar(new)
    # Perform the backward Durbin-Levinson recursion.  This recovers the
    # partial autocorrelations from the AR coefficients.
    # This is confusing be carriful.
    for j in p:-1:2
        a = new[j]
        denom = 1 - a^2
        denom ≠ 0 || throw(ArgumentError("Encountered unit root at j=$j (a=±1)."))
        for k in 1:j-1
            work[k] = (new[k] + a * new[j-k]) / denom
        end
        new[1:j-1] = work[1:j-1]
    end
    return map(x -> abs(x) <= 1 ? atanh(x) : NaN, new)
end

"""
    inverse_arima_parameter_transform(θ, arma)

Apply the inverse ARIMA parameter transformation to a parameter vector.

Given a parameter vector `θ` and the ARIMA specification `arma`, this
function applies the inverse transformation used in the ARIMA fitting
process to recover the unconstrained parameters.  It reverses the
seasonal and non-seasonal AR transformations by calling
`inverse_ar_parameter_transform` on the appropriate slices.

Arguments
---------
- `θ::AbstractVector`: vector of transformed parameters.
- `arma::AbstractVector{Int}`: vector encoding the ARIMA order.  The
  first three elements correspond to the non-seasonal AR (`p`), MA (`q`)
  and seasonal AR (`P`) orders.

Returns
-------
A new vector containing the unconstrained parameters.
"""
# The function is tested works as expected
function inverse_arima_parameter_transform(θ::AbstractVector, arma::AbstractVector{Int})
    mp, mq, msp = arma
    n = length(θ)
    v = mp + mq
    v + msp ≤ n || throw(ArgumentError("Sum mp+mq+msp exceeds length(θ)"))
    raw = Array{Float64}(undef, n)
    copy!(raw, θ)
    transformed = raw

    # non‐seasonal AR
    if mp > 0
        transformed[1:mp] = inverse_ar_parameter_transform(raw[1:mp])
    end

    # seasonal AR
    if msp > 0
        transformed[v+1:v+msp] = inverse_ar_parameter_transform(raw[v+1:v+msp])
    end

    return transformed
end

# Helper for getQ0
function compute_v(phi::AbstractArray, theta::AbstractArray, r::Int)
    p = length(phi)
    q = length(theta)
    num_params = r * (r + 1) ÷ 2
    V = zeros(Float64, num_params)

    ind = 0
    for j = 0:(r-1)
        vj = 0.0
        if j == 0
            vj = 1.0
        elseif (j - 1) < q && (j - 1) ≥ 0
            vj = theta[j-1+1]
        end

        for i = j:(r-1)
            vi = 0.0
            if i == 0
                vi = 1.0
            elseif (i - 1) < q && (i - 1) ≥ 0
                vi = theta[i-1+1]
            end

            V[ind+1] = vi * vj
            ind += 1
        end
    end
    return V
end

# Helper for getQ0
function handle_r_equals_1(p::Int, phi::AbstractArray)
    res = zeros(Float64, 1, 1)
    if p == 0

        res[1, 1] = 1.0
    else

        res[1, 1] = 1.0 / (1.0 - phi[1]^2)
    end
    return res
end


# Helper for getQ0
function handle_p_equals_0(V::AbstractArray, r::Int)
    num_params = r * (r + 1) ÷ 2
    res = zeros(Float64, r * r)

    ind = num_params
    indn = num_params

    for i = 0:(r-1)
        for j = 0:i
            ind -= 1

            res[ind + 1] = V[ind+1]

            if j != 0
                indn -= 1
                res[ind+1] += res[indn+1]
            end
        end
    end
    return res
end

# Helper for getQ0
function handle_p_greater_than_0(
    V::AbstractArray,
    phi::AbstractArray,
    p::Int,
    r::Int,
    num_params::Int,
    nrbar::Int,
)

    res = zeros(Float64, r * r)

    rbar = zeros(Float64, nrbar)
    thetab = zeros(Float64, num_params)
    xnext = zeros(Float64, num_params)
    xrow = zeros(Float64, num_params)

    ind = 0
    ind1 = -1
    npr = num_params - r
    npr1 = npr + 1
    indj = npr
    ind2 = npr - 1

    for j = 0:(r-1)

        phij = (j < p) ? phi[j+1] : 0.0

        xnext[indj+1] = 0.0
        indj += 1

        indi = npr1 + j
        for i = j:(r-1)
            ynext = V[ind+1]
            ind += 1

            phii = (i < p) ? phi[i+1] : 0.0

            if j != (r - 1)
                xnext[indj+1] = -phii
                if i != (r - 1)
                    xnext[indi+1] -= phij
                    ind1 += 1
                    xnext[ind1+1] = -1.0
                end
            end

            xnext[npr+1] = -phii * phij
            ind2 += 1
            if ind2 >= num_params
                ind2 = 0
            end
            xnext[ind2+1] += 1.0

            update_least_squares!(num_params, xnext, xrow, ynext, res, rbar, thetab)

            xnext[ind2+1] = 0.0
            if i != (r - 1)
                xnext[indi+1] = 0.0
                indi += 1
                xnext[ind1+1] = 0.0
            end
        end
    end

    ithisr = nrbar - 1
    im = num_params - 1

    for i = 0:(num_params-1)
        bi = thetab[im+1]
        jm = num_params - 1
        for j = 0:(i-1)

            bi -= rbar[ithisr+1] * res[jm+1]

            ithisr -= 1
            jm -= 1
        end
        res[im+1] = bi
        im -= 1
    end

    xcopy = zeros(Float64, r)
    ind = npr
    for i = 0:(r-1)
        xcopy[i+1] = res[ind+1]
        ind += 1
    end

    ind = num_params - 1
    ind1 = npr - 1
    for i = 1:(npr)
        res[ind+1] = res[ind1+1]
        ind -= 1
        ind1 -= 1
    end

    for i = 0:(r-1)
        res[i+1] = xcopy[i+1]
    end

    return res
end

# Helper for getQ0
function unpack_full_matrix(res_flat::AbstractArray, r::Int)
    num_params = r * (r + 1) ÷ 2

    for i = (r-1):-1:1
        for j = (r-1):-1:i

            idx = i * r + j
            res_flat[idx+1] = res_flat[num_params]
            num_params -= 1
        end
    end

    for i = 0:(r-1)
        for j = (i+1):(r-1)

            res_flat[j*r+i+1] = res_flat[i*r+j+1]
        end
    end

    return reshape(res_flat, r, r)
end



"""
    compute_q0_covariance_matrix(phi, theta)

Compute the initial state covariance matrix for the AR component of an ARIMA model.

This function implements the algorithm described in the R `getQ0` function.  It
takes the AR (`phi`) and MA (`theta`) coefficient vectors and returns the
covariance matrix `Q₀` used to initialize the state-space representation of the
ARIMA model.  Internally it constructs the vector of autocovariances and
invokes a series of helper functions to fill in the appropriate blocks of
the matrix.

Arguments
---------
- `phi::AbstractArray`: vector of non-seasonal AR coefficients.
- `theta::AbstractArray`: vector of non-seasonal MA coefficients.

Returns
-------
A symmetric matrix of size `r*r`, where `r = max(length(phi), length(theta) + 1)`.
"""
# The function is tested works as expected
function compute_q0_covariance_matrix(phi::AbstractArray, theta::AbstractArray)
    p = length(phi)
    q = length(theta)

    r = max(p, q + 1)
    num_params = r * (r + 1) ÷ 2
    nrbar = num_params * (num_params - 1) ÷ 2

    V = compute_v(phi, theta, r)

    if r == 1
        return handle_r_equals_1(p, phi)
    end

    if p > 0

        res_flat = handle_p_greater_than_0(V, phi, p, r, num_params, nrbar)
    else

        res_flat = handle_p_equals_0(V, r)
    end

    res_full = unpack_full_matrix(res_flat, r)
    return res_full
end

"""
    compute_q0_bis_covariance_matrix(phi, theta, tol)

Compute the initial covariance matrix for the AR component of an ARIMA model
using the Rossignol (2011) method.

The original C code in R exposes two methods for computing the initial
covariance matrix used by the Kalman filter: the Gardner (1980) approach
(`getQ0`) and the Rossignol (2011) approach (`getQ0bis`).  The latter is
more computationally intensive and relies on numerically solving a set of
Yule-Walker equations.  For the purposes of this translation we provide
a simplified implementation: this function simply delegates to
`compute_q0_covariance_matrix`, which corresponds to the Gardner method.
If a more faithful implementation is required, this function can be
replaced by an appropriate algorithm.  The parameter `tol` is currently
ignored.

Arguments
---------
- `phi::AbstractArray`: vector of non-seasonal AR coefficients.
- `theta::AbstractArray`: vector of non-seasonal MA coefficients.
- `tol::Real`: tolerance parameter (unused).

Returns
-------
A symmetric matrix of size `r*r`, where `r = max(length(phi), length(theta) + 1)`.
"""
function compute_q0_bis_covariance_matrix(phi::AbstractVector{<:Real},
                                          theta::AbstractVector{<:Real},
                                          tol::Real = eps(Float64))

    φ = Float64.(phi)
    θ = Float64.(theta)

    p = length(φ)
    q = length(θ)
    r = max(p, q + 1)

    ttheta = zeros(Float64, r + q)
    @inbounds ttheta[1] = 1.0
    @inbounds for i in 1:q
        ttheta[i + 1] = θ[i]
    end

    P = zeros(Float64, r, r)

    if p > 0
        r2 = max(p + q, p + 1)

        tphi = Vector{Float64}(undef, p + 1)
        @inbounds tphi[1] = 1.0
        @inbounds for i in 1:p
            tphi[i + 1] = -φ[i]
        end

        Γ = zeros(Float64, r2, r2)

        @inbounds for j0 in 0:(r2-1)
            j_idx = j0 + 1
            for i0 in j0:(r2-1)
                d = i0 - j0
                if d <= p
                    Γ[j_idx, i0 + 1] += tphi[d + 1]
                end
            end
        end

        @inbounds for i0 in 0:(r2-1)
            i_idx = i0 + 1
            for j0 in 1:(r2-1)
                s = i0 + j0
                if s <= p
                    Γ[j0 + 1, i_idx] += tphi[s + 1]
                end
            end
        end

        g = zeros(Float64, r2)
        @inbounds g[1] = 1.0
        u = let
            ok = true
            κ = Inf
            try
                κ = cond(Γ)
            catch
                ok = false
            end

            if ok && isfinite(κ) && κ < 1/tol
                Γ \ g
            else
                (Γ + tol * I) \ g
            end
        end

        @inbounds for i0 in 0:(r-1)
            i_idx = i0 + 1
            φ_i_base = i0 + 1
            k_max = p - 1 - i0

            for j0 in i0:(r-1)
                j_idx = j0 + 1
                φ_j_base = j0 + 1
                m_max = p - 1 - j0
                acc = 0.0

                for k0 in 0:k_max
                    φ_ik = φ[φ_i_base + k0]
                    L_start = k0
                    L_end = k0 + q

                    for L0 in L_start:L_end
                        tLk = ttheta[L0 - k0 + 1]
                        φ_ik_tLk = φ_ik * tLk

                        for m0 in 0:m_max
                            φ_jm = φ[φ_j_base + m0]
                            φ_product = φ_ik_tLk * φ_jm
                            n_start = m0
                            n_end = m0 + q

                            for n0 in n_start:n_end
                                tnm = ttheta[n0 - m0 + 1]
                                u_idx = abs(L0 - n0) + 1
                                acc += φ_product * tnm * u[u_idx]
                            end
                        end
                    end
                end
                P[i_idx, j_idx] += acc
            end
        end

        rrz = zeros(Float64, q)
        if q > 0
            @inbounds for i0 in 0:(q-1)
                i_idx = i0 + 1
                val = ttheta[i_idx]
                jstart = max(0, i0 - p)
                for j0 in jstart:(i0-1)
                    val -= rrz[j0 + 1] * tphi[i0 - j0 + 1]
                end
                rrz[i_idx] = val
            end
        end

        @inbounds for i0 in 0:(r-1)
            i_idx = i0 + 1
            k_max_i = p - 1 - i0

            for j0 in i0:(r-1)
                j_idx = j0 + 1
                k_max_j = p - 1 - j0
                acc = 0.0

                for k0 in 0:k_max_i
                    φ_ik = φ[i0 + k0 + 1]
                    L_start = k0 + 1
                    L_end = k0 + q

                    for L0 in L_start:L_end
                        j0_L0 = j0 + L0
                        if j0_L0 < q + 1
                            acc += φ_ik * ttheta[j0_L0 + 1] * rrz[L0 - k0]
                        end
                    end
                end

                for k0 in 0:k_max_j
                    φ_jk = φ[j0 + k0 + 1]
                    L_start = k0 + 1
                    L_end = k0 + q

                    for L0 in L_start:L_end
                        i0_L0 = i0 + L0
                        if i0_L0 < q + 1
                            acc += φ_jk * ttheta[i0_L0 + 1] * rrz[L0 - k0]
                        end
                    end
                end

                P[i_idx, j_idx] += acc
            end
        end
    end

    @inbounds for i0 in 0:(r-1)
        i_idx = i0 + 1
        for j0 in i0:(r-1)
            j_idx = j0 + 1
            k_max = q - j0
            acc = 0.0
            @simd for k0 in 0:k_max
                acc += ttheta[i0 + k0 + 1] * ttheta[j0 + k0 + 1]
            end
            P[i_idx, j_idx] += acc
        end
    end

    @inbounds for i in 1:r
        for j in (i+1):r
            P[j, i] = P[i, j]
        end
    end

    return P
end

"""
    transform_arima_parameters(params_in, arma, trans)

Transform a flat parameter vector into AR and MA coefficient vectors.

This function expands and optionally transforms the parameters of an ARIMA
model.  Given a vector of raw parameters `params_in` and the ARIMA order
`arma`, it produces the non-seasonal and seasonal AR (`phi`) and MA
(`theta`) coefficient vectors.  If `trans` is `true` the unconstrained
parameters are first passed through `transform_unconstrained_to_ar_params!`
for stability.

Arguments
---------
- `params_in::AbstractArray`: input parameter vector.
- `arma::Vector{Int}`: model specification `[p, q, P, Q, s, d, D]`.
- `trans::Bool`: whether to apply the parameter transformation before
  expansion.

Returns
-------
A tuple `(phi, theta)` containing the expanded AR and MA coefficient vectors.
"""
# The function is tested works as expected
function transform_arima_parameters(
    params_in::AbstractArray,
    arma::Vector{Int},
    trans::Bool,
)
mp, mq, msp, msq, ns = arma
    p = mp + ns * msp
    q = mq + ns * msq

    phi = zeros(Float64, p)
    theta = zeros(Float64, q)
    params = copy(params_in)

    if trans
        if mp > 0
            transform_unconstrained_to_ar_params!(mp, params_in, params)
        end
        v = mp + mq
        if msp > 0
            transform_unconstrained_to_ar_params!(msp, @view(params_in[v+1:end]), @view(params[v+1:end]))
        end
    end

    if ns > 0
        @inbounds phi[1:mp] .= params[1:mp]
        @inbounds theta[1:mq] .= params[mp+1:mp+mq]

        @inbounds for j = 0:(msp-1)
            phi[(j+1)*ns] += params[mp+mq+j+1]
            for i = 0:(mp-1)
                phi[((j+1)*ns)+(i+1)] -= params[i+1] * params[mp+mq+j+1]
            end
        end

        @inbounds for j = 0:(msq-1)
            theta[(j+1)*ns] += params[mp+mq+msp+j+1]
            for i = 0:(mq-1)
                theta[((j+1)*ns)+(i+1)] += params[mp+i+1] * params[mp+mq+msp+j+1]
            end
        end
    else
        @inbounds phi[1:mp] .= params[1:mp]
        @inbounds theta[1:mq] .= params[mp+1:mp+mq]
    end

    return (phi, theta)
end
