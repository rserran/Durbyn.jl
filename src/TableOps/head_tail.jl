"""
    head(data, n::Integer=6)

Return the first `n` rows of a Tables.jl-compatible data source.

- If `n ≥ 0`, keep the first `min(n, nrows)` rows.
- If `n < 0`, drop the last `-n` rows.

# Examples
```julia
tbl = (x = 1:10, y = 11:20)
head(tbl, 3)   # (x = [1, 2, 3], y = [11, 12, 13])
head(tbl, -7)  # same — drop last 7
```
"""
function head(data, n::Integer=6)
    Tables.istable(data) || throw(ArgumentError(
        "head expects a Tables.jl-compatible source; use the array method for AbstractArrays."))
    ct = _to_columns(data)
    nr = _nrows(ct)
    keep = n >= 0 ? min(n, nr) : max(nr + n, 0)
    keep == 0 && return _subset_indices(ct, Int[])
    return _subset_indices(ct, collect(1:keep))
end

"""
    tail(data, n::Integer=6)

Return the last `n` rows of a Tables.jl-compatible data source.

- If `n ≥ 0`, keep the last `min(n, nrows)` rows.
- If `n < 0`, drop the first `-n` rows.

# Examples
```julia
tbl = (x = 1:10, y = 11:20)
tail(tbl, 3)   # (x = [8, 9, 10], y = [18, 19, 20])
tail(tbl, -7)  # same — drop first 7
```
"""
function tail(data, n::Integer=6)
    Tables.istable(data) || throw(ArgumentError(
        "tail expects a Tables.jl-compatible source; use the array method for AbstractArrays."))
    ct = _to_columns(data)
    nr = _nrows(ct)
    keep = n >= 0 ? min(n, nr) : max(nr + n, 0)
    keep == 0 && return _subset_indices(ct, Int[])
    return _subset_indices(ct, collect(nr - keep + 1 : nr))
end

"""
    head(gt::GroupedTable, n::Integer=6)

Return a new `GroupedTable` with the first `n` rows per group.

- If `n ≥ 0`, keep the first `min(n, group_size)` rows in each group.
- If `n < 0`, drop the last `-n` rows in each group.

# Examples
```julia
gt = groupby(tbl, [:category])
head(gt, 3)  # first 3 rows per category
```
"""
function head(gt::GroupedTable, n::Integer=6)
    new_indices = Vector{Vector{Int}}(undef, length(gt.keys))
    for (i, idxs) in enumerate(gt.indices)
        nr = length(idxs)
        keep = n >= 0 ? min(n, nr) : max(nr + n, 0)
        new_indices[i] = keep == 0 ? Int[] : idxs[1:keep]
    end
    return GroupedTable(gt.data, gt.keycols, gt.keys, new_indices)
end

"""
    tail(gt::GroupedTable, n::Integer=6)

Return a new `GroupedTable` with the last `n` rows per group.

- If `n ≥ 0`, keep the last `min(n, group_size)` rows in each group.
- If `n < 0`, drop the first `-n` rows in each group.

# Examples
```julia
gt = groupby(tbl, [:category])
tail(gt, 3)  # last 3 rows per category
```
"""
function tail(gt::GroupedTable, n::Integer=6)
    new_indices = Vector{Vector{Int}}(undef, length(gt.keys))
    for (i, idxs) in enumerate(gt.indices)
        nr = length(idxs)
        keep = n >= 0 ? min(n, nr) : max(nr + n, 0)
        new_indices[i] = keep == 0 ? Int[] : idxs[nr - keep + 1 : nr]
    end
    return GroupedTable(gt.data, gt.keycols, gt.keys, new_indices)
end
