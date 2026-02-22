using Test
using Durbyn

import Durbyn.Utils: NamedMatrix
import Durbyn.TableOps: groupby, GroupedTable

@testset "head & tail" begin

    # ── Vectors ───────────────────────────────────────────────────────────

    @testset "head - Vector" begin
        v = collect(1:10)

        @test collect(head(v)) == [1, 2, 3, 4, 5, 6]       # default n=6
        @test collect(head(v, 3)) == [1, 2, 3]
        @test collect(head(v, 0)) == Int[]
        @test collect(head(v, 20)) == collect(1:10)          # clamped
        @test collect(head(v, -3)) == collect(1:7)           # drop last 3
        @test collect(head(v, -10)) == Int[]                 # drop all
        @test collect(head(v, -15)) == Int[]                 # drop more than length
    end

    @testset "tail - Vector" begin
        v = collect(1:10)

        @test collect(tail(v)) == [5, 6, 7, 8, 9, 10]       # default n=6
        @test collect(tail(v, 3)) == [8, 9, 10]
        @test collect(tail(v, 0)) == Int[]
        @test collect(tail(v, 20)) == collect(1:10)          # clamped
        @test collect(tail(v, -3)) == collect(4:10)          # drop first 3
        @test collect(tail(v, -10)) == Int[]                 # drop all
        @test collect(tail(v, -15)) == Int[]                 # drop more than length
    end

    # ── Matrices ──────────────────────────────────────────────────────────

    @testset "head - Matrix" begin
        M = reshape(1:12, 3, 4)   # 3×4

        @test collect(head(M, 2)) == [1 4 7 10; 2 5 8 11]              # first 2 rows
        @test collect(head(M, 2; dims=2)) == [1 4; 2 5; 3 6]           # first 2 cols
        @test collect(head(M, -1)) == [1 4 7 10; 2 5 8 11]             # drop last 1 row
        @test collect(head(M, -1; dims=2)) == [1 4 7; 2 5 8; 3 6 9]   # drop last 1 col
    end

    @testset "tail - Matrix" begin
        M = reshape(1:12, 3, 4)   # 3×4

        @test collect(tail(M, 2)) == [2 5 8 11; 3 6 9 12]              # last 2 rows
        @test collect(tail(M, 2; dims=2)) == [7 10; 8 11; 9 12]        # last 2 cols
        @test collect(tail(M, -1)) == [2 5 8 11; 3 6 9 12]             # drop first 1 row
        @test collect(tail(M, -1; dims=2)) == [4 7 10; 5 8 11; 6 9 12] # drop first 1 col
    end

    # ── Edge cases ────────────────────────────────────────────────────────

    @testset "edge cases" begin
        empty_v = Int[]
        @test collect(head(empty_v)) == Int[]
        @test collect(tail(empty_v)) == Int[]

        single = [42]
        @test collect(head(single)) == [42]
        @test collect(tail(single)) == [42]
        @test collect(head(single, 0)) == Int[]
        @test collect(tail(single, 0)) == Int[]

        # dims validation
        v = [1, 2, 3]
        @test_throws ArgumentError head(v; dims=2)
        @test_throws ArgumentError tail(v; dims=0)
    end

    # ── Returns views ─────────────────────────────────────────────────────

    @testset "returns views (not copies)" begin
        v = collect(1:10)
        h = head(v, 3)
        @test h isa SubArray
        h[1] = 99
        @test v[1] == 99     # mutated original

        v2 = collect(1:10)
        t = tail(v2, 3)
        @test t isa SubArray
        t[end] = 99
        @test v2[end] == 99  # mutated original
    end

    # ── NamedMatrix ───────────────────────────────────────────────────────

    @testset "head - NamedMatrix" begin
        data = Float64[1 2 3 4; 5 6 7 8; 9 10 11 12]   # 3×4
        nm = NamedMatrix(data, ["c1", "c2", "c3", "c4"])

        h = head(nm, 2)   # first 2 rows
        @test size(h.data) == (2, 4)
        @test h.data == [1 2 3 4; 5 6 7 8]
        @test h.colnames == ["c1", "c2", "c3", "c4"]

        h2 = head(nm, 2; dims=2)   # first 2 cols
        @test size(h2.data) == (3, 2)
        @test h2.data == [1 2; 5 6; 9 10]
        @test h2.colnames == ["c1", "c2"]

        # negative n
        h3 = head(nm, -1)   # drop last 1 row
        @test size(h3.data) == (2, 4)

        h4 = head(nm, -1; dims=2)   # drop last 1 col
        @test size(h4.data) == (3, 3)
        @test h4.colnames == ["c1", "c2", "c3"]
    end

    @testset "tail - NamedMatrix" begin
        data = Float64[1 2 3 4; 5 6 7 8; 9 10 11 12]   # 3×4
        nm = NamedMatrix(data, ["c1", "c2", "c3", "c4"])

        t = tail(nm, 2)   # last 2 rows
        @test size(t.data) == (2, 4)
        @test t.data == [5 6 7 8; 9 10 11 12]
        @test t.colnames == ["c1", "c2", "c3", "c4"]

        t2 = tail(nm, 2; dims=2)   # last 2 cols
        @test size(t2.data) == (3, 2)
        @test t2.data == [3 4; 7 8; 11 12]
        @test t2.colnames == ["c3", "c4"]

        # negative n
        t3 = tail(nm, -1)   # drop first 1 row
        @test size(t3.data) == (2, 4)

        t4 = tail(nm, -1; dims=2)   # drop first 1 col
        @test size(t4.data) == (3, 3)
        @test t4.colnames == ["c2", "c3", "c4"]
    end

    @testset "NamedMatrix dims validation" begin
        data = Float64[1 2; 3 4]
        nm = NamedMatrix(data, ["a", "b"])
        @test_throws ArgumentError head(nm; dims=3)
        @test_throws ArgumentError tail(nm; dims=0)
    end

    @testset "NamedMatrix zero-keep" begin
        data = Float64[1 2; 3 4]
        nm = NamedMatrix(data, ["a", "b"])
        h = head(nm, 0)
        @test size(h.data) == (0, 2)
        t = tail(nm, 0)
        @test size(t.data) == (0, 2)

        h2 = head(nm, 0; dims=2)
        @test size(h2.data) == (2, 0)
        t2 = tail(nm, 0; dims=2)
        @test size(t2.data) == (2, 0)
    end

    # ── Tables (NamedTuples of vectors) ───────────────────────────────────

    @testset "head - Table" begin
        tbl = (x = collect(1:10), y = collect(11:20), name = ["a","b","c","d","e","f","g","h","i","j"])

        h = head(tbl, 3)
        @test h.x == [1, 2, 3]
        @test h.y == [11, 12, 13]
        @test h.name == ["a", "b", "c"]

        # default n=6
        h6 = head(tbl)
        @test length(h6.x) == 6

        # clamped
        h20 = head(tbl, 20)
        @test length(h20.x) == 10

        # negative n: drop last 7
        hn = head(tbl, -7)
        @test hn.x == [1, 2, 3]

        # zero
        h0 = head(tbl, 0)
        @test length(h0.x) == 0
    end

    @testset "tail - Table" begin
        tbl = (x = collect(1:10), y = collect(11:20), name = ["a","b","c","d","e","f","g","h","i","j"])

        t = tail(tbl, 3)
        @test t.x == [8, 9, 10]
        @test t.y == [18, 19, 20]
        @test t.name == ["h", "i", "j"]

        # default n=6
        t6 = tail(tbl)
        @test length(t6.x) == 6
        @test t6.x == collect(5:10)

        # clamped
        t20 = tail(tbl, 20)
        @test length(t20.x) == 10

        # negative n: drop first 7
        tn = tail(tbl, -7)
        @test tn.x == [8, 9, 10]

        # zero
        t0 = tail(tbl, 0)
        @test length(t0.x) == 0
    end

    @testset "Table - empty" begin
        tbl = (x = Int[], y = Float64[])
        @test length(head(tbl).x) == 0
        @test length(tail(tbl).x) == 0
    end

    # ── GroupedTable ──────────────────────────────────────────────────────

    @testset "head - GroupedTable" begin
        tbl = (
            cat = ["a", "a", "a", "a", "b", "b", "b"],
            val = [1, 2, 3, 4, 10, 20, 30]
        )
        gt = groupby(tbl, [:cat])

        hgt = head(gt, 2)
        @test hgt isa GroupedTable
        @test length(hgt.keys) == 2  # still 2 groups

        # each group has at most 2 rows
        for idxs in hgt.indices
            @test length(idxs) <= 2
        end

        # check that a group with 4 rows got truncated to 2
        a_idx = findfirst(k -> k.cat == "a", hgt.keys)
        @test length(hgt.indices[a_idx]) == 2
        # first 2 rows of group "a" are the original first 2
        @test hgt.data.val[hgt.indices[a_idx]] == [1, 2]

        b_idx = findfirst(k -> k.cat == "b", hgt.keys)
        @test length(hgt.indices[b_idx]) == 2
        @test hgt.data.val[hgt.indices[b_idx]] == [10, 20]
    end

    @testset "tail - GroupedTable" begin
        tbl = (
            cat = ["a", "a", "a", "a", "b", "b", "b"],
            val = [1, 2, 3, 4, 10, 20, 30]
        )
        gt = groupby(tbl, [:cat])

        tgt = tail(gt, 2)
        @test tgt isa GroupedTable
        @test length(tgt.keys) == 2

        a_idx = findfirst(k -> k.cat == "a", tgt.keys)
        @test length(tgt.indices[a_idx]) == 2
        @test tgt.data.val[tgt.indices[a_idx]] == [3, 4]

        b_idx = findfirst(k -> k.cat == "b", tgt.keys)
        @test length(tgt.indices[b_idx]) == 2
        @test tgt.data.val[tgt.indices[b_idx]] == [20, 30]
    end

    @testset "GroupedTable - negative n" begin
        tbl = (cat = ["a", "a", "a", "b", "b"], val = [1, 2, 3, 10, 20])
        gt = groupby(tbl, [:cat])

        # head with negative: drop last 1 per group
        hgt = head(gt, -1)
        a_idx = findfirst(k -> k.cat == "a", hgt.keys)
        @test length(hgt.indices[a_idx]) == 2  # 3 - 1 = 2
        b_idx = findfirst(k -> k.cat == "b", hgt.keys)
        @test length(hgt.indices[b_idx]) == 1  # 2 - 1 = 1

        # tail with negative: drop first 1 per group
        tgt = tail(gt, -1)
        a_idx = findfirst(k -> k.cat == "a", tgt.keys)
        @test length(tgt.indices[a_idx]) == 2
        b_idx = findfirst(k -> k.cat == "b", tgt.keys)
        @test length(tgt.indices[b_idx]) == 1
    end

    @testset "GroupedTable - zero" begin
        tbl = (cat = ["a", "a", "b"], val = [1, 2, 3])
        gt = groupby(tbl, [:cat])

        hgt = head(gt, 0)
        for idxs in hgt.indices
            @test isempty(idxs)
        end

        tgt = tail(gt, 0)
        for idxs in tgt.indices
            @test isempty(idxs)
        end
    end

    # ── Generic dispatch: same function across types ──────────────────────

    @testset "single generic function" begin
        # head/tail from Durbyn top-level works on arrays AND tables
        @test collect(head([1,2,3,4,5], 2)) == [1, 2]
        @test head((x=[1,2,3,4,5],), 2).x == [1, 2]
    end
end
