using LinearAlgebra

X = rand(2, 4)
r = rand(size(X, 2))
x = rand(size(X, 1))

using .RangeSearch

dists = map(
    r -> norm(x .- r), eachcol(X)
)

tree = RangeTree(X, r; leaf_size = 2)

_perfect_find_in_range(X, R, x, r) = let d = map(
    (c, rr) -> norm(x .- c) - rr - r, eachcol(X), R
)

    findall(
        dd -> dd <= 0.0,
        d
    )

end

@show tree

@assert Set(find_in_range(tree, x, 0.12)) == Set(_perfect_find_in_range(X, r, x, 0.12))

@info "Performance tests"

X = rand(3, 1000000)
x = rand(size(X, 1))

perc = 1.0 / (size(X, 2) ^ (1.0 - 1.0 / size(X, 1)))
r = (perc * 1e-2) ^ (1.0 / length(x))

tree = RangeTree(X)

@show size(X)

@info "Tree building"

@time tree = RangeTree(X)
@time tree = RangeTree(X)
@time tree = RangeTree(X)
@time tree = RangeTree(X)
@time tree = RangeTree(X)

@info "Point finding - $r range (approx. $perc percent of points)"

@time find_in_range(tree, x, r)
@time find_in_range(tree, x, r)
@time find_in_range(tree, x, r)
@time find_in_range(tree, x, r)
@time find_in_range(tree, x, r)
