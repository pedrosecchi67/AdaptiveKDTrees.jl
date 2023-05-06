using LinearAlgebra

X = rand(2, 4)

x = rand(2)

using .KNN

tree = KDTree(X; leaf_size = 2,)
@show tree

dists = map(
    c -> norm(c .- x),
    eachcol(X),
)

_nn_perfect(X, x) = let d = map(
    c -> norm(x .- c),
    eachcol(X)
)

    d, i = findmin(d)

    (i, d)

end

_knn_perfect(X, x, k) = let d = map(
    c -> norm(x .- c),
    eachcol(X)
)

    asrt = sortperm(d)

    inds = asrt[1:k]

    (
        inds,
        d[inds]
    )

end

@assert all(nn(tree, x) .≈ _nn_perfect(X, x))

ni = add_point!(tree, x)

@assert nn(tree, x)[1] == ni

X = [X x]
@assert all(knn(tree, x, 2) .≈ _knn_perfect(X, x, 2))

@assert add_point!(tree, x) == 6

X = [X x]
@assert all(knn(tree, x, 2) .≈ _knn_perfect(X, x, 2))

@info "Performance test"

X = rand(3, 1000000)
@show size(X)

@info "Tree creation"

@time tree = KDTree(X)
@time tree = KDTree(X)
@time tree = KDTree(X)
@time tree = KDTree(X)
@time tree = KDTree(X)

@info "Evaluation of a single point"

x = rand(size(X, 1))

@time nn(tree, x)
@time nn(tree, x)
@time nn(tree, x)
@time nn(tree, x)
@time nn(tree, x)

@info "Find point closest to all in dataset"

@time map(pt -> nn(tree, pt, eps(Float64)), eachcol(X))
@time map(pt -> nn(tree, pt, eps(Float64)), eachcol(X))
@time map(pt -> nn(tree, pt, eps(Float64)), eachcol(X))
@time map(pt -> nn(tree, pt, eps(Float64)), eachcol(X))
@time map(pt -> nn(tree, pt, eps(Float64)), eachcol(X))