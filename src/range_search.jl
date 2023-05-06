module RangeSearch

    using LinearAlgebra
    using Statistics

    export RangeTree, add_point!, find_in_range

    """
    Struct defining a range tree leaf
    """
    mutable struct RangeTreeLeaf
        max_size::Int
        points::AbstractMatrix
        indices::Vector{Int64}
        radii::AbstractVector
        maximum_radius::Real
    end

    """
    Struct definining a range tree node
    """
    mutable struct RangeTreeNode
        dimension::Int
        limit::Real
        maximum_radius::Real
        left::Union{RangeTreeLeaf, RangeTreeNode}
        right::Union{RangeTreeLeaf, RangeTreeNode}
    end

    """
    Split a range tree leaf if it violates its max size
    """
    function split(
        leaf::RangeTreeLeaf,
    )

        if size(leaf.points, 2) <= leaf.max_size
            return leaf
        end

        _, dimension = findmax(
            r -> maximum(r) - minimum(r),
            eachrow(leaf.points)
        )

        x = view(leaf.points, dimension, :)

        limit = median(x)

        isleft = @. x <= limit
        isright = @. !isleft

        if all(isleft) || all(isright)
            return leaf
        end

        left = RangeTreeLeaf(
            leaf.max_size,
            leaf.points[:, isleft],
            leaf.indices[isleft],
            let r = leaf.radii[isleft]
                rmax = maximum(r)

                (r, rmax)
            end...
        )
        right = RangeTreeLeaf(
            leaf.max_size,
            leaf.points[:, isright],
            leaf.indices[isright],
            let r = leaf.radii[isright]
                rmax = maximum(r)

                (r, rmax)
            end...
        )

        RangeTreeNode(
            dimension, limit, max(
                left.maximum_radius, right.maximum_radius
            ),
            split(left), split(right)
        )

    end

    #=
    """
    Join a node into a leaf if size requirements are met
    """
    function join_leaves(
        node::RangeTreeNode,
    )

        if isa(
            node.left, RangeTreeNode
        ) || isa(
            node.right, RangeTreeNode
        )
            return node
        end

        max_size = min(
            node.left.max_size,
            node.right.max_size
        )

        if size(node.left.points, 2) + size(node.right.points, 2) > max_size
            return node
        end

        RangeTreeLeaf()

    end
    =#

    """
    Struct to hold a range tree
    """
    mutable struct RangeTree
        n_points::Int
        head::Union{RangeTreeLeaf, RangeTreeNode}
    end

    """
    Build a range tree from a set of points
    """
    RangeTree(
        X::AbstractMatrix,
        radii = 0.0;
        leaf_size::Int = 10,
    ) = RangeTree(
        size(X, 2),
        split(
            RangeTreeLeaf(
                leaf_size,
                X,
                collect(1:size(X, 2)),
                let r = (
                    isa(radii, AbstractVector) ?
                    radii :
                    fill(radii, size(X, 2))
                )
                    (
                        r, (
                            isempty(r) ?
                            0.0 :
                            maximum(r)
                        )
                    )
                end...
            )
        ),
    )

    function _find_in_range(
        leaf::RangeTreeLeaf,
        x::AbstractVector,
        r::Real = 0.0,
    )

        leaf.indices[
            map(
                (rr, p) -> norm(p .- x) <= rr + r,
                leaf.radii, eachcol(leaf.points),
            )
        ]

    end

    function _find_in_range(
        node::RangeTreeNode,
        x::AbstractVector,
        r::Real = 0.0,
    )

        g = (
            x[node.dimension] - node.limit
        )

        b1, b2 = node.left, node.right
        if g > 0.0
            b2, b1 = b1, b2
        end

        if abs(g) <= r + b2.maximum_radius
            return [
                _find_in_range(
                    b1, x, r
                );
                _find_in_range(
                    b2, x, r
                )
            ]
        end

        _find_in_range(b1, x, r)

    end

    """
    ```
        function find_in_range(
            tree::RangeTree,
            x::AbstractVector,
            r::Real = 0.0,
        )
    ```

    Find indices of spheres within given range
    """
    function find_in_range(
        tree::RangeTree,
        x::AbstractVector,
        r::Real = 0.0,
    )

        _find_in_range(tree.head, x, r)

    end

    function add_point!(
        leaf::RangeTreeLeaf,
        x::AbstractVector,
        r::Real,
        n::Int,
    ) 

        if r > leaf.maximum_radius
            leaf.maximum_radius = r
        end

        leaf.points = [leaf.points x]
        push!(leaf.radii, r)
        push!(leaf.indices, n)

        split(leaf)

    end

    function add_point!(
        node::RangeTreeNode,
        x::AbstractVector,
        r::Real,
        n::Int,
    ) 

        if r > node.maximum_radius
            node.maximum_radius = r
        end

        if x[node.dimension] > node.limit
            node.right = add_point!(
                node.right, x, r, n
            )
        else
            node.left = add_point!(
                node.left, x, r, n
            )
        end

        node

    end

    """
    Add point to tree
    """
    function add_point!(
        tree::RangeTree,
        x::AbstractVector,
        r::Real = 0.0,
    ) 

        tree.n_points += 1

        tree.head = add_point!(
            tree.head,
            x, r, tree.n_points
        )

    end

    """
    ```
        RangeTree(ndim::Int; leaf_size::Int = 10,)
    ```

    Initialize an empty range tree with given dimensionality
    """
    RangeTree(ndim::Int; leaf_size::Int = 10,) = RangeTree(
        Matrix{Float64}(undef, ndim, 0);
        leaf_size = leaf_size,
    )

    """
    Show a range tree
    """
    Base.show(io::IO, tree::RangeTree) = print(io, "RangeTree with $(tree.n_points) points")

end

#=
using Random: seed!
seed!(42)

X = rand(2, 4)
r = rand(size(X, 2))
x = rand(size(X, 1))

dists = map(
    r -> norm(x .- r), eachcol(X)
)

tree = RangeTree(X, r; leaf_size = 2)

@show X
@show r
@show x
@show dists

@show tree

@show find_in_range(tree, x, 0.12)

@show add_point!(tree, x,)

@show find_in_range(tree, x, 0.12)

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
=#
