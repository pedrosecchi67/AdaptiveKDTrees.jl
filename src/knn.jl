module KNN

    using LinearAlgebra
    using Statistics

    export KDTree, add_point!, nn, knn, npoints

    """
    Type to include both leaf and node
    """
    abstract type AbstractKD end

    """
    Struct to define a KDLeaf
    """
    mutable struct KDLeaf <: AbstractKD
        indices::AbstractVector
        points::AbstractMatrix
        max_size::Int64
    end

    """
    Build an empty KDLeaf
    """
    KDLeaf(
        ;
        max_size::Int64 = 10,
        ndims::Int64,
    ) = KDLeaf(
        Int64[],
        Matrix{Float64}(undef, ndims, 0),
        max_size,
    )

    """
    Struct to define a KDNode
    """
    mutable struct KDNode <: AbstractKD
        left::AbstractKD
        right::AbstractKD
        dimension::Int64
        limit::Real
    end

    """
    Evaluate restriction for a KDNode
    """
    (node::KDNode)(point::AbstractVector) = point[node.dimension] - node.limit

    """
    If size is beyond limits, split a KDLeaf into a KDNode and
    two adjacent leaves. Is recursive
    """
    function split(
        leaf::KDLeaf,
    )

        # is there a need to split?
        if length(leaf.indices) <= leaf.max_size
            return leaf
        end

        # find dimension for the limit
        _, dimension = findmax(
            r -> maximum(r) - minimum(r),
            eachrow(leaf.points)
        )

        # find limit (median)
        row = view(leaf.points, dimension, :)
        limit = median(
            row
        )

        isleft = @. row <= limit
        isright = @. !isleft

        # if there is no viable split (all points in a single leaf), just return the leaf
        if let split_number = sum(isleft)
            (split_number == length(isleft) || split_number == 0)
        end
            return leaf
        end

        KDNode(
            split(
                KDLeaf(
                    leaf.indices[isleft],
                    leaf.points[:, isleft],
                    leaf.max_size,
                )
            ),
            split(
                KDLeaf(
                    leaf.indices[isright],
                    leaf.points[:, isright],
                    leaf.max_size,
                )
            ),
            dimension,
            limit
        )

    end

    """
    ```
        mutable struct KDTree
            n_points::Int64
            head::AbstractKD

            KDTree(
                points::AbstractMatrix;
                leaf_size::Int = 10,
            ) = new(
                size(points, 2),
                split(
                    KDLeaf(
                        collect(1:size(points, 2)),
                        points,
                        leaf_size,
                    )
                ),
            )
        end
    ```

    Struct defining a KDTree
    """
    mutable struct KDTree
        n_points::Int64
        head::AbstractKD

        KDTree(
            points::AbstractMatrix;
            leaf_size::Int = 10,
        ) = new(
            size(points, 2),
            split(
                KDLeaf(
                    collect(1:size(points, 2)),
                    points,
                    leaf_size,
                )
            ),
        )
    end

    """
    ```
        KDTree(
            dimensionality::Int;
            leaf_size::Int = 10,
        ) = KDTree(
            Matrix{Float64}(undef, dimensionality, 0);
            leaf_size = leaf_size,
        )
    ```

    Create an empty KD tree with given dimensionality
    """
    KDTree(
        dimensionality::Int;
        leaf_size::Int = 10,
    ) = KDTree(
        Matrix{Float64}(undef, dimensionality, 0);
        leaf_size = leaf_size,
    )

    """
    Show a KDTree
    """
    Base.show(io::IO, tree::KDTree) = print(io, "KDTree with $(tree.n_points) points")

    _find_leaf(leaf::KDLeaf, point::AbstractVector) = leaf
    _find_leaf(node::KDNode, point::AbstractVector) = (
        node(point) > 0.0 ?
        _find_leaf(node.right, point) : _find_leaf(node.left, point)
    )

    """
    ```
        find_leaf(tree::KDTree, point::AbstractVector)
    ```

    Find leaf of the current point in the tree
    """
    find_leaf(tree::KDTree, point::AbstractVector) = _find_leaf(tree, point)

    #=
    """
    Merge distance vectors for search
    """
    function merge_distvecs(
        k::Int,
        inds::AbstractVector, dists::AbstractVector,
        inds2::AbstractVector, dists2::AbstractVector,
    )

        # if one of the sets is empty
        if length(inds) == 0
            return (inds2, dists2)
        end
        if length(inds2) == 0
            return (inds, dists)
        end

        # if all are smaller
        if dists[end] < dists2[1]
            return (inds, dists)
        end
        if dists2[end] < dists[1]
            return (inds2, dists2)
        end

        # merge and sort
        inds = [inds; inds2]
        dists = [dists; dists2]

        asrt = sortperm(dists)
        if length(asrt) > k
            asrt = asrt[1:k]
        end

        inds = inds[asrt]
        dists = dists[asrt]

        (inds, dists)

    end
    =#

    function _nn(
        leaf::KDLeaf, point::AbstractVector, ind::Int, dist::Real, min_radius::Real = 0.0;
        exclusion = nothing,
    )

        points = leaf.points
        indices = leaf.indices

        if !isnothing(exclusion)
            isval = map(
                i -> !(i in exclusion),
                indices
            )

            indices = indices[isval]
            points = points[:, isval]
        end

        if length(indices) == 0
            return (ind, dist)
        end

        dmin, indmin = (
            min_radius == 0.0 ?
            findmin(
                c -> norm(c .- point),
                eachcol(points)
            ) :
            findmin(
                c -> (
                    let L = norm(c .- point)
                        (
                            L <= min_radius ?
                            Inf :
                            L
                        )
                    end
                ),
                eachcol(points)
            )
        )

        if dmin < dist
            return (indices[indmin], dmin)
        end

        (ind, dist)

    end

    function _nn(
        node::KDNode, point::AbstractVector, ind::Int, dist::Real, min_radius::Real = 0.0;
        exclusion = nothing,
    )

        b1, b2 = (
            node(point) > 0.0 ?
            (node.right, node.left) :
            (node.left, node.right)
        )

        ind, dist = _nn(b1, point, ind, dist, min_radius; exclusion = exclusion,)

        if abs(node(point)) <= dist
            ind, dist = _nn(b2, point, ind, dist, min_radius; exclusion = exclusion,)
        end

        (ind, dist)

    end

    """
    ```
        function nn(tree::KDTree, point::AbstractVector, min_radius::Real = 0.0,)
    ```

    Find the nearest neighbor in a tree
    """
    function nn(tree::KDTree, point::AbstractVector, min_radius::Real = 0.0; exclusion = nothing,)

        ind = 0
        dist = Inf

        _nn(tree.head, point, ind, dist, min_radius; exclusion = exclusion,)

    end

    """
    ```
        function knn(tree::KDTree, point::AbstractVector, k::Int, min_radius::Real = 0.0,)
    ```

    Return vectors of indices and distances to the k nearest neighbors to a point
    """
    function knn(tree::KDTree, point::AbstractVector, k::Int, min_radius::Real = 0.0,)

        inds = zeros(Int64, k)
        dists = Vector{eltype(point)}(undef, k)

        for j = 1:k
            i, d = nn(tree, point, min_radius; exclusion = inds,)

            inds[j] = i
            dists[j] = d
        end

        (inds, dists)

    end

    function _add_point!(
        leaf::KDLeaf,
        point::AbstractVector,
        n::Int
    )

        leaf.points = [leaf.points point]
        leaf.indices = [leaf.indices; n]

        split(leaf)

    end

    function _add_point!(
        node::KDNode,
        point::AbstractVector,
        n::Int
    )

        if node(point) > 0.0
            node.right = _add_point!(node.right, point, n)
        else
            node.left = _add_point!(node.left, point, n)
        end

        node

    end

    """
    ```
        function add_point!(tree::KDTree, point::AbstractVector)
    ```

    Add new point to tree. Returns index of the new point
    """
    function add_point!(tree::KDTree, point::AbstractVector)

        tree.n_points += 1

        tree.head = _add_point!(tree.head, point, tree.n_points)

        tree.n_points

    end

    
    """
    ```
        npoints(tree::KDTree) 
    ```

    Get number of points in KDTree
    """
    @inline npoints(tree::KDTree) = tree.n_points

end

#=
using LinearAlgebra

using Random: seed!
seed!(42)

X = rand(2, 4)
@show X

x = rand(2)
@show x

using .KNN

tree = KDTree(X; leaf_size = 2,)
@show tree

dists = map(
    c -> norm(c .- x),
    eachcol(X),
)

@show dists

@show nn(tree, x)

@show add_point!(tree, x)

@show nn(tree, x)

@show knn(tree, x, 2)

@show add_point!(tree, x)

@show knn(tree, x, 2)

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
=#
