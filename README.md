# AdaptiveKDTrees.jl: Adaptive kNN and Range Search Queries in Julia

AdaptiveKDTrees.jl aims to complement some of the functionalities of NearestNeighbors.jl, namely:

* To allow for the progressive inclusion of new data points to the KDTree; and
* To allow for range searches with specific search radii not only for the query points, but to each point in the dataset.

## kNN Queries

An example:

```
using AdaptiveKDTrees.KNN

X = rand(3, 100)

tree = KDTree(X; leaf_size = 10)

ind, dist = nn(tree, rand(3))
inds, dists = knn(tree, rand(3), 10) # 10 element vectors
```

To add more points to the KD Tree:

```
add_point!(tree, rand(3)) # returns index
```

Some other functions:

```
@show npoints(tree)
# 101

@show tree
# KDTree with 101 points
```

## Range Search

Similarly:

```
using AdaptiveKDTrees.RangeSearch

X = rand(3, 100)

tree = RangeTree(X; leaf_size = 10)

add_point!(tree, rand(3))
```

One may also specify the radius from which each point is visible:

```
X = rand(3, 100)
R = rand(100)

tree = RangeTree(X, R)
```

For queries,

```
r = rand(Float64)
x = rand(3)

inds = find_in_range(
    tree, x, r
)
```

...returns a vector of points with distances to `x` such that

```
distance(X[:, i], x) < R[i] + r
```
