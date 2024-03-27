[comment]: # (KDTree module help)
[version]: # (0.5)

# KDTree
[tagkdtree]: # (kdtree)

The `kdtree` module implements a k-dimensional tree, a space partitioning data structure that can be used to accelerate computational geometry calculations.

To use the module, first import it:

    import kdtree

To create a tree from a list of points:

    var pts = []
    for (i in 0...100) pts.append(Matrix([random(), random(), random()]))
    var tree=KDTree(pts)

Add further points:

    tree.insert(Matrix([0,0,0]))

Test whether a given point is present in the tree:

    tree.ismember(Matrix([1,0,0]))

Find all points within a given bounding box:

    var pts = tree.search([[-1,1], [-1,1], [-1,1]])
    for (x in pts) print x.location

Find the nearest point to a given point:

    var pt = tree.nearest(Matrix([0.1, 0.1, 0.5]))
    print pt.location

[showsubtopics]: # (subtopics)

## Insert
[taginsert]: # (insert)

Inserts a new point into a k-d tree. Returns a KDTreeNode object.

    var node = tree.insert(Matrix([0,0,0]))

Note that, for performance reasons, if the set of points is known ahead of time, it is generally better to build the tree using the constructor function KDTree rather than one-by-one with insert.

## Ismember
[tagismember]: # (ismember)

Checks if a point is a member of a k-d tree. Returns `true` or `false`.

    print tree.ismember(Matrix([0,0,0]))

## Nearest
[tagnearest]: # (nearest)

Finds the point in a k-d tree nearest to a point of interest. Returns a KDTreeNode object.

    var pt = tree.nearest(Matrix([0.1, 0.1, 0.5]))

To get the location of this nearest point, access the location property:

    print pt.location

## Search
[tagsearch]: # (search)

Finds all points in a k-d tree that lie within a cuboidal bounding box. Returns a list of KDTreeNode objects.

Find and display all points that lie in a cuboid 0<=x<=1, 0<=y<=2, 1<=z<=2:

    var result = tree.search([[0,1], [0,2], [1,2]])
    for (x in result) print x.location

## KDTreeNode
[tagkdtreenode]: # (kdtreenode)

An object corresponding to a single node in a k-d tree. To get the location of the node, access the `location` property:

    print node.location
