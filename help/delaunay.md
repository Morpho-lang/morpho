[comment]: # (Delaunay module help)
[version]: # (0.5)

# Delaunay
[tagdelaunay]: # (delaunay)

The `delaunay` module creates Delaunay triangulations from point clouds. It is dimensionally independent, so generates tetrahedra in 3D and higher order simplices beyond.

To use the module, first import it:

    import delaunay

To create a Delaunay triangulation from a list of points:

    var pts = []
    for (i in 0...100) pts.append(Matrix([random(), random()]))
    var del=Delaunay(pts)
    print del.triangulate()

To create a `Mesh` object directly from a Delaunay triangulation, use `DelaunayMesh` from the `meshtools` module:

    import meshtools
    var m = DelaunayMesh(pts)

[showsubtopics]: # (subtopics)

## Triangulate
[tagtriangulate]: # (triangulate)

The `triangulate` method performs the delaunay triangulation. To use it, first construct a `Delaunay` object with the point cloud of interest: 

    var del=Delaunay(pts)

Then call `triangulate`:

    var tri = del.triangulate()

This returns a list of triangles `[ [i, j, k], ... ]`.

The supersimplex that encloses the cloud can be enlarged by setting `del.sscale` (default 1) before calling `triangulate`.

## Circumsphere
[tagcircumsphere]: # (circumsphere)

The `Circumsphere` class calculates the circumsphere of a set of points, i.e. a sphere such that all the points are on the surface of the sphere. It is used internally by the `delaunay` module.

Create a `Circumsphere` from a list of points and a triangle specified by indices into that list:

    var sph = Circumsphere(pts, [i,j,k]) 

Test if an arbitrary point is inside the `Circumsphere` or not: 

    print sph.pointinsphere(pt)
