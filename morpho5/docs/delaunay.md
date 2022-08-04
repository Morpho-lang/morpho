[comment]: # (Delaunay module help)
[version]: # (0.5)

# Delaunay
[tagdelaunay]: # (delaunay)

The `delaunay` module creates Delaunay triangulations from point clouds. It is dimensionally independent, so generates tetrahedra in 3D and higher order simplices beyond.

To use the module, first import it:

    import delaunay

To create a Delaunary triangulation from a list of points:

    var pts = []
    for (i in 0...100) pts.append(Matrix([random(), random()]))
    var del=Delaunay(pts)
    print del.triangulate()

The module also provides `DelaunayMesh` to directly create meshes from Delaunay triangulations.

[showsubtopics]: # (subtopics)

## Triangulate
[tagtriangulate]: # (triangulate)

The `triangulate` method performs the delaunay triangulation. To use it, first construct a `Delaunay` object with the point cloud of interest: 

    var del=Delaunay(pts)

Then call `triangulate`:

    var tri = del.triangulate()

This returns a list of triangles `[ [i, j, k], ... ]`.

## Circumsphere
[tagcircumsphere]: # (circumsphere)

The `Circumsphere` class calculates the circumsphere of a set of points, i.e. a sphere such that all the points are on the surface of the sphere. It is used internally by the `delaunay` module.

Create a `Circumsphere` from a list of points and a triangle specified by indices into that list:

    var sph = Circumsphere(pts, [i,j,k]) 

Test if an arbitrary point is inside the `Circumsphere` or not: 

    print sph.pointinsphere(pt)
