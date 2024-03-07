[comment]: # (Morpho vtk module help file)
[version]: # (0.5)

# VTK
[tagvtk]: # (vtk)

The vtk module contains classes to allow I/O of meshes and fields using
the VTK Legacy Format. Note that this currently only supports scalar or 2D/3D vector (column matrix) fields that live on the vertices ( shape `[1,0,0]`). Support for
tensorial fields and fields on cells coming soon.

[showsubtopics]: # (subtopics)

## VTKExporter
[tagvtkexporter]: # (VTKExporter)

This class can be used to export the field(s) and/or a mesh at a given state
to a single .vtk file. To use it, import the `vtk` module:

    import vtk

Initialize the `VTKExporter`

    var vtkE = VTKExporter(obj)

where `obj` can either be

* A `Mesh` object: This prepares the Mesh for exporting.
* A `Field` object: This prepares both the Field and the Mesh associated
  with it for exporting.

Use the `export` method to export to a VTK file. 

    vtkE.export("output.vtk")
 
Optionally, use the `addfield` method to add one or more fields before
exporting:

    vtkE.addfield(f, fieldname="f")

where,

* `f` is the field object to be exported
* `fieldname` is an optional argument that assigns a name to the field
  in the VTK file. This name is required to be a character
  string without embedded whitespace. If not provided, the name would be
  either "scalars" or "vectors" depending on the field type**. 

** Note that this currently only supports scalar or 2D/3D vector (column
matrix) fields that live on the vertices ( shape `[1,0,0]`). Support for
tensorial fields and fields on cells coming soon.

Minimal example:

    import vtk
    import meshtools

    var m1 = LineMesh(fn (t) [t,0,0], -1..1:2)

    var vtkE = VTKExporter(m1) // Export just the mesh 
    
    vtkE.export("mesh.vtk")

    var f1 = Field(m1, fn(x,y,z) x)

    var g1 = Field(m1, fn(x,y,z) Matrix([x,2*x,3*x]))

    vtkE = VTKExporter(f1, fieldname="f") // Export fields

    vtkE.addfield(g1, fieldname="g")

    vtkE.export("data.vtk")

## VTKImporter
[tagvtkimporter]: # (VTKImporter)

This class can be used to import the field(s) and/or the  mesh at a
given state from a single .vtk file. To use it, import the `vtk` module:

    import vtk

Initialize the `VTKImporter` with the filename

    var vtkI = VTKImporter("output.vtk")

Use the `mesh` method to get the mesh:

    var mesh = vtkI.mesh()

Use the `field` method to get the field:

    var f = vtkI.field(fieldname)

Use the `fieldlist` method to get the list of the names of the fields contained in the file:

    print vtkI.fieldlist()

Use the `containsfield` method to check whether the file contains a field by a given `fieldname`:

    if (tkI.containsfield(fieldname)) {
        ... 
    }

where `fieldname` is the name assigned to the field in the .vtk file

Minimal example:

    import vtk
    import meshtools 

    var vtkI = VTKImporter("data.vtk")

    var m = vtkI.mesh()

    var f = vtkI.field("f")

    var g = vtkI.field("g")

