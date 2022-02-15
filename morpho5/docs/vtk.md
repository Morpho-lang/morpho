[comment]: # (Morpho vtk module help file)
[version]: # (0.5)

# VTK
[tagvtk]: # (vtk)

The vtk module contains classes to allow I/O of meshes and fields using
the VTK Legacy Format.

[showsubtopics]: # (subtopics)

## VTKExporter
[tagvtkexporter]: # (VTKExporter)

This class can be used to export the mesh and field(s) at a given state
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

** Note that this currently only supports scalar or vector (column
matrix) fields that live on the vertices ( shape `[1,0,0]`). Support for
tensorial fields and fields on cells coming soon.
