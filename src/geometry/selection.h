/** @file selection.h
 *  @author T J Atherton
 *
 *  @brief Selections
 */

#ifndef selection_h
#define selection_h

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "mesh.h"

/* -------------------------------------------------------
 * Selection objects
 * ------------------------------------------------------- */

extern objecttype objectselectiontype;
#define OBJECT_SELECTION objectselectiontype

typedef struct {
    object obj;
    objectmesh *mesh; /** The mesh the selection is referring to */
    
    unsigned int ngrades; /** Number of grades */
    dictionary selected[]; /** Selections */
} objectselection;

/** Tests whether an object is a selection */
#define MORPHO_ISSELECTION(val) object_istype(val, OBJECT_SELECTION)

/** Gets the object as a selection */
#define MORPHO_GETSELECTION(val)   ((objectselection *) MORPHO_GETOBJECT(val))

/* -------------------------------------------------------
 * Selection class
 * ------------------------------------------------------- */

#define SELECTION_CLASSNAME "Selection"
#define SELECTION_ISSELECTEDMETHOD "isselected"
#define SELECTION_IDLISTFORGRADEMETHOD "idlistforgrade"
#define SELECTION_ADDGRADEMETHOD "addgrade"
#define SELECTION_REMOVEGRADEMETHOD "removegrade"
#define SELECTION_MESHMETHOD    "mesh"

#define SELECTION_BOUNDARYOPTION "boundary"
#define SELECTION_PARTIALSOPTION "partials"

#define SELECTION_BND                        "SlBnd"
#define SELECTION_BND_MSG                    "Mesh has no boundary elements."

#define SELECTION_FLDMSH                     "SlFldMsh"
#define SELECTION_FLDMSH_MSG                 "Field must refer to the same Mesh as the Selection."

#define SELECTION_MSH                        "SlMsh"
#define SELECTION_MSH_MSG                    "Selections must refer to the same Mesh."

bool selection_isselected(objectselection *sel, grade g, elementid id);
unsigned int selection_count(objectselection *sel, grade g);
void selection_initialize(void);

#endif

#endif /* selection_h */
