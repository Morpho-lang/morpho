/** @file selection.c
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
    
    enum {
        SELECT_ALL, SELECT_NONE, SELECT_SOME
    } mode; /** What is selected? */
    
    unsigned int ngrades; /** Number of grades */
    dictionary selected[]; /** Selections */
} objectselection;

/** Tests whether an object is a selection */
#define MORPHO_ISSELECTION(val) object_istype(val, OBJECT_SELECTION)

/** Gets the object as a selection */
#define MORPHO_GETSELECTION(val)   ((objectselection *) MORPHO_GETOBJECT(val))

/** Creates an empty selection object */
objectselection *object_newselection(objectmesh *mesh);

/* -------------------------------------------------------
 * Selection class
 * ------------------------------------------------------- */

#define SELECTION_CLASSNAME "Selection"
#define SELECTION_ISSELECTEDMETHOD "isselected"
#define SELECTION_IDLISTFORGRADEMETHOD "idlistforgrade"
#define SELECTION_ADDGRADEMETHOD "addgrade"
#define SELECTION_REMOVEGRADEMETHOD "removegrade"

#define SELECTION_COMPLEMENTMETHOD "complement"

#define SELECTION_BOUNDARYOPTION "boundary"
#define SELECTION_PARTIALSOPTION "partials"

#define SELECTION_NOMESH                     "SlNoMsh"
#define SELECTION_NOMESH_MSG                 "Selection requires a Mesh object as an argument."

#define SELECTION_ISSLCTDARG                 "SlIsSlArg"
#define SELECTION_ISSLCTDARG_MSG             "Selection.isselected requires a grade and element id as arguments."

#define SELECTION_GRADEARG                   "SlGrdArg"
#define SELECTION_GRADEARG_MSG               "Method requires a grade as the argument."

#define SELECTION_STARG                      "SlStArg"
#define SELECTION_STARG_MSG                  "Selection set methods require a selection as the argument."

#define SELECTION_BND                        "SlBnd"
#define SELECTION_BND_MSG                    "Mesh has no boundary elements."

void selection_clear(objectselection *s);

bool selection_isselected(objectselection *sel, grade g, elementid id);
void selection_initialize(void);

#endif

#endif /* selection_h */
