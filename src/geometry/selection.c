/** @file selection.c
 *  @author T J Atherton
 *
 *  @brief Selections
 */

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "selection.h"
#include "morpho.h"
#include "classes.h"
#include "field.h"

static void selection_clear(objectselection *s);

/* **********************************************************************
 * Selection object definitions
 * ********************************************************************** */

objecttype objectselectiontype;

/** Selection object definitions */
void objectselection_printfn(object *obj, void *v) {
    morpho_printf(v, "<Selection>");
}

void objectselection_markfn(object *obj, void *v) {
    objectselection *s = (objectselection *) obj;
    morpho_markobject(v, (object *) s->mesh);
}

void objectselection_freefn(object *obj) {
    objectselection *s = (objectselection *) obj;
    selection_clear(s);
}

size_t objectselection_sizefn(object *obj) {
    return sizeof(objectselection)+sizeof(dictionary)*((objectselection *) obj)->ngrades;
}

objecttypedefn objectselectiondefn = {
    .printfn=objectselection_printfn,
    .markfn=objectselection_markfn,
    .freefn=objectselection_freefn,
    .sizefn=objectselection_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

/* **********************************************************************
 * Selection object constructor
 * ********************************************************************** */

/** Create a new empty selection object */
static objectselection *object_newselection(objectmesh *mesh) {
    unsigned int ngrades = mesh->dim+1;
    objectselection *new=(objectselection *) object_new(sizeof(objectselection)+sizeof(dictionary)*ngrades, OBJECT_SELECTION);
    
    if (new) {
        new->mesh=mesh;
        new->ngrades=ngrades;
        for (unsigned int i=0; i<ngrades; i++) dictionary_init(&new->selected[i]);
    }
    
    return new;
}

/** Clones a selection */
objectselection *selection_clone(objectselection *sel) {
    objectselection *new=object_newselection(sel->mesh);
    
    if (new) {
        for (unsigned int i=0; i<sel->ngrades; i++) dictionary_copy(&sel->selected[i], &new->selected[i]);
    }
    
    return new;
}

/** Clears all data structures associated with a selection */
static void selection_clear(objectselection *s) {
    for (grade i=0; i<s->ngrades; i++) {
        dictionary_clear(&s->selected[i]);
    }
}

/** Removes a grade from a selection */
void selection_removegrade(objectselection *sel, grade g) {
    dictionary_clear(&sel->selected[g]);
}

/** Selects an element */
bool selection_selectelement(objectselection *sel, grade g, elementid id) {
    return dictionary_insert(&sel->selected[g], MORPHO_INTEGER(id), MORPHO_NIL);
}

/** Attempts to change the grade of a selection by raising
 * @param[in] sel - selection to change
 * @param[in] g - grade to add
 * @param[in] includepartials - whether to include partially selected elements or not  */
bool selection_addgraderaise(objectselection *sel, grade g, bool includepartials) {
    // Get the corresponding grade from the mesh
    objectsparse *conn=mesh_getconnectivityelement(sel->mesh, 0, g);
    if (!conn) return false;
    
    int nentries, *entries=NULL;
    
    for (elementid id=0; id<conn->ccs.ncols; id++) {
        if (mesh_getconnectivity(conn, id, &nentries, &entries)) {
            int k=0;
            for (int j=0; j<nentries; j++) {
                if (dictionary_get(&sel->selected[0], MORPHO_INTEGER(entries[j]), NULL)) k++;
            }
            if ((includepartials && k>0) || (k==nentries)) {
                dictionary_insert(&sel->selected[g], MORPHO_INTEGER(id), MORPHO_NIL);
            }
        }
    }
    
    return true;
}

/** Attempts to change the grade of a selection by lowering
 * @param[in] sel - selection to change
 * @param[in] g - grade to add */
bool selection_addgradelower(objectselection *sel, grade g) {
    for (grade i=sel->ngrades-1; i>g; i--) { // Loop over grades higher than g
        objectsparse *conn = mesh_addconnectivityelement(sel->mesh, g, i);
        if (!conn) continue;
        
        // Look through the selected elements in grade i
        for (unsigned int k=0; k<sel->selected[i].capacity; k++) {
            value el = sel->selected[i].contents[k].key;
            if (MORPHO_ISINTEGER(el)) {
                int nentries, *entries;
                elementid id = MORPHO_GETINTEGERVALUE(el);
                 
                // Get the element ids
                if (mesh_getconnectivity(conn, id, &nentries, &entries)) {
                    for (int j=0; j<nentries; j++) {
                        dictionary_insert(&sel->selected[g], MORPHO_INTEGER(entries[j]), MORPHO_NIL);
                    }
                }
            }
        }
    }
    
    return true;
}

/** Selects an element
 * @param v - the virtual machine to use
 * @param sel - selection object
 * @param fn - function to call */
void selection_selectwithfunction(vm *v, objectselection *sel, value fn) {
    if (!sel->mesh || !sel->mesh->vert) UNREACHABLE("Selection on mesh with invalid vertex structure.");
    
    objectmatrix *vert=sel->mesh->vert;
    int nv = vert->ncols;
    
    value ret=MORPHO_NIL; // Return value
    value coords[sel->mesh->dim+1]; // Vertex coords
    
    for (elementid i=0; i<nv; i++) {
        if (mesh_getvertexcoordinatesasvalues(sel->mesh, i, coords)) {
            if (!morpho_call(v, fn, sel->mesh->dim, coords, &ret)) break;
            if (MORPHO_ISTRUE(ret)) selection_selectelement(sel, MESH_GRADE_VERTEX, i);
        }
    }
}

/** Selects elements by mapping a function over a matrix.
 * @param v - the virtual machine to use
 * @param sel - selection object
 * @param fn - function to call
 * @param matrix - matrix to map over */
void selection_selectwithmatrix(vm *v, objectselection *sel, value fn, objectmatrix *matrix) {
    if (!sel->mesh || !sel->mesh->vert) UNREACHABLE("Selection on mesh with invalid vertex structure.");
    
    objectmatrix *vert=sel->mesh->vert;
    int nv = vert->ncols;

    if (matrix->ncols!=nv) {
        morpho_runtimeerror(v, LINALG_INCOMPATIBLEMATRICES);
        return;
    }
    
    int nargs = matrix->nrows; // Number of args to pass to function
    value args[nargs]; // Vertex coords
    double *x; // Matrix column
    value ret=MORPHO_NIL; // Return value
    
    for (elementid i=0; i<nv; i++) {
        if (matrix_getcolumnptr(matrix, i, &x)==LINALGERR_OK) {
            for (unsigned int j=0; j<(unsigned int) nargs; j++) args[j]=MORPHO_FLOAT(x[j]);
        }
        
        if (!morpho_call(v, fn, nargs, args, &ret)) break;
        if (MORPHO_ISTRUE(ret)) selection_selectelement(sel, MESH_GRADE_VERTEX, i);
    }
}

/** Selects elements by mapping a function over a Field.
 * Iterates every nonempty grade; `fn` receives each stored value. */
void selection_selectwithfield(vm *v, objectselection *sel, value fn, objectfield *field) {
    if (!sel->mesh || field->mesh!=sel->mesh) {
        morpho_runtimeerror(v, SELECTION_FLDMSH);
        return;
    }

    value arg, ret=MORPHO_NIL;

    for (grade g=0; g<(grade) field->ngrades; g++) {
        unsigned int dof=field_dofforgrade(field, g);
        if (dof==0) continue;

        elementid nel=mesh_nelementsforgrade(sel->mesh, g);
        for (elementid id=0; id<nel; id++) {
            for (unsigned int indx=0; indx<dof; indx++) {
                if (!field_getelement(field, g, id, indx, &arg)) continue;
                if (!morpho_call(v, fn, 1, &arg, &ret)) return;
                if (MORPHO_ISTRUE(ret)) {
                    selection_selectelement(sel, g, id);
                    break;
                }
            }
        }
    }
}

/** Selects boundary elements
 * @param v - the virtual machine to use
 * @param sel - selection object */
void selection_selectboundary(vm *v, objectselection *sel) {
    grade max = mesh_maxgrade(sel->mesh);
    if (max<1) { morpho_runtimeerror(v, SELECTION_BND); return; }
    
    grade bnd = max-1;
    
    objectsparse *conn=mesh_addconnectivityelement(sel->mesh, max, bnd);
    
    if (conn) {
        int nentries, *entries;
        for (elementid i=0; i<mesh_nelements(conn); i++) {
            if (mesh_getconnectivity(conn, i, &nentries, &entries)) {
                if (nentries==1) {
                    selection_selectelement(sel, bnd, i);
                }
            }
        }
    }
    // Add vertices if the boundary elements are higher in grade than vertices
    if (bnd!=0) {selection_addgradelower(sel, 0); }
}

/** Selects an element */
void selection_selectwithid(objectselection *sel, grade g, elementid id, bool selected) {
    if (g>=sel->ngrades) return;

    if (selected) {
        dictionary_insert(&sel->selected[g], MORPHO_INTEGER(id), MORPHO_NIL);
    } else {
        dictionary_remove(&sel->selected[g], MORPHO_INTEGER(id));
    }
}

/** Tests if an element is selected */
bool selection_isselected(objectselection *sel, grade g, elementid id) {
    if (g<0 || (unsigned int) g>=sel->ngrades) return false;
    return dictionary_get(&sel->selected[g], MORPHO_INTEGER(id), NULL);
}

/** Number of selected elements of grade g */
unsigned int selection_count(objectselection *sel, grade g) {
    if (g<0 || (unsigned int) g>=sel->ngrades) return 0;
    return sel->selected[g].count;
}

/** Finds the maximum nonempty grade in a selection */
grade selection_maxgrade(objectselection *sel) {
    for (grade g=sel->ngrades-1; g>0; g--) {
        if (sel->selected[g].count>0) return g;
    }
    return 0;
}

/** Gets the element ids for a given grade as a list */
objectlist *selection_idlistforgrade(objectselection *sel, grade g) {
    objectlist *new = object_newlist(0, NULL);
    if (!new) return NULL;
    if (g<0 || (unsigned int) g>=sel->ngrades) return new;

    dictionary *dict = &sel->selected[g];
    list_resize(new, dict->count);
    for (unsigned int i=0; i<dict->capacity; i++) {
        if (MORPHO_ISINTEGER(dict->contents[i].key)) {
            list_append(new, dict->contents[i].key);
        }
    }

    return new;
}

/* **********************************************************************
 * Selection set operations
 * ********************************************************************** */

/* Computes the union of selections a & b */
objectselection *selection_union(objectselection *a, objectselection *b) {
    objectselection *new = object_newselection(a->mesh);

    for (grade g=0; g<a->ngrades && g<b->ngrades; g++) {
        dictionary_union(&a->selected[g], &b->selected[g], &new->selected[g]);
    }

    return new;
}

/* Computes the intersection of selections a & b */
objectselection *selection_intersection(objectselection *a, objectselection *b) {
    objectselection *new = object_newselection(a->mesh);

    for (grade g=0; g<a->ngrades && g<b->ngrades; g++) {
        dictionary_intersection(&a->selected[g], &b->selected[g], &new->selected[g]);
    }

    return new;
}

/* Computes the difference of selections a & b */
objectselection *selection_difference(objectselection *a, objectselection *b) {
    objectselection *new = object_newselection(a->mesh);

    for (grade g=0; g<a->ngrades && g<b->ngrades; g++) {
        dictionary_difference(&a->selected[g], &b->selected[g], &new->selected[g]);
    }

    return new;
}

/* **********************************************************************
 * Selection veneer class
 * ********************************************************************** */

static value selection_boundaryoption;
static value selection_partialsoption;

/* ----------------------
 * Constructors
 * ---------------------- */

value selection_constructor__mesh(vm *v, int nargs, value *args) {
    objectselection *new = object_newselection(MORPHO_GETMESH(MORPHO_GETARG(args, 0)));
    if (new) {
        value boundary = MORPHO_FALSE;
        builtin_options(v, nargs, args, NULL, 1, selection_boundaryoption, &boundary);
        if (MORPHO_ISTRUE(boundary)) selection_selectboundary(v, new);
    }
    return morpho_wrapandbind(v, (object *) new);
}

value selection_constructor__mesh_fn(vm *v, int nargs, value *args) {
    objectselection *new = object_newselection(MORPHO_GETMESH(MORPHO_GETARG(args, 0)));
    if (new) selection_selectwithfunction(v, new, MORPHO_GETARG(args, 1));
    return morpho_wrapandbind(v, (object *) new);
}

value selection_constructor__mesh_fn_matrix(vm *v, int nargs, value *args) {
    objectselection *new = object_newselection(MORPHO_GETMESH(MORPHO_GETARG(args, 0)));
    if (new) selection_selectwithmatrix(v, new, MORPHO_GETARG(args, 1), MORPHO_GETMATRIX(MORPHO_GETARG(args, 2)));
    return morpho_wrapandbind(v, (object *) new);
}

value selection_constructor__mesh_fn_field(vm *v, int nargs, value *args) {
    objectselection *new = object_newselection(MORPHO_GETMESH(MORPHO_GETARG(args, 0)));
    if (new) selection_selectwithfield(v, new, MORPHO_GETARG(args, 1), MORPHO_GETFIELD(MORPHO_GETARG(args, 2)));
    return morpho_wrapandbind(v, (object *) new);
}

/* ----------------------
 * Methods
 * ---------------------- */

/** Fallback for Object-overridden methods with a single typed implementation.
 * A second signature is required so the method is wrapped in a metafunction. */
value Selection_dispatcherr(vm *v, int nargs, value *args) {
    morpho_runtimeerror(v, VM_MLTPLDSPTCHFLD);
    return MORPHO_NIL;
}

/** Select an element by id */
value Selection_setindex__int_int_x(vm *v, int nargs, value *args) {
    selection_selectwithid(MORPHO_GETSELECTION(MORPHO_SELF(args)),
                           MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
                           MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1)),
                           MORPHO_ISTRUE(MORPHO_GETARG(args, 2)));
    return MORPHO_NIL;
}

/** Tests if something is selected */
value Selection_isselected__int_int(vm *v, int nargs, value *args) {
    return MORPHO_BOOL(selection_isselected(MORPHO_GETSELECTION(MORPHO_SELF(args)),
                                            MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)),
                                            MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1))));
}

/** Get the attached mesh */
value Selection_mesh(vm *v, int nargs, value *args) {
    objectselection *sel = MORPHO_GETSELECTION(MORPHO_SELF(args));
    return (sel->mesh ? MORPHO_OBJECT(sel->mesh) : MORPHO_NIL);
}

/** Get the id list for a given grade */
value Selection_idlistforgrade__int(vm *v, int nargs, value *args) {
    return morpho_wrapandbind(v, (object *) selection_idlistforgrade(MORPHO_GETSELECTION(MORPHO_SELF(args)),
                                                                    MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0))));
}

/** Adds a grade to a selection */
value Selection_addgrade__int(vm *v, int nargs, value *args) {
    objectselection *sel = MORPHO_GETSELECTION(MORPHO_SELF(args));
    value partials = MORPHO_FALSE;
    builtin_options(v, nargs, args, NULL, 1, selection_partialsoption, &partials);

    grade g = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    if (g>selection_maxgrade(sel)) {
        selection_addgraderaise(sel, g, MORPHO_ISTRUE(partials));
    } else {
        selection_addgradelower(sel, g);
    }
    return MORPHO_NIL;
}

/** Removes a grade from a selection */
value Selection_removegrade__int(vm *v, int nargs, value *args) {
    selection_removegrade(MORPHO_GETSELECTION(MORPHO_SELF(args)),
                          MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)));
    return MORPHO_NIL;
}

/** Print the selection */
value Selection_print(vm *v, int nargs, value *args) {
    value self = MORPHO_SELF(args);
    if (!MORPHO_ISSELECTION(self)) return Object_print(v, nargs, args);
    morpho_printf(v, "<Selection>");
    return MORPHO_NIL;
}

/** Counts number of elements selected in each grade  */
value Selection_count__int(vm *v, int nargs, value *args) {
    return MORPHO_INTEGER(selection_count(MORPHO_GETSELECTION(MORPHO_SELF(args)),
                                          MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0))));
}

/** Clones a selection */
value Selection_clone(vm *v, int nargs, value *args) {
    return morpho_wrapandbind(v, (object *) selection_clone(MORPHO_GETSELECTION(MORPHO_SELF(args))));
}

static value selection_setop(vm *v, value *args, objectselection *(*op)(objectselection *, objectselection *)) {
    objectselection *a = MORPHO_GETSELECTION(MORPHO_SELF(args));
    objectselection *b = MORPHO_GETSELECTION(MORPHO_GETARG(args, 0));
    if (a->mesh != b->mesh) {
        morpho_runtimeerror(v, SELECTION_MSH);
        return MORPHO_NIL;
    }
    return morpho_wrapandbind(v, (object *) op(a, b));
}

value Selection_union__selection(vm *v, int nargs, value *args) {
    return selection_setop(v, args, selection_union);
}

value Selection_intersection__selection(vm *v, int nargs, value *args) {
    return selection_setop(v, args, selection_intersection);
}

value Selection_difference__selection(vm *v, int nargs, value *args) {
    return selection_setop(v, args, selection_difference);
}

MORPHO_BEGINCLASS(Selection)
MORPHO_METHOD_SIGNATURE(SELECTION_ISSELECTEDMETHOD, "Bool (Int, Int)", Selection_isselected__int_int, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Bool (Int, Int)", Selection_isselected__int_int, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_GETINDEX_METHOD, "Nil (...)", Selection_dispatcherr, MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "(Int, Int, _)", Selection_setindex__int_int_x, MORPHO_FN_MUTATES),
MORPHO_METHOD_SIGNATURE(MORPHO_SETINDEX_METHOD, "Nil (...)", Selection_dispatcherr, MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(SELECTION_IDLISTFORGRADEMETHOD, "List (Int)", Selection_idlistforgrade__int, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(SELECTION_MESHMETHOD, "Mesh ()", Selection_mesh, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_COUNT_METHOD, "Int (Int)", Selection_count__int, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(MORPHO_COUNT_METHOD, "Nil (...)", Selection_dispatcherr, MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_PRINT_METHOD, "()", Selection_print, MORPHO_FN_IO),
MORPHO_METHOD_SIGNATURE(MORPHO_UNION_METHOD, "Selection (Selection)", Selection_union__selection, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_INTERSECTION_METHOD, "Selection (Selection)", Selection_intersection__selection, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_DIFFERENCE_METHOD, "Selection (Selection)", Selection_difference__selection, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_ADD_METHOD, "Selection (Selection)", Selection_union__selection, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(MORPHO_SUB_METHOD, "Selection (Selection)", Selection_difference__selection, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(SELECTION_ADDGRADEMETHOD, "(Int)", Selection_addgrade__int, MORPHO_FN_MUTATES|MORPHO_FN_OPTARGS),
MORPHO_METHOD_SIGNATURE(SELECTION_REMOVEGRADEMETHOD, "(Int)", Selection_removegrade__int, MORPHO_FN_MUTATES),
MORPHO_METHOD_SIGNATURE(MORPHO_CLONE_METHOD, "Selection ()", Selection_clone, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void selection_initialize(void) {
    objectselectiontype=object_addtype(&objectselectiondefn);
    
    selection_boundaryoption=builtin_internsymbolascstring(SELECTION_BOUNDARYOPTION);
    selection_partialsoption=builtin_internsymbolascstring(SELECTION_PARTIALSOPTION);
    
#define SELECTION_CONS_FLGS (MORPHO_FN_CONSTRUCTOR|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS)
    morpho_addfunction(SELECTION_CLASSNAME, "Selection (Mesh)", selection_constructor__mesh, SELECTION_CONS_FLGS|MORPHO_FN_OPTARGS, NULL);
    morpho_addfunction(SELECTION_CLASSNAME, "Selection (Mesh, Callable)", selection_constructor__mesh_fn, SELECTION_CONS_FLGS|MORPHO_FN_REENTRANT, NULL);
    morpho_addfunction(SELECTION_CLASSNAME, "Selection (Mesh, Callable, Matrix)", selection_constructor__mesh_fn_matrix, SELECTION_CONS_FLGS|MORPHO_FN_REENTRANT, NULL);
    morpho_addfunction(SELECTION_CLASSNAME, "Selection (Mesh, Callable, Field)", selection_constructor__mesh_fn_field, SELECTION_CONS_FLGS|MORPHO_FN_REENTRANT, NULL);
    
    value objclass = builtin_findclassfromcstring(OBJECT_CLASSNAME);
    
    value selectionclass=builtin_addclass(SELECTION_CLASSNAME, MORPHO_GETCLASSDEFINITION(Selection), objclass);
    object_setveneerclass(OBJECT_SELECTION, selectionclass);
    
    morpho_defineerror(SELECTION_BND, ERROR_HALT, SELECTION_BND_MSG);
    morpho_defineerror(SELECTION_FLDMSH, ERROR_HALT, SELECTION_FLDMSH_MSG);
    morpho_defineerror(SELECTION_MSH, ERROR_HALT, SELECTION_MSH_MSG);
}

#endif
