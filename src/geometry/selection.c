/** @file selection.c
 *  @author T J Atherton
 *
 *  @brief Selections
 */

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "morpho.h"
#include "object.h"
#include "builtin.h"
#include "classes.h"
#include "matrix.h"
#include "sparse.h"
#include "mesh.h"
#include "selection.h"

/* **********************************************************************
 * Selection object definitions
 * ********************************************************************** */

objecttype objectselectiontype;

/** Selection object definitions */
void objectselection_printfn(object *obj, void *v) {
    morpho_printf(v, "<Selection>");
}

void objectselection_freefn(object *obj) {
    objectselection *s = (objectselection *) obj;
    selection_clear(s);
}

size_t objectselection_sizefn(object *obj) {
    return sizeof(objectselection)+sizeof(objectsparse *)*((objectselection *) obj)->ngrades;
}

objecttypedefn objectselectiondefn = {
    .printfn=objectselection_printfn,
    .markfn=NULL,
    .freefn=objectselection_freefn,
    .sizefn=objectselection_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

/* **********************************************************************
 * Selection object constructor
 * ********************************************************************** */

/** Create a new empty selection object */
objectselection *object_newselection(objectmesh *mesh) {
    unsigned int ngrades = mesh->dim+1;
    objectselection *new=(objectselection *) object_new(sizeof(objectselection)+sizeof(dictionary)*ngrades, OBJECT_SELECTION);
    
    if (new) {
        new->mesh=mesh;
        new->ngrades=ngrades;
        new->mode=SELECT_NONE; 
        for (unsigned int i=0; i<ngrades; i++) dictionary_init(&new->selected[i]);
    }
    
    return new;
}

/** Clones a selection */
objectselection *selection_clone(objectselection *sel) {
    objectselection *new=object_newselection(sel->mesh);
    
    if (new) {
        new->mode=sel->mode;
        for (unsigned int i=0; i<sel->ngrades; i++) dictionary_copy(&sel->selected[i], &new->selected[i]);
    }
    
    return new;
}

/** Clears all data structures associated with a selection */
void selection_clear(objectselection *s) {
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
    if (sel->mode==SELECT_ALL) return true; /* No need to store info in the selectall scenario */
    
    sel->mode=SELECT_SOME; // Ensure that we change modes
    
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
    //dictionary *dest = &sel->selected[g];
    
    for (grade i=sel->ngrades; i>g; i--) { // Loop over grades higher than g
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
        morpho_runtimeerror(v, MATRIX_INCOMPATIBLEMATRICES);
        return;
    }
    
    int nargs = matrix->nrows; // Number of args to pass to function
    value args[nargs]; // Vertex coords
    double *x; // Matrix column
    value ret=MORPHO_NIL; // Return value
    
    for (elementid i=0; i<nv; i++) {
        bool success=matrix_getcolumn(matrix, i, &x);
        
        if (success) {
            for (unsigned int i=0; i<nargs; i++) args[i]=MORPHO_FLOAT(x[i]);
        }
        
        if (!morpho_call(v, fn, nargs, args, &ret)) break;
        if (MORPHO_ISTRUE(ret)) selection_selectelement(sel, MESH_GRADE_VERTEX, i);
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
    if (selected && (sel->mode==SELECT_NONE || sel->mode==SELECT_SOME)) {
        if (g<sel->ngrades) {
            dictionary_insert(&sel->selected[g], MORPHO_INTEGER(id), MORPHO_NIL);
        }
        
        sel->mode=SELECT_SOME;
    } else if (!selected && (sel->mode==SELECT_SOME)) {
        if (g<sel->ngrades) {
            dictionary_remove(&sel->selected[g], MORPHO_INTEGER(id));
        }
        
        sel->mode=SELECT_SOME;
    }
    
    if (sel->mode==SELECT_ALL) {
        UNREACHABLE("Unimplemented modification to SELECTALL not implemented.");
    }
    
}

/** Tests if an element is selected */
bool selection_isselected(objectselection *sel, grade g, elementid id) {
    switch (sel->mode) {
        case SELECT_NONE: return false;
        case SELECT_ALL: return true;
        case SELECT_SOME: {
            return dictionary_get(&sel->selected[g], MORPHO_INTEGER(id), NULL);
        }
    }
}

/** Finds the maximum nonempty grade in a selection */
grade selection_maxgrade(objectselection *sel) {
    switch (sel->mode) {
        case SELECT_NONE: return 0;
        case SELECT_ALL: return mesh_maxgrade(sel->mesh);
        case SELECT_SOME:
            for (grade g=sel->ngrades-1; g>0; g--) {
                if (sel->selected[g].count>0) return g;
            }
    }
    return 0;
}

/** Gets the element ids for a given grade as a list */
objectlist *selection_idlistforgrade(objectselection *sel, grade g) {
    objectlist *new = object_newlist(0, NULL);
    dictionary *dict = &sel->selected[g];
    
    if (new) switch(sel->mode) {
        case SELECT_NONE: break;
        case SELECT_ALL: {
            UNREACHABLE("ID list for select all not implemented.");
        }
            break;
        case SELECT_SOME: {
            list_resize(new, dict->count);
            for (unsigned int i=0; i<dict->capacity; i++) {
                if (MORPHO_ISINTEGER(dict->contents[i].key)) {
                    list_append(new, dict->contents[i].key);
                }
            }
        }
            break;
    }
    
    return new;
}

/* **********************************************************************
 * Selection set operations
 * ********************************************************************** */

/* Computes the union of selections a & b */
objectselection *selection_union(objectselection *a, objectselection *b) {
    objectselection *new = object_newselection(a->mesh);
    
    if (a->mode==SELECT_ALL || b->mode==SELECT_ALL) { // No need to copy a select all element
        new->mode=SELECT_ALL;
    } else {
        for (grade g=0; g<a->ngrades && g<b->ngrades; g++) {
            dictionary_union(&a->selected[g], &b->selected[g], &new->selected[g]);
            if (new->selected[g].count>0) new->mode=SELECT_SOME;
        }
    }
    
    return new;
}

/* Computes the union of selections a & b */
objectselection *selection_intersection(objectselection *a, objectselection *b) {
    objectselection *new = object_newselection(a->mesh);
    
    if (a->mode==SELECT_ALL && b->mode==SELECT_ALL) { // No need to copy a select all element
        new->mode=SELECT_ALL;
    } else if (a->mode!=SELECT_NONE && b->mode!=SELECT_NONE) {
        for (grade g=0; g<a->ngrades && g<b->ngrades; g++) {
            dictionary_intersection(&a->selected[g], &b->selected[g], &new->selected[g]);
            if (new->selected[g].count>0) new->mode=SELECT_SOME;
        }
    }
    
    return new;
}

/* Computes the union of selections a & b */
objectselection *selection_difference(objectselection *a, objectselection *b) {
    objectselection *new = object_newselection(a->mesh);
    
    if (a->mode==SELECT_ALL) {
        if (b->mode==SELECT_NONE) {
            new->mode=SELECT_ALL;
        } else UNREACHABLE("Selectall difference not implemented.");
    } else if (a->mode!=SELECT_NONE) {
        for (grade g=0; g<a->ngrades && g<b->ngrades; g++) {
            dictionary_difference(&a->selected[g], &b->selected[g], &new->selected[g]);
            if (new->selected[g].count>0) new->mode=SELECT_SOME;
        }
    }
    
    return new;
}

/* **********************************************************************
 * Selection veneer class
 * ********************************************************************** */

static value selection_boundaryoption;
static value selection_partialsoption;

/** Constructs a Selection object */
value selection_constructor(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectmesh *mesh=NULL;
    objectselection *new=NULL;
    value fn=MORPHO_NIL;
    value fnargs=MORPHO_NIL;
    value boundary=MORPHO_FALSE;
    int nfixed=nargs;
    
    builtin_options(v, nargs, args, &nfixed, 1, selection_boundaryoption, &boundary);
    
    /* Get mesh as first argument */
    if (nfixed>0) {
        if (MORPHO_ISMESH(MORPHO_GETARG(args, 0))) mesh=MORPHO_GETMESH(MORPHO_GETARG(args, 0));
    }
    
    /* Selection function as optional second argument */
    if (nfixed>1) fn = MORPHO_GETARG(args, 1);
    
    /* Selection function arguments as optional second argument */
    if (nfixed>2) fnargs = MORPHO_GETARG(args, 2);
    
    if (mesh) {
        new=object_newselection(mesh);
    } else {
        morpho_runtimeerror(v, SELECTION_NOMESH);
        return out;
    }
    
    if (new) {
        if (!MORPHO_ISNIL(fn)) {
            if (MORPHO_ISNIL(fnargs)) {
                selection_selectwithfunction(v, new, fn);
            } else if (MORPHO_ISMATRIX(fnargs)) {
                selection_selectwithmatrix(v, new, fn, MORPHO_GETMATRIX(fnargs));
            }
        } else if (MORPHO_ISTRUE(boundary)) {
            selection_selectboundary(v, new);
        }
        
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    
    return out;
}

/** Select an element by id */
value Selection_setindex(vm *v, int nargs, value *args) {
    objectselection *sel = MORPHO_GETSELECTION(MORPHO_SELF(args));
    
    if (nargs==3 &&
        MORPHO_ISINTEGER(MORPHO_GETARG(args, 0)) &&
        MORPHO_ISINTEGER(MORPHO_GETARG(args, 1))) {
    
        selection_selectwithid(sel, MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)), MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1)), MORPHO_ISTRUE(MORPHO_GETARG(args, 2)));
    } else morpho_runtimeerror(v, SELECTION_ISSLCTDARG);
    
    return MORPHO_NIL;
}

/** Tests if something is selected */
value Selection_isselected(vm *v, int nargs, value *args) {
    objectselection *sel = MORPHO_GETSELECTION(MORPHO_SELF(args));
    value out = MORPHO_FALSE;
    
    if (nargs==2 &&
        MORPHO_ISINTEGER(MORPHO_GETARG(args, 0)) &&
        MORPHO_ISINTEGER(MORPHO_GETARG(args, 1))) {
     
        out = MORPHO_BOOL(selection_isselected(sel, MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)), MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1))));
    } else morpho_runtimeerror(v, SELECTION_ISSLCTDARG);
    
    return out;
}

/** Get the id list for a given grade */
value Selection_idlistforgrade(vm *v, int nargs, value *args) {
    objectselection *sel = MORPHO_GETSELECTION(MORPHO_SELF(args));
    value out = MORPHO_NIL;
    
    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        objectlist *lst=selection_idlistforgrade(sel, MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)));
        if (lst) {
            out = MORPHO_OBJECT(lst);
            morpho_bindobjects(v, 1, &out);
        } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    } else morpho_runtimeerror(v, SELECTION_GRADEARG);
    
    return out;
}

/** Adds a grade to a selection */
value Selection_addgrade(vm *v, int nargs, value *args) {
    objectselection *sel = MORPHO_GETSELECTION(MORPHO_SELF(args));
    value partials = MORPHO_FALSE;
    int nfixed;

    builtin_options(v, nargs, args, &nfixed, 1, selection_partialsoption, &partials);
    
    if (nargs>0 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        grade g = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        grade max = selection_maxgrade(sel);
        if (g>max) {
            selection_addgraderaise(sel, g, MORPHO_ISTRUE(partials));
        } else {
            selection_addgradelower(sel, g);
        }
    } else morpho_runtimeerror(v, SELECTION_GRADEARG);
    
    return MORPHO_NIL;
}

/** Removes a grade from a selection */
value Selection_removegrade(vm *v, int nargs, value *args) {
    objectselection *sel = MORPHO_GETSELECTION(MORPHO_SELF(args));

    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        selection_removegrade(sel, MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0)));
    } else morpho_runtimeerror(v, SELECTION_GRADEARG);
    
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
value Selection_count(vm *v, int nargs, value *args) {
    objectselection *sel = MORPHO_GETSELECTION(MORPHO_SELF(args));
    value out = MORPHO_NIL;

    if (nargs==1 && MORPHO_ISINTEGER(MORPHO_GETARG(args, 0))) {
        grade g = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
        out = MORPHO_INTEGER(sel->selected[g].count);
    } else morpho_runtimeerror(v, SELECTION_GRADEARG);
    
    return out;
}

/** Clones a selection */
value Selection_clone(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectselection *a=MORPHO_GETSELECTION(MORPHO_SELF(args));
    objectselection *new=selection_clone(a);
    if (new) {
        out=MORPHO_OBJECT(new);
        morpho_bindobjects(v, 1, &out);
    } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    return out;
}

#define SELECTION_SETOP(op) \
value Selection_##op(vm *v, int nargs, value *args) { \
    objectselection *slf = MORPHO_GETSELECTION(MORPHO_SELF(args)); \
    value out=MORPHO_NIL; \
    \
    if (nargs>0 && MORPHO_ISSELECTION(MORPHO_GETARG(args, 0))) { \
        objectselection *new = selection_##op(slf, MORPHO_GETSELECTION(MORPHO_GETARG(args, 0))); \
        \
        if (new) { \
            out=MORPHO_OBJECT(new); \
            morpho_bindobjects(v, 1, &out); \
        } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); \
    } else morpho_runtimeerror(v, SELECTION_STARG); \
    \
    return out; \
}

SELECTION_SETOP(union)
SELECTION_SETOP(intersection)
SELECTION_SETOP(difference)

MORPHO_BEGINCLASS(Selection)
MORPHO_METHOD(SELECTION_ISSELECTEDMETHOD, Selection_isselected, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_GETINDEX_METHOD, Selection_isselected, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SETINDEX_METHOD, Selection_setindex, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SELECTION_IDLISTFORGRADEMETHOD, Selection_idlistforgrade, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_COUNT_METHOD, Selection_count, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_PRINT_METHOD, Selection_print, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_UNION_METHOD, Selection_union, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_INTERSECTION_METHOD, Selection_intersection, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_DIFFERENCE_METHOD, Selection_difference, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_ADD_METHOD, Selection_union, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_SUB_METHOD, Selection_difference, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SELECTION_ADDGRADEMETHOD, Selection_addgrade, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(SELECTION_REMOVEGRADEMETHOD, Selection_removegrade, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(MORPHO_CLONE_METHOD, Selection_clone, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void selection_initialize(void) {
    objectselectiontype=object_addtype(&objectselectiondefn);
    
    selection_boundaryoption=builtin_internsymbolascstring(SELECTION_BOUNDARYOPTION);
    selection_partialsoption=builtin_internsymbolascstring(SELECTION_PARTIALSOPTION);
    
    builtin_addfunction(SELECTION_CLASSNAME, selection_constructor, BUILTIN_FLAGSEMPTY);
    
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    value selectionclass=builtin_addclass(SELECTION_CLASSNAME, MORPHO_GETCLASSDEFINITION(Selection), objclass);
    object_setveneerclass(OBJECT_SELECTION, selectionclass);
    
    morpho_defineerror(SELECTION_NOMESH, ERROR_HALT, SELECTION_NOMESH_MSG);
    morpho_defineerror(SELECTION_ISSLCTDARG, ERROR_HALT, SELECTION_ISSLCTDARG_MSG);
    morpho_defineerror(SELECTION_GRADEARG, ERROR_HALT, SELECTION_GRADEARG_MSG);
    morpho_defineerror(SELECTION_STARG, ERROR_HALT, SELECTION_STARG_MSG);
    morpho_defineerror(SELECTION_BND, ERROR_HALT, SELECTION_BND_MSG);
}

#endif
