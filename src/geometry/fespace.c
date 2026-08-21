/** @file fespace.c
 *  @author T J Atherton
 *
 *  @brief Finite element fespaces and veneer class
 */

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include "geometry.h"

/* **********************************************************************
 * fespace objects
 * ********************************************************************** */

objecttype objectfespacetype;

/** Field object definitions */
void objectfespace_printfn(object *obj, void *v) {
    objectfespace *disc=(objectfespace *) obj;
    morpho_printf(v, "<FunctionSpace %s>", FESPACE_NAME(disc->fespace));
}

size_t objectfespace_sizefn(object *obj) {
    return sizeof(objectfespace);
}

objecttypedefn objectfespacedefn = {
    .printfn=objectfespace_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectfespace_sizefn
};

/** Creates a new fespace object
 * @param[in] fespace - fespace definition to use */
objectfespace *objectfespace_new(fespace *disc) {
    objectfespace *new = (objectfespace *) object_new(sizeof(objectfespace), OBJECT_FESPACE);
    if (new) new->fespace=disc;
    
    return new;
}

/* **********************************************************************
 * Discretization functions
 * ********************************************************************** */

extern fespace *fespaces[];

/** Find a fespace definition based on a name and grade */
fespace *fespace_find(char *name, grade g) {
    for (int i=0; fespaces[i]!=NULL; i++) {
        if (strcmp(name, fespaces[i]->name)==0 &&
            g==fespaces[i]->grade) return fespaces[i];
    }
    return NULL;
}

/** Finds a linear fespace for a given grade */
fespace *fespace_findlinear(grade g) {
    for (int i=0; fespaces[i]!=NULL; i++) {
        if (fespaces[i]->grade==g && fespaces[i]->degree==1) return fespaces[i];
    }
    return NULL;
}

#define FETCH(instr) (*(instr++))

typedef struct {
    elementid id;
    bool reversed;
} fespacesubelement;

/** Remaps an edge-local quantity index if the matched line orientation is reversed. */
static int fespace_orientedquantity(fespace *disc, grade g, fespacesubelement *subel, int sid, int indx) {
    int stride = disc->nsubel+1;
    if (g!=MESH_GRADE_LINE || !subel[g*stride+sid].reversed) return indx;

    fespace *lower;
    if (!fespace_lower(disc, g, &lower)) return indx;

    int nedgeq = lower->shape[MESH_GRADE_LINE];
    if (indx<0 || indx>=nedgeq) return indx;

    return nedgeq-indx-1;
}

/** Returns the FE-local field index tuple (grade, subelement id, local dof index) for a node. */
bool fespace_nodefieldindex(fespace *disc, int node, grade *g, int *sid, int *indx) {
    if (node<0 || node>=disc->nnodes) return false;

    int k=0;
    for (eldefninstruction *instr=disc->eldefn; instr!=NULL && *instr!=ELEMENT_ENDDEFN; ) {
        eldefninstruction op=FETCH(instr);
        switch(op) {
            case ELEMENT_LINE_OPCODE:
            case ELEMENT_AREA_OPCODE:
            case ELEMENT_VOLUME_OPCODE:
                FETCH(instr); // local subelement id
                for (int i=0; i<=op; i++) FETCH(instr); // local vertex ids
                break;
            case ELEMENT_QUANTITY_OPCODE:
            {
                grade qg = FETCH(instr);
                int qsid = FETCH(instr);
                int qindx = FETCH(instr);
                if (k==node) {
                    if (g) *g=qg;
                    if (sid) *sid=qsid;
                    if (indx) *indx=qindx;
                    return true;
                }
                k++;
            }
                break;
            default:
                UNREACHABLE("Error in finite element definition");
        }
    }

    return false;
}

/** Steps through an element definition, generating subelements and identifying quantities */
bool fespace_doftofieldindx(objectfield *field, fespace *disc, int nv, int *vids, fieldindx *findx) {
    int stride = disc->nsubel+1;
    fespacesubelement subel[(disc->grade+1)*stride]; // Element IDs and orientation of subelements
    int sid, svids[nv], nmatch, k=0;
    
    objectsparse *vmatrix[disc->grade+1]; // Vertex->elementid connectivity matrices
    for (grade g=0; g<=disc->grade; g++) {
        vmatrix[g]=mesh_addconnectivityelement(field->mesh, g, 0);
        if (!vmatrix[g] && g>0 && disc->shape[g]>0) {
            mesh_addgrade(field->mesh, g);
            vmatrix[g]=mesh_addconnectivityelement(field->mesh, g, 0);
        }
    }
    objectsparse *lineconn = mesh_getconnectivityelement(field->mesh, 0, MESH_GRADE_LINE);
    
    for (eldefninstruction *instr=disc->eldefn; instr!=NULL && *instr!=ELEMENT_ENDDEFN; ) {
        eldefninstruction op=FETCH(instr);
        switch(op) {
            case ELEMENT_LINE_OPCODE: // Find an element defined by n vertices
            case ELEMENT_AREA_OPCODE: // TODO: Need to cope with (mis) orientation of these subelements
            case ELEMENT_VOLUME_OPCODE: // P0 ignores orientation
            {
                sid = FETCH(instr);
                for (int i=0; i<=op; i++) svids[i] = vids[FETCH(instr)];
                
                fespacesubelement *matched = &subel[op*stride+sid];
                matched->id = -1;
                if (!mesh_matchelements(vmatrix[op], op, op+1, svids, 1, &nmatch, &matched->id)) return false;
                if (nmatch!=1 || matched->id<0) return false;

                matched->reversed=false;
                if (op==ELEMENT_LINE_OPCODE) {
                    int nlinev, *linevids;
                    if (!lineconn || !mesh_getconnectivity(lineconn, matched->id, &nlinev, &linevids)) return false;
                    if (nlinev!=2) return false;

                    if (linevids[0]==svids[1] && linevids[1]==svids[0]) {
                        matched->reversed=true;
                    } else if (!(linevids[0]==svids[0] && linevids[1]==svids[1])) {
                        return false;
                    }
                }
            }
                break;
            case ELEMENT_QUANTITY_OPCODE:
            {
                findx[k].g=FETCH(instr);
                int sid=FETCH(instr);
                findx[k].id=(findx[k].g==0 ? vids[sid]: subel[findx[k].g*stride+sid].id);
                findx[k].indx=fespace_orientedquantity(disc, findx[k].g, subel, sid, FETCH(instr));
                k++;
            }
                break;
            default:
                UNREACHABLE("Error in finite element definition");
        }
    }
    return true;
}

/** Searches a fespace's lower list to find a fespace to use on a lower grade */
bool fespace_lower(fespace *disc, grade target, fespace **out) {
    if (disc->lower) for (int i=0; disc->lower[i]!=NULL; i++) {
        if (disc->lower[i]->grade==target) {
            *out = disc->lower[i];
            return true;
        }
    }
    return false;
}

/** Returns the barycentric coordinates of a node in the reference element */
bool fespace_getnodecoords(fespace *disc, int node, double *lambda) {
    if (!disc || !lambda) return false;
    if (node<0 || node>=disc->nnodes) return false;

    double l0 = 1.0;

    for (int i=0; i<disc->grade; i++) {
        double li = disc->nodes[node*disc->grade+i];
        lambda[i+1] = li;
        l0 -= li;
    }

    lambda[0] = l0;

    return true;
}

/** Constructs a layout matrix that maps element ids (columns) to degree of freedom indices in a field */
bool fespace_layout(objectfield *field, fespace *disc, objectsparse **out) {
    *out = NULL;
    objectsparse *conn = mesh_getconnectivityelement(field->mesh, 0, disc->grade);
    if (!conn) conn = mesh_addconnectivityelement(field->mesh, 0, disc->grade);
    elementid nel=mesh_nelements(conn);
    
    objectsparse *new = object_newsparse(NULL, NULL);
    if (!new) return false;
    sparseccs_resize(&new->ccs, field->nelements, nel, nel*disc->nnodes, NULL);
    
    for (elementid id=0; id<nel; id++) {
        int nv, *vids;
        if (!mesh_getconnectivity(conn, id, &nv, &vids)) goto fespace_layout_cleanup;
     
        new->ccs.cptr[id]=id*disc->nnodes;
        fieldindx findx[disc->nnodes];
        if (!fespace_doftofieldindx(field, disc, nv, vids, findx)) goto fespace_layout_cleanup;
        for (int i=0; i<disc->nnodes; i++) {
            if (!field_getindex(field, findx[i].g, findx[i].id, findx[i].indx, new->ccs.rix+new->ccs.cptr[id]+i)) goto fespace_layout_cleanup;
        }
    }
    new->ccs.cptr[nel]=nel*disc->nnodes; // Last column pointer points to next column
    
    *out=new;
    return true;
    
fespace_layout_cleanup:
    if (new) object_free((object *) new);
    return false;
}

/** @brief Calculates the gradient of the basis functions with respect to the reference coordinates.
 *  @param[in] disc - fespace to query
 *  @param[in] lambda - position in barycentric coordinates
 *  @param[out] grad - gradient of basis functions with respect to reference coordinates.
 *                     Layout is column-major by component:
 *                     grad->elements[FESPACE_GRAD_INDEX(disc->nnodes, component, node)].
 *  @pre FESPACE_HASGRADIENT(disc)
 */
void fespace_gradient(fespace *disc, double *lambda, objectmatrix *grad) {
    int nbary = disc->grade+1;
    
    // Compute gradients of the basis functions with respect to barycentric coordinates
    double gdata[disc->nnodes*nbary];
    (disc->gfn) (lambda, gdata);
    
    for (int i=0; i<disc->grade; i++) {
        functional_vecsub(disc->nnodes, gdata+(i+1)*disc->nnodes, gdata, grad->elements+i*disc->nnodes);
    }
}

/** @brief Calculates the Hessian of the basis functions with respect to the reference coordinates.
 *  @param[in] disc - fespace to query
 *  @param[in] lambda - position in barycentric coordinates
 *  @param[out] hess - Hessian of basis functions with respect to reference coordinates
 *                     in column-major tensor order:
 *                     hess->elements[FESPACE_HESS_INDEX(disc->nnodes, disc->grade, row, col, node)].
 *  @pre FESPACE_HASHESSIAN(disc)
 */
void fespace_hessian(fespace *disc, double *lambda, objectmatrix *hess) {
    if (disc->hfn) (disc->hfn) (lambda, hess->elements);
}

/* **********************************************************************
 * FiniteElementSpace class
 * ********************************************************************** */

/** Constructs a fespace object */
value fespace_constructor(vm *v, int nargs, value *args) {
    value grd=MORPHO_INTEGER(1);
    int nfixed;

    if (!builtin_options(v, nargs, args, &nfixed, 1, field_gradeoption, &grd) ||
        nfixed!=1 || !MORPHO_ISINTEGER(grd)) {
        morpho_runtimeerror(v, FNSPC_ARGS);
        return MORPHO_NIL;
    }

    char *label = MORPHO_GETCSTRING(MORPHO_GETARG(args, 0));
    fespace *d=fespace_find(label, MORPHO_GETINTEGERVALUE(grd));

    if (!d) {
        morpho_runtimeerror(v, FNSPC_NOTFOUND, label, MORPHO_GETINTEGERVALUE(grd));
        return MORPHO_NIL;
    }

    return morpho_wrapandbind(v, (object *) objectfespace_new(d));
}

value FiniteElementSpace_count(vm *v, int nargs, value *args) {
    objectfespace *slf = MORPHO_GETFESPACE(MORPHO_SELF(args));
    return MORPHO_INTEGER(slf->fespace->nnodes);
}

value FiniteElementSpace_grade(vm *v, int nargs, value *args) {
    objectfespace *slf = MORPHO_GETFESPACE(MORPHO_SELF(args));
    return MORPHO_INTEGER(slf->fespace->grade);
}

value FiniteElementSpace_layout(vm *v, int nargs, value *args) {
    objectfespace *slf = MORPHO_GETFESPACE(MORPHO_SELF(args));
    objectsparse *new = NULL;

    fespace_layout(MORPHO_GETFIELD(MORPHO_GETARG(args, 0)), slf->fespace, &new);
    return morpho_wrapandbind(v, (object *) new);
}

value FiniteElementSpace_nodeelementindex(vm *v, int nargs, value *args) {
    objectfespace *slf = MORPHO_GETFESPACE(MORPHO_SELF(args));
    int i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    grade g;
    int sid, indx;

    if (!fespace_nodefieldindex(slf->fespace, i, &g, &sid, &indx)) {
        morpho_runtimeerror(v, VM_OUTOFBOUNDS);
        return MORPHO_NIL;
    }

    value entries[3] = { MORPHO_INTEGER(g), MORPHO_INTEGER(sid), MORPHO_INTEGER(indx) };
    return morpho_wrapandbind(v, (object *) object_newtuple(3, entries));
}

value FiniteElementSpace_nodecoords(vm *v, int nargs, value *args) {
    objectfespace *slf = MORPHO_GETFESPACE(MORPHO_SELF(args));
    fespace *disc = slf->fespace;
    int nrows = disc->grade+1;
    objectmatrix *new = matrix_new(nrows, disc->nnodes, true);

    if (new) {
        for (int i=0; i<disc->nnodes; i++) {
            double lambda[nrows];
            if (!fespace_getnodecoords(disc, i, lambda)) {
                object_free((object *) new);
                new = NULL;
                break;
            }
            matrix_setcolumnptr(new, i, lambda);
        }
    }

    return morpho_wrapandbind(v, (object *) new);
}

value FiniteElementSpace_nodecoords__int(vm *v, int nargs, value *args) {
    objectfespace *slf = MORPHO_GETFESPACE(MORPHO_SELF(args));
    fespace *disc = slf->fespace;
    int nrows = disc->grade+1;
    int i = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 0));
    double lambda[nrows];

    if (!fespace_getnodecoords(disc, i, lambda)) {
        morpho_runtimeerror(v, VM_OUTOFBOUNDS);
        return MORPHO_NIL;
    }

    objectmatrix *new = matrix_new(nrows, 1, true);
    if (new) matrix_setcolumnptr(new, 0, lambda);
    return morpho_wrapandbind(v, (object *) new);
}

MORPHO_BEGINCLASS(FiniteElementSpace)
MORPHO_METHOD_SIGNATURE(MORPHO_COUNT_METHOD, "Int ()", FiniteElementSpace_count, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(FINITEELEMENTSPACE_GRADE_METHOD, "Int ()", FiniteElementSpace_grade, MORPHO_FN_PUREFN),
MORPHO_METHOD_SIGNATURE(FINITEELEMENTSPACE_LAYOUT_METHOD, "Sparse (Field)", FiniteElementSpace_layout, MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(FINITEELEMENTSPACE_NODEELEMENTINDEX_METHOD, "Tuple (Int)", FiniteElementSpace_nodeelementindex, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(FINITEELEMENTSPACE_NODECOORDS_METHOD, "Matrix ()", FiniteElementSpace_nodecoords, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD_SIGNATURE(FINITEELEMENTSPACE_NODECOORDS_METHOD, "Matrix (Int)", FiniteElementSpace_nodecoords__int, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void fespace_initialize(void) {
    objectfespacetype=object_addtype(&objectfespacedefn);

    morpho_addfunction(FINITEELEMENTSPACE_CLASSNAME, "FiniteElementSpace (String)", fespace_constructor, MORPHO_FN_CONSTRUCTOR|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_OPTARGS, NULL);

    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));

    value fespaceclass=builtin_addclass(FINITEELEMENTSPACE_CLASSNAME, MORPHO_GETCLASSDEFINITION(FiniteElementSpace), objclass);
    object_setveneerclass(OBJECT_FESPACE, fespaceclass);

    morpho_defineerror(FNSPC_ARGS, ERROR_HALT, FNSPC_ARGS_MSG);
    morpho_defineerror(FNSPC_NOTFOUND, ERROR_HALT, FNSPC_NOTFOUND_MSG);
}

#endif