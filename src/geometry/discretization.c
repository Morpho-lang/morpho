/** @file discretization.c
 *  @author T J Atherton
 *
 *  @brief Finite element discretizations
 */

#include "geometry.h"

/* **********************************************************************
 * Discretization objects
 * ********************************************************************** */

objecttype objectdiscretizationtype;

/** Field object definitions */
void objectdiscretization_printfn(object *obj, void *v) {
    objectdiscretization *disc=(objectdiscretization *) obj;
    morpho_printf(v, "<FunctionSpace %s>", disc->discretization->name);
}

size_t objectdiscretization_sizefn(object *obj) {
    return sizeof(objectdiscretization);
}

objecttypedefn objectdiscretizationdefn = {
    .printfn=objectdiscretization_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectdiscretization_sizefn
};

/** Creates a new discretization object
 * @param[in] discretization - discretization definition to use */
objectdiscretization *objectdiscretization_new(discretization *disc) {
    objectdiscretization *new = (objectdiscretization *) object_new(sizeof(objectdiscretization), OBJECT_DISCRETIZATION);
    if (new) new->discretization=disc;
    
    return new;
}

/* **********************************************************************
 * Discretization definitions
 * ********************************************************************** */

#define LINE_OPCODE 1
#define AREA_OPCODE 2
#define QUANTITY_OPCODE 255

#define LINE(id, v1, v2)      LINE_OPCODE, id, v1, v2           // Identify a grade 1 subelement given by two vertex indices
#define AREA(id, v1, v2, v3)  AREA_OPCODE, id, v1, v2, v3       // Identify a grade 2 subelement given by three vertex indices
#define QUANTITY(grade, id, qno) QUANTITY_OPCODE, grade, id, qno  // Fetch quantity from subelement of grade with id and quantity number
#define ENDDEFN -1

/* -------------------------------------------------------
 * CG1 element in 1D
 * ------------------------------------------------------- */

/*
 *   0 - 1    // One degree of freedom per vertex
 */

void cg1_1dinterpolate(double *lambda, double *wts) {
    wts[0]=lambda[0];
    wts[1]=lambda[1];
}

unsigned int cg1_1dshape[] = { 1, 0 };

double cg1_1dnodes[] = { 0.0, 1.0 };

eldefninstruction cg1_1ddefn[] = {
    QUANTITY(0,0,0), // Fetch quantity on vertex 0
    QUANTITY(0,1,0), // Fetch quantity on vertex 1
    ENDDEFN
};

discretization cg1_1d = {
    .name = "CG1",
    .grade = 1,
    .shape = cg1_1dshape,
    .degree = 1,
    .nnodes = 2,
    .nsubel = 0,
    .nodes = cg1_1dnodes,
    .ifn = cg1_1dinterpolate,
    .eldefn = cg1_1ddefn
};

/* -------------------------------------------------------
 * CG2 element in 1D
 * ------------------------------------------------------- */

/*
 *   0 - 2 - 1    // One degree of freedom per vertex; one at the midpoint
 */

void cg2_1dinterpolate(double *lambda, double *wts) {
    double dl = (lambda[0]-lambda[1]);
    wts[0]=lambda[0]*dl;
    wts[1]=-lambda[1]*dl;
    wts[2]=4*lambda[0]*lambda[1];
}

unsigned int cg2_1dshape[] = { 1, 1 };

double cg2_1dnodes[] = { 0.0, 1.0, 0.5 };

eldefninstruction cg2_1ddefn[] = {
    LINE(0,0,1),     // Identify line subelement with vertex indices (0,1)
    QUANTITY(0,0,0), // Fetch quantity on vertex 0
    QUANTITY(0,1,0), // Fetch quantity on vertex 1
    QUANTITY(1,0,0), // Fetch quantity from line subelement
    ENDDEFN
};

discretization cg2_1d = {
    .name = "CG2",
    .grade = 1,
    .shape = cg2_1dshape,
    .degree = 2,
    .nnodes = 3,
    .nsubel = 1,
    .nodes = cg2_1dnodes,
    .ifn = cg2_1dinterpolate,
    .eldefn = cg2_1ddefn
};

/* -------------------------------------------------------
 * CG3 element in 1D
 * ------------------------------------------------------- */

/*
 *   0 - 2 - 3 - 1    // One degree of freedom per vertex; two on the line
 */

void cg3_1dinterpolate(double *lambda, double *wts) {
    double a = (9.0/2.0)*lambda[0]*lambda[1];
    wts[0]=lambda[0]*(1-a);
    wts[1]=lambda[1]*(1-a);
    wts[2]=a*(2*lambda[0]-lambda[1]);
    wts[3]=a*(2*lambda[1]-lambda[0]);
}

unsigned int cg3_1dshape[] = { 1, 2 };

double cg3_1dnodes[] = { 0.0, 1.0, 1.0/3.0, 2.0/3.0 };

eldefninstruction cg3_1ddefn[] = {
    LINE(0,0,1),     // Identify line subelement with vertex indices (0,1)
    QUANTITY(0,0,0), // Fetch quantity on vertex 0
    QUANTITY(0,1,0), // Fetch quantity on vertex 1
    QUANTITY(1,0,0), // Fetch quantity from line subelement
    QUANTITY(1,0,1), // Fetch quantity from line subelement
    ENDDEFN
};

discretization cg3_1d = {
    .name = "CG3",
    .grade = 1,
    .shape = cg3_1dshape,
    .degree = 3,
    .nnodes = 4,
    .nsubel = 1,
    .nodes = cg3_1dnodes,
    .ifn = cg3_1dinterpolate,
    .eldefn = cg3_1ddefn
};

/* -------------------------------------------------------
 * CG1 element in 2D
 * ------------------------------------------------------- */

/*   2
 *   |\
 *   0-1    // One degree of freedom per vertex
 */

void cg1_2dinterpolate(double *lambda, double *wts) {
    wts[0]=lambda[0];
    wts[1]=lambda[1];
    wts[2]=lambda[2];
}

unsigned int cg1_2dshape[] = { 1, 0, 0 };

double cg1_2dnodes[] = { 0.0, 0.0,
                         1.0, 0.0,
                         0.0, 1.0 };

eldefninstruction cg1_2deldefn[] = {
    QUANTITY(0,0,0), // Fetch quantity on vertex 0
    QUANTITY(0,1,0), // Fetch quantity on vertex 1
    QUANTITY(0,2,0), // Fetch quantity on vertex 2
    ENDDEFN
};

discretization cg1_2d = {
    .name = "CG1",
    .grade = 2,
    .shape = cg1_2dshape,
    .degree = 1,
    .nnodes = 3,
    .nsubel = 0,
    .nodes = cg1_2dnodes,
    .ifn = cg1_2dinterpolate,
    .eldefn = cg1_2deldefn
};

/* -------------------------------------------------------
 * CG2 element in 2D
 * ------------------------------------------------------- */

/*   2
 *   |\
 *   5 4
 *   |  \
 *   0-3-1    // One degree of freedom per vertex; one at the midpoint
 */

void cg2_2dinterpolate(double *lambda, double *wts) {
    wts[0]=lambda[0]*(2*lambda[0]-1);
    wts[1]=lambda[1]*(2*lambda[1]-1);
    wts[2]=lambda[2]*(2*lambda[2]-1);
    wts[3]=4*lambda[0]*lambda[1];
    wts[4]=4*lambda[1]*lambda[2];
    wts[5]=4*lambda[2]*lambda[0];
}

unsigned int cg2_2dshape[] = { 1, 1, 0 };

double cg2_2dnodes[] = { 0.0, 0.0,
                         1.0, 0.0,
                         0.0, 1.0,
                         0.5, 0.0,
                         0.5, 0.5,
                         0.0, 0.5 };

eldefninstruction cg2_2deldefn[] = {
    LINE(0,0,1),     // Identify line subelement with vertex indices (0,1)
    LINE(1,1,2),     // Identify line subelement with vertex indices (1,2)
    LINE(2,2,0),     // Identify line subelement with vertex indices (2,0)
    QUANTITY(0,0,0), // Fetch quantity on vertex 0
    QUANTITY(0,1,0), // Fetch quantity on vertex 1
    QUANTITY(0,2,0), // Fetch quantity on vertex 2
    QUANTITY(1,0,0), // Fetch quantity from line 0
    QUANTITY(1,1,0), // Fetch quantity from line 1
    QUANTITY(1,2,0), // Fetch quantity from line 2
    ENDDEFN
};

discretization cg2_2d = {
    .name = "CG2",
    .grade = 2,
    .shape = cg2_2dshape,
    .degree = 2,
    .nnodes = 6,
    .nsubel = 3,
    .nodes = cg2_2dnodes,
    .ifn = cg2_2dinterpolate,
    .eldefn = cg2_2deldefn
};

discretization *discretizations[] = {
    &cg1_1d,
    &cg2_1d,
    &cg1_2d,
    &cg2_2d,
    NULL
};

/* **********************************************************************
 * Discretization functions
 * ********************************************************************** */

/** Find a discretization definition based on a name and grade */
discretization *discretization_find(char *name, grade g) {
    for (int i=0; discretizations[i]!=NULL; i++) {
        if (strcmp(name, discretizations[i]->name)==0 &&
            g==discretizations[i]->grade) return discretizations[i];
    }
    return NULL;
}

/** Finds a linear discretization for a given grade */
discretization *discretization_findlinear(grade g) {
    for (int i=0; discretizations[i]!=NULL; i++) {
        if (discretizations[i]->grade && discretizations[i]->degree==1) return discretizations[i];
    }
    return NULL;
}

#define FETCH(instr) (*(instr++))

/** Steps through an element definition, generating subelements and identifying quantities */
bool discretization_doftofieldindx(objectfield *field, discretization *disc, int nv, int *vids, int *dof) {
    elementid subel[disc->nsubel+1]; // Element IDs of sub elements
    int sid, svids[nv], nmatch, k=0;
    
    objectsparse *vmatrix[disc->grade+1]; // Vertex->elementid connectivity matrices
    for (grade g=0; g<=disc->grade; g++) vmatrix[g]=mesh_addconnectivityelement(field->mesh, g, 0);
    
    for (eldefninstruction *instr=disc->eldefn; instr!=NULL && *instr!=ENDDEFN; ) {
        eldefninstruction op=FETCH(instr);
        switch(op) {
            case LINE_OPCODE: // Find an element defined by n vertices
            case AREA_OPCODE: // TODO: Need to cope with (mis) orientation of these subelements
            {
                sid = FETCH(instr);
                for (int i=0; i<=op; i++) svids[i] = vids[FETCH(instr)];
                
                if (!mesh_matchelements(vmatrix[1], op, op+1, svids, 1, &nmatch, &subel[sid])) return false;
            }
                break;
            case QUANTITY_OPCODE:
            {
                grade g = FETCH(instr);
                int sid = FETCH(instr), qno = FETCH(instr);
                
                if (!field_getindex(field, g, (g==0 ? vids[sid]: subel[sid]), qno, &dof[k])) return false;
                k++;
            }
                break;
            default:
                UNREACHABLE("Error in finite element definition");
        }
    }
    return true;
}

/** Constructs a layout matrix that maps element ids (columns) to degree of freedom indices in a field */
bool discretization_layout(objectfield *field, discretization *disc, objectsparse **out) {
    objectsparse *conn = mesh_getconnectivityelement(field->mesh, 0, disc->grade);
    elementid nel=mesh_nelements(conn);
    
    objectsparse *new = object_newsparse(NULL, NULL);
    if (!new) return false;
    sparseccs_resize(&new->ccs, field->nelements, nel, nel*disc->nnodes, NULL);
    
    for (elementid id=0; id<nel; id++) {
        int nv, *vids;
        if (!mesh_getconnectivity(conn, id, &nv, &vids)) goto discretization_layout_cleanup;
     
        new->ccs.cptr[id]=id*disc->nnodes;
        if (!discretization_doftofieldindx(field, disc, nv, vids, new->ccs.rix+new->ccs.cptr[id])) goto discretization_layout_cleanup;
    }
    new->ccs.cptr[nel]=nel*disc->nnodes; // Last column pointer points to next column
    
    *out=new;
    return true;
    
discretization_layout_cleanup:
    if (new) object_free((object *) new);
    return false;
}

/* **********************************************************************
 * FunctionSpace class
 * ********************************************************************** */

/** Constructs a functionspace object */
value functionspace_constructor(vm *v, int nargs, value *args) {
    value grd=MORPHO_INTEGER(1);
    value out=MORPHO_NIL;
    int nfixed;
    
    if (!builtin_options(v, nargs, args, &nfixed, 1, field_gradeoption, &grd))
        morpho_runtimeerror(v, FNSPC_ARGS);
    
    if (nfixed==1 &&
        MORPHO_ISSTRING(MORPHO_GETARG(args, 0)) &&
        MORPHO_ISINTEGER(grd)) {
        char *label = MORPHO_GETCSTRING(MORPHO_GETARG(args, 0)); 
        
        discretization *d=discretization_find(label, MORPHO_GETINTEGERVALUE(grd));
        
        if (d) {
            objectdiscretization *obj=objectdiscretization_new(d);
            if (obj) {
                out = MORPHO_OBJECT(obj);
                morpho_bindobjects(v, 1, &out);
            } else morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
        } else morpho_runtimeerror(v, FNSPC_NOTFOUND, label, MORPHO_GETINTEGERVALUE(grd));
        
    } else morpho_runtimeerror(v, FNSPC_ARGS);
    
    return out;
}

value FunctionSpace_layout(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    objectdiscretization *slf = MORPHO_GETDISCRETIZATION(MORPHO_SELF(args));
    if (nargs==1 && MORPHO_ISFIELD(MORPHO_GETARG(args, 0))) {
        objectfield *field = MORPHO_GETFIELD(MORPHO_GETARG(args, 0));
        objectsparse *new;
        
        if (discretization_layout(field, slf->discretization, &new)) {
            out=MORPHO_OBJECT(new);
            morpho_bindobjects(v, 1, &out);
        }
    }
    return out;
}

MORPHO_BEGINCLASS(FunctionSpace)
MORPHO_METHOD(FUNCTIONSPACE_LAYOUT_METHOD, FunctionSpace_layout, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void discretization_initialize(void) {
    objectdiscretizationtype=object_addtype(&objectdiscretizationdefn);
    
    builtin_addfunction(FUNCTIONSPACE_CLASSNAME, functionspace_constructor, BUILTIN_FLAGSEMPTY);
    
    objectstring objname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objname));
    
    value functionspaceclass=builtin_addclass(FUNCTIONSPACE_CLASSNAME, MORPHO_GETCLASSDEFINITION(FunctionSpace), objclass);
    object_setveneerclass(OBJECT_DISCRETIZATION, functionspaceclass);
    
    morpho_defineerror(FNSPC_ARGS, ERROR_HALT, FNSPC_ARGS_MSG);
    morpho_defineerror(FNSPC_NOTFOUND, ERROR_HALT, FNSPC_NOTFOUND_MSG);
}
