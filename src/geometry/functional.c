/** @file functional.c
 *  @author T J Atherton
 *
 *  @brief Functionals
 */

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include <float.h>
#include <math.h>
#include <stdio.h>

#include "functional.h"
#include "morpho.h"
#include "classes.h"
#include "common.h"

#include "threadpool.h"

#include "linalg.h"
#include "sparse.h"
#include "geometry.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

value functional_gradeproperty;
value functional_fieldproperty;

typedef struct jumpref_s jumpref;
static bool jump_getadjacentparents(jumpref *ref, elementid interfaceid, int *nparents, int **parents);
static bool jump_startfn(vm *v, functional_mapinfo *info);
static void jump_orderparents(int *parents, elementid *plusid, elementid *minusid);
static void functional_fespaceerror(vm *v, objectfield *field, grade g);

/* **********************************************************************
 * Utility functions
 * ********************************************************************** */

double fddelta1, // = pow(MORPHO_EPS, 1.0/3.0),
       fddelta2; // = pow(MORPHO_EPS, 1.0/4.0);

// Estimates the correct stepsize for cell centered finite differences
double functional_fdstepsize(double x, int order) {
    double h = fddelta1;
    if (order==2) h = fddelta2;
    
    // h should be multiplied by an estimate of the lengthscale over which f changes,
    //      | f / f''' | ^ (1/3)
    double absx = fabs(x);
    if (absx>1) h*=absx; // In the absence of other information, and unless we're near 0, use x as the best estimate.
    
    // Ensure stepsize results in a representable number
    volatile double temp = x+h; // Prevent compiler optimizing this away
    return temp-x;
}

static void functional_clearmapinfo(functional_mapinfo *info) {
    info->mesh=NULL;
    info->field=NULL;
    info->sel=NULL;
    info->g=-1; /* Vertex grade is 0; -1 means unset */
    info->id=0;
    info->integrand=NULL;
    info->grad=NULL;
    info->start=NULL;
    info->end=NULL;
    info->dependencies=NULL;
    info->cloneref=NULL;
    info->freeref=NULL;
    info->ref=NULL;
    info->sym=SYMMETRY_NONE;
}

/** Fill mapinfo from typed pointers. Unused slots are NULL.
 * If field is set and mesh is NULL, the field's mesh is used. */
static void _functional_mapinfo(functional_mapinfo *info,
                               objectmesh *mesh,
                               objectselection *sel,
                               objectfield *field) {
    functional_clearmapinfo(info);
    info->mesh = mesh;
    info->sel = sel;
    info->field = field;
    if (field && !mesh) info->mesh = field->mesh;
}

/** Map helpers used by FUNCTIONAL_MD_* wrappers.
 * _functional_run applies the class default grade if info->g is unset. */
static value _functional_run(vm *v, functional_mapinfo *info, grade g, functional_mapcallback *mapfn, bool bind) {
    if (info->g < 0) info->g = g;
    value out=MORPHO_NIL;
    functional_runmap(v, info, mapfn, &out);
    if (bind && !MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

static value _functional_integrand(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn) {
    info->integrand = fn;
    return _functional_run(v, info, g, functional_mapintegrand, true);
}

static value _functional_integrand_elem(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn) {
    info->integrand = fn;
    return _functional_run(v, info, g, functional_mapintegrandforelement, false);
}

static value _functional_total(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn) {
    info->integrand = fn;
    return _functional_run(v, info, g, functional_sumintegrand, false);
}

static value _functional_gradient(vm *v, functional_mapinfo *info, grade g, functional_gradient *fn, symmetrybhvr sym) {
    info->grad = fn;
    info->sym = sym;
    return _functional_run(v, info, g, functional_mapgradient, true);
}

static value _functional_numericalgradient(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn, symmetrybhvr sym) {
    info->integrand = fn;
    info->sym = sym;
    return _functional_run(v, info, g, functional_mapnumericalgradient, true);
}

static value _functional_hessian(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn) {
    info->integrand = fn;
    return _functional_run(v, info, g, functional_mapnumericalhessian, true);
}

/** Validates the arguments provided to a functional
 * @param[in] v - vm
 * @param[in] nargs - number of arguments
 * @param[in] args - the arguments
 * @param[out] info - mapinfo block  */
bool functional_validateargs(vm *v, int nargs, value *args, functional_mapinfo *info) {
    functional_clearmapinfo(info);

    for (unsigned int i=0; i<nargs; i++) {
        if (MORPHO_ISMESH(MORPHO_GETARG(args,i))) {
            info->mesh = MORPHO_GETMESH(MORPHO_GETARG(args,i));
        } else if (MORPHO_ISSELECTION(MORPHO_GETARG(args,i))) {
            info->sel = MORPHO_GETSELECTION(MORPHO_GETARG(args,i));
        } else if (MORPHO_ISFIELD(MORPHO_GETARG(args,i))) {
            info->field = MORPHO_GETFIELD(MORPHO_GETARG(args,i));
            if (info->field) info->mesh = (info->field->mesh); // Retrieve the mesh from the field
        } else if (MORPHO_ISINTEGER(MORPHO_GETARG(args,i))) {
            info->id = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args,i));
        }
    }


    if (info->mesh) return true;
    MORPHO_FAIL(v, FUNCTIONAL_ARGS);
}

bool functional_readgrade(objectinstance *self, grade *g) {
    value val=MORPHO_NIL;
    if (!objectinstance_getpropertyinterned(self, functional_gradeproperty, &val) ||
        !MORPHO_ISINTEGER(val)) return false;
    *g = MORPHO_GETINTEGERVALUE(val);
    return true;
}

void functional_setgrade(objectinstance *self, grade g) {
    objectinstance_setproperty(self, functional_gradeproperty, MORPHO_INTEGER(g));
}


/* **********************************************************************
 * Common routines
 * ********************************************************************** */

/** Internal function to count the number of elements */
static bool functional_countelements(vm *v, objectmesh *mesh, grade g, int *n, objectsparse **s) {
    /* How many elements? */
    if (g==MESH_GRADE_VERTEX) {
        *n=mesh->vert->ncols;
    } else {
        *s=mesh_getconnectivityelement(mesh, 0, g);
        if (*s) {
            *n=(*s)->ccs.ncols; // Number of elements
        } else MORPHO_FAILVARGS(v, FUNC_ELNTFND, g);
    }
    return true;
}

/** Call the optional start hook once per user-facing map. Also verifies the mesh
    provides the requested grade. */
bool functional_startmap(vm *v, functional_mapinfo *info) {
    int n=0;
    objectsparse *s=NULL;
    if (info->mesh && !functional_countelements(v, info->mesh, info->g, &n, &s)) return false;
    if (!info->start) return true;
    return (*info->start) (v, info);
}

/** Call the optional end hook once per user-facing map. */
bool functional_endmap(vm *v, functional_mapinfo *info) {
    if (!info->end) return true;
    return (*info->end) (v, info);
}

/** Run start, then a map callback, then end. End is called whenever start succeeded. */
bool functional_runmap(vm *v, functional_mapinfo *info, functional_mapcallback *mapfn, value *out) {
    if (!functional_startmap(v, info)) return false;
    bool ok = (*mapfn) (v, info, out);
    if (!functional_endmap(v, info)) ok = false;
    return ok;
}

static int functional_symmetryimagelistfn(const void *a, const void *b) {
    elementid i=*(elementid *) a; elementid j=*(elementid *) b;
    return (int) i-j;
}

/** Gets a list of all image elements (those that map onto a target element)
 * @param[in] mesh - the mesh
 * @param[in] g - grade to look up
 * @param[in] sort - whether to sort othe results
 * @param[out] ids - varray is filled with image element ids */
void functional_symmetryimagelist(objectmesh *mesh, grade g, bool sort, varray_elementid *ids) {
    objectsparse *conn=mesh_getconnectivityelement(mesh, g, g);

    ids->count=0; // Initialize the varray

    if (conn) {
        int i,j;
        void *ctr=sparsedok_loopstart(&conn->dok);

        while (sparsedok_loop(&conn->dok, &ctr, &i, &j)) {
            varray_elementidwrite(ids, j);
        }

        if (sort) qsort(ids->data, ids->count, sizeof(elementid), functional_symmetryimagelistfn);
    }
}

/** Sums forces on symmetry vertices
 * @param[in] mesh - mesh object
 * @param frc - force object; updated if symmetries are present. */
bool functional_symmetrysumforces(objectmesh *mesh, objectmatrix *frc) {
    objectsparse *s=mesh_getconnectivityelement(mesh, 0, 0); // Checking for vertex symmetries

    if (s) {
        int i,j;
        void *ctr = sparsedok_loopstart(&s->dok);
        double *fi, *fj, fsum[mesh->dim];

        while (sparsedok_loop(&s->dok, &ctr, &i, &j)) {
            if (matrix_getcolumnptr(frc, i, &fi)==LINALGERR_OK &&
                matrix_getcolumnptr(frc, j, &fj)==LINALGERR_OK) {

                for (unsigned int k=0; k<mesh->dim; k++) fsum[k]=fi[k]+fj[k];
                if (matrix_setcolumnptr(frc, i, fsum)!=LINALGERR_OK) return false;
                if (matrix_setcolumnptr(frc, j, fsum)!=LINALGERR_OK) return false; 
            }
        }
    }

    return s;
}

bool functional_inlist(varray_elementid *list, elementid id) {
    for (unsigned int i=0; i<list->count; i++) if (list->data[i]==id) return true;
    return false;
}

bool functional_containsvertex(int nv, int *vid, elementid id) {
    for (unsigned int i=0; i<nv; i++) if (vid[i]==id) return true;
    return false;
}

/* **********************************************************************
 * Map functions
 * ********************************************************************** */

/** Kahan summation over a list */
double functional_sumlist(double *list, unsigned int nel) {
    double sum=0.0, c=0.0, y,t;

    for (unsigned int i=0; i<nel; i++) {
        y=list[i]-c;
        t=sum+y;
        c=(t-sum)-y;
        sum=t;
    }

    return sum;
}

/* **********************************************************************
 * Task-based map functions
 * ********************************************************************** */

threadpool functional_pool;
bool functional_poolinitialized;

/** Gradient function */
typedef bool (functional_mapfn) (vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, void *out);

/** Optionally process results from mapfn */
typedef bool (functional_processfn) (void *task);

/** Work to be done is divided into "tasks" which are then dispatched to the threadpool for execution. */
typedef struct {
    elementid start, end; /* Start and end indices for the task */
    elementid id; /* Current element id */
    elementid nel; /* Current element id */
    
    varray_elementid *skip; /* Sorted list of element ids to skip; set to NULL if not needed */
    unsigned int sindx;
    
    grade g; /* Grade of element */
    objectsparse *conn; /* Connectivity matrix */
    
    functional_mapfn *mapfn; /* Map function */
    functional_processfn *processfn; /* Post process results */
    
    vm *v; /* Virtual machine in use */
    objectmesh *mesh; /* Mesh in use */
    objectfield *field; /* Field in use */
    objectselection *selection; /* Selection to use if any */
    void *ref; /* Ref as an opaque pointer */
    
    void *result; /* Result of individual element as an opaque pointer */
    void *out; /* Overall output as an opaque pointer */
    bool usesubkernel; /* True if v is a worker subkernel */
    _MORPHO_PADDING;
} functional_task;

/* Initialize a task structure */
void functionaltask_init(functional_task *task, elementid start, elementid end, functional_mapinfo *info)  {
    task->start=start;
    task->end=end;
    task->nel=0;
    task->id=0;
    task->g=(info ? info->g : 0);
    
    task->skip=NULL;
    task->sindx=0;
    
    task->conn=NULL;
    
    task->mapfn=NULL;
    task->processfn=NULL;
    
    task->mesh=(info ? info->mesh : NULL);
    task->field=(info ? info->field : NULL);
    task->selection=(info ? info->sel : NULL);
    
    task->v=NULL;
    task->usesubkernel=false;
    task->ref=(info ? info->ref : NULL);
    task->out=NULL;
    task->result=NULL;
}

/** Check if we should skip element id */
bool functional_checkskip(functional_task *task) {
    if ((task->skip) &&
        (task->sindx<task->skip->count) &&
        task->skip->data[task->sindx]==task->id) {
        task->sindx++;
        return true;
    }
    return false;
}

/** Worker function to map a function over elements */
bool functional_mapfn_elements(void *arg) {
    functional_task *task = (functional_task *) arg;
    dictionary *selected=NULL;
    elementid *vid=&task->id; /* Will hold element definition */
    int nv=1; /* Number of vertices per element; default to 1  */
    
    if (task->selection) {
        selected=&task->selection->selected[task->g];
        if (selected->count==0) return true;
    }
    
    // Loop over required elements
    for (elementid i=task->start; i<task->end; i++) {
        if (selected) {
            // Skip empty dictionary entries
            if (!MORPHO_ISINTEGER(selected->contents[i].key)) continue;
            
            // Fetch the element id from the dictionary
            task->id = MORPHO_GETINTEGERVALUE(selected->contents[i].key);
        } else task->id = i;
        
        // Skip this element if it's an image element
        if (functional_checkskip(task)) continue;
        
        // Fetch element definition
        if (task->conn) {
            if (!sparseccs_getrowindices(&task->conn->ccs, task->id, &nv, &vid)) return false;
        }
        
        // Perform the map function
        if (!(*task->mapfn) (task->v, task->mesh, task->id, nv, vid, task->ref, task->result)) return false;
        
        // Perform post-processing if needed
        if (task->processfn) if (!(*task->processfn) (task)) return false;
        
        // Temporary objects on worker VMs must not accumulate across elements
        if (task->usesubkernel) vm_cleansubkernel(task->v);
    }
    return true;
}

/** Execute tasks on the calling thread */
bool functional_serialmap(int ntasks, functional_task *tasks) {
    for (int i=0; i<ntasks; i++) {
        if (!functional_mapfn_elements((void *) &tasks[i])) return false;
    }
    return true;
}

/** Dispatch tasks to the threadpool */
bool functional_parallelmap(int ntasks, functional_task *tasks) {
    int nthreads = morpho_threadnumber();
    if (nthreads<1) return functional_serialmap(ntasks, tasks);
    
    if (!functional_poolinitialized) {
        functional_poolinitialized=threadpool_init(&functional_pool, nthreads);
        if (!functional_poolinitialized) return false;
    }
    
    for (int i=0; i<ntasks; i++) {
        threadpool_add_task(&functional_pool, functional_mapfn_elements, (void *) &tasks[i]);
    }
    return threadpool_fence(&functional_pool);
}

/** Map over prepared tasks, using a threadpool only when worker threads are available */
bool functional_map(int ntasks, functional_task *tasks) {
    if (ntasks<1) return true;
    if (ntasks==1 || morpho_threadnumber()<1) return functional_serialmap(ntasks, tasks);
    return functional_parallelmap(ntasks, tasks);
}

/** Number of map tasks: one on the serial path, otherwise the worker count */
static int functional_ntasks(void) {
    int n=morpho_threadnumber();
    return (n<1 ? 1 : n);
}

/** Calculate bin sizes */
void functional_binbounds(int nel, int nbins, int *binbounds) {
    int binsizes[nbins+1];
    
    int defsize = nel / nbins;
    for (int i=0; i<nbins; i++) binsizes[i]=defsize;
    
    int rem = nel % nbins;
    while (rem>0) {
        for (int i=0; i<nbins && rem>0; i++) { binsizes[i]++; rem--; }
    }
    
    int bindx=0;
    for (int i=0; i<=nbins; i++) {
        binbounds[i]=bindx;
        bindx+=binsizes[i];
    }
}

/** Prepare tasks for submitting
 * @param[in] v - Virtual machine to use
 * @param[in] info - Info structure with functional information
 * @param[in] ntask - Number of tasks
 * @param[out] task - Task structures updated
 * @param[out] imageids - Updated to include symmetry image ids */
int functional_preparetasks(vm *v, functional_mapinfo *info, int ntask, functional_task *task, varray_elementid *imageids) {
    int nel=0;
    objectsparse *conn=NULL; // The associated connectivity matrix if any
    
    if (ntask<1) {
        varray_elementidclear(imageids);
        return false;
    }
    
    /* Work out the number of elements */
    if (!functional_countelements(v, info->mesh, info->g, &nel, &conn)) {
        varray_elementidclear(imageids);
        return false;
    }
    
    int cmax=nel;
    if (info->sel) {
        cmax=info->sel->selected[info->g].capacity;
    }
    
    int bins[ntask+1];
    functional_binbounds(cmax, ntask, bins);
    
    /* Ensure all mesh topology matrices have CCS */
    int maxgrade=mesh_maxgrade(info->mesh);
    for (int i=0; i<=maxgrade; i++) {
        for (int j=0; j<=maxgrade; j++) {
            objectsparse *s = mesh_getconnectivityelement(info->mesh, i, j);
            if (s) sparse_checkformat(s, SPARSE_CCS, true, false);
        }
    }
    
    /* Find any image elements so they can be skipped */
    functional_symmetryimagelist(info->mesh, info->g, true, imageids);
    if (info->field) field_addpool(info->field);
    
    vm *subkernels[ntask];
    if (ntask==1) {
        subkernels[0]=v; /* Serial maps reuse the calling VM */
    } else if (!vm_subkernels(v, ntask, subkernels)) {
        varray_elementidclear(imageids);
        MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    }

    if (ntask>1 && !functional_poolinitialized) {
        functional_poolinitialized=threadpool_init(&functional_pool, ntask);
        if (!functional_poolinitialized) {
            for (int i=0; i<ntask; i++) vm_releasesubkernel(subkernels[i]);
            varray_elementidclear(imageids);
            MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
        }
    }
    
    /** Initialize task structures */
    for (int i=0; i<ntask; i++) {
        functionaltask_init(task+i, bins[i], bins[i+1], info); // Setup the task
        
        task[i].v=subkernels[i];
        task[i].usesubkernel=(ntask>1);
        task[i].nel=nel;
        task[i].conn=conn;
        if (imageids->count>0) task[i].skip=imageids;
    }
    
    return true;
}

/** Cleans up task structures after executing them. */
void functional_cleanuptasks(vm *v, int ntask, functional_task *task, varray_elementid *imageids) {
    for (int i=0; i<ntask; i++) {
        if (task[i].v!=v) vm_releasesubkernel(task[i].v);
    }
    varray_elementidclear(imageids);
}

/* ----------------------------
 * Sum integrands
 * ---------------------------- */

/* Structure to store intermediate results for Kahan summation */
typedef struct {
    double result;
    double c;
    double sum;
    _MORPHO_PADDING;
} functional_sumintermediate;

/** Perform Kahan summation for total */
bool functional_sumintegrandprocessfn(void *arg) {
    functional_task *task = (functional_task *) arg;
    functional_sumintermediate *ks = (functional_sumintermediate *) task->out;
    double y=ks->result-ks->c;
    double t=ks->sum+y;
    ks->c=(t-ks->sum)-y;
    ks->sum=t; // Kahan summation
    return true;
}

/** Sum the integrand, mapping over integrand function */
bool functional_sumintegrand(vm *v, functional_mapinfo *info, value *out) {
    int ntask=functional_ntasks();
    
    functional_task task[ntask];
    
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    
    functional_sumintermediate sums[ntask];
    
    for (int i=0; i<ntask; i++) {
        task[i].mapfn=(functional_mapfn *) info->integrand;
        task[i].processfn=functional_sumintegrandprocessfn;
        
        task[i].result=(void *) &sums[i].result;
        task[i].out=(void *) &sums[i];
        sums[i].c=0.0; sums[i].sum=0.0;
    }
    
    bool success=functional_map(ntask, task);
    if (success) { // Sum up the results from each task...
        double sumlist[ntask];
        for (int i=0; i<ntask; i++) sumlist[i]=sums[i].sum;
    
        *out = MORPHO_FLOAT(functional_sumlist(sumlist, ntask));
    }

    functional_cleanuptasks(v, ntask, task, &imageids);
    return success;
}

/* ----------------------------
 * Map integrands
 * ---------------------------- */

/** Calculate the integrand at a particular element
 * @param[in] v - virtual machine in use
 * @param[in] info - map info
 * @param[out] out - a matrix of integrand values
 * @returns true on success, false otherwise. Error reporting through VM. */
bool functional_mapintegrandforelement(vm *v, functional_mapinfo *info, value *out) {
    objectmesh *mesh = info->mesh;
    grade g = info->g;
    elementid id = info->id;
    functional_integrand *integrand = info->integrand;
    void *ref = info->ref;
    objectsparse *s=NULL;
    bool ret=false;
    int n=0;
    
    /* How many elements? */
    if (!functional_countelements(v, mesh, g, &n, &s)) return false;
    if (id<0 || id>=n) MORPHO_FAIL(v, VM_OUTOFBOUNDS);
    
    int vertexid; // Use this if looping over grade 0
    int *vid=(g==0 ? &vertexid : NULL),
        nv=(g==0 ? 1 : 0); // The vertex indices
    if (s) sparseccs_getrowindices(&s->ccs, id, &nv, &vid);
    else vertexid=id;

    double result=0.0;
    if (vid && nv>0) {
        if (! (*integrand) (v, mesh, id, nv, vid, ref, &result)) {
            return false;
        }
    }
    *out = MORPHO_FLOAT(result);
    ret=true;
    
    return ret;
}

/** Set relevant matrix element to the result of the integrand */
bool functional_mapintegrandprocessfn(void *arg) {
    functional_task *task = (functional_task *) arg;
    objectmatrix *new = (objectmatrix *) task->out;
    matrix_setelement(new, 0, task->id, *(double *) task->result);
    return true;
}

/** Map integrand function, storing the results in a matrix */
bool functional_mapintegrand(vm *v, functional_mapinfo *info, value *out) {
    int success=false;
    int ntask=functional_ntasks();
    functional_task task[ntask];
    functional_sumintermediate sums[ntask];
    
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    objectmatrix *new = NULL;
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    
    if (task[0].nel>0) {
        new=matrix_new(1, task[0].nel, true);
        if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_mapintegrand_cleanup; }
    }
    
    for (int i=0; i<ntask; i++) {
        task[i].mapfn=(functional_mapfn *) info->integrand;
        task[i].processfn=functional_mapintegrandprocessfn;
    
        task[i].result=(void *) &sums[i].result;
        task[i].out=(void *) new;
    }
    
    if (!functional_map(ntask, task)) goto functional_mapintegrand_cleanup;
    
    success=true;
    *out = MORPHO_OBJECT(new);
    
functional_mapintegrand_cleanup:
    if (!success && new) object_free((object *) new);
    
    functional_cleanuptasks(v, ntask, task, &imageids);
    return success;
}

/* ----------------------------
 * Map gradients
 * ---------------------------- */

/** Compute the gradient */
bool functional_mapgradient(vm *v, functional_mapinfo *info, value *out) {
    int success=false;
    int ntask=functional_ntasks();
    functional_task task[ntask];
    
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    objectmatrix *new[ntask];
    for (int i=0; i<ntask; i++) new[i]=NULL;
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    
    /* Create output matrix */
    for (int i=0; i<ntask; i++) {
        // Create one per thread
        new[i]=matrix_new(info->mesh->vert->nrows, info->mesh->vert->ncols, true);
        if (!new[i]) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_mapgradient_cleanup; }
        
        task[i].mapfn=(functional_mapfn *) info->grad;
        task[i].result=(void *) new[i];
    }
    
    if (!functional_map(ntask, task)) goto functional_mapgradient_cleanup;
    
    /* Then add up all the matrices */
    for (int i=1; i<ntask; i++) matrix_axpy(1.0, new[i], new[0]);
    
    // Use symmetry actions
    if (info->sym==SYMMETRY_ADD) functional_symmetrysumforces(info->mesh, new[0]);
    
    success=true;
    
    // ...and return the result
    *out = MORPHO_OBJECT(new[0]);
    
functional_mapgradient_cleanup:
    for (int i=1; i<ntask; i++) if (new[i]) object_free((object *) new[i]);
    if (!success && new[0]) object_free((object *) new[0]);
    
    functional_cleanuptasks(v, ntask, task, &imageids);
    return success;
}

/* ----------------------------
 * Map numerical gradients
 * ---------------------------- */

/** Computes the gradient of element eid with respect to vertex i */
bool functional_numericalgrad(vm *v, objectmesh *mesh, elementid eid, elementid i, int nv, int *vid, functional_integrand *integrand, void *ref, objectmatrix *frc) {
    double f0,fp,fm,x0,eps=1e-6;
    
    // Loop over coordinates
    for (unsigned int k=0; k<mesh->dim; k++) {
        matrix_getelement(frc, k, i, &f0);

        matrix_getelement(mesh->vert, k, i, &x0);
        
        eps=functional_fdstepsize(x0, 1);
        matrix_setelement(mesh->vert, k, i, x0+eps);
        if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fp)) return false;
        matrix_setelement(mesh->vert, k, i, x0-eps);
        if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fm)) return false;
        matrix_setelement(mesh->vert, k, i, x0);

        matrix_setelement(frc, k, i, f0+(fp-fm)/(2*eps));
    }
    
    return true;
}

/** Computes the gradient of element id with respect to its constituent vertices and any dependencies */
bool functional_numericalgradientmapfn(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, void *out) {
    bool success=true;
    functional_mapinfo *info=(functional_mapinfo *) ref;
    
    for (int i=0; i<nv; i++) {
        if (!functional_numericalgrad(v, mesh, id, vid[i], nv, vid, info->integrand, info->ref, out)) return false;
    }
    
    // Now handle dependencies
    if (info->dependencies) {
        varray_elementid dependencies;
        varray_elementidinit(&dependencies);
        
        // Get list of vertices this element depends on
        if ((info->dependencies) (info, id, &dependencies)) {
            for (int j=0; j<dependencies.count; j++) {
                if (functional_containsvertex(nv, vid, dependencies.data[j])) continue;
                if (!functional_numericalgrad(v, mesh, id, dependencies.data[j], nv, vid, info->integrand, info->ref, out)) success=false;
            }
        }
        
        varray_elementidclear(&dependencies);
    }
    
    return success;
}

/** Compute the gradient numerically */
bool functional_mapnumericalgradient(vm *v, functional_mapinfo *info, value *out) {
    int success=false;
    int ntask=functional_ntasks();
    functional_task task[ntask];
    
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    objectmatrix *new[ntask]; // Create an output matrix for each thread
    objectmesh meshclones[ntask]; // Shallow clones with private vertex matrices (parallel only)
    for (int i=0; i<ntask; i++) {
        new[i]=NULL;
        meshclones[i].vert=NULL;
    }
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    
    for (int i=0; i<ntask; i++) {
        // Create one output matrix per thread
        new[i]=matrix_new(info->mesh->vert->nrows, info->mesh->vert->ncols, true);
        if (!new[i]) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_mapgradient_cleanup; }
        
        // Serial maps perturb the original vertices in place; clone only for workers
        if (ntask>1) {
            objectmatrix *vert=matrix_clone(info->mesh->vert);
            if (!vert) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_mapgradient_cleanup; }
            meshclones[i]=*info->mesh;
            meshclones[i].vert=vert;
            task[i].mesh=&meshclones[i];
        }
        
        task[i].ref=(void *) info; // Use this to pass the info structure
        task[i].mapfn=functional_numericalgradientmapfn;
        task[i].result=(void *) new[i];
    }
    
    if (!functional_map(ntask, task)) goto functional_mapgradient_cleanup;
    
    /* Then add up all the matrices */
    for (int i=1; i<ntask; i++) matrix_axpy(1.0, new[i], new[0]);
    
    success=true;
    
    // Use symmetry actions
    if (info->sym==SYMMETRY_ADD) functional_symmetrysumforces(info->mesh, new[0]);
    
    // ...and return the result
    *out = MORPHO_OBJECT(new[0]);
    
functional_mapgradient_cleanup:
    // Free the temporary copies of the vertex matrices
    for (int i=0; i<ntask; i++) if (meshclones[i].vert) object_free((object *) meshclones[i].vert);
    // Free spare output matrices
    for (int i=1; i<ntask; i++) if (new[i]) object_free((object *) new[i]);
    if (!success && new[0]) object_free((object *) new[0]);
    
    functional_cleanuptasks(v, ntask, task, &imageids);
    
    return success;
}

/* ----------------------------
 * Map numerical field gradients
 * ---------------------------- */

/** Computes the field gradient of element eid with respect to field grade g element i */
bool functional_numericalfieldgrad(vm *v, objectmesh *mesh, elementid eid, objectfield *field, grade g, elementid i, int nv, int *vid, functional_integrand *integrand, void *ref, objectfield *grad) {
    double fr,fl,eps=1e-6;
    
    /* Loop over dofs in field entry */
    for (int j=0; j<field->psize*field->dof[g]; j++) {
        int k=(field->offset[g]+i)*field->psize*field->dof[g]+j;
        
        double f0=field->data.elements[k];
        
        eps=functional_fdstepsize(f0, 1);
        
        field->data.elements[k]+=eps;
        if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fr)) return false;

        field->data.elements[k]=f0-eps;
        if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fl)) return false;

        field->data.elements[k]=f0;

        grad->data.elements[k]+=(fr-fl)/(2*eps);
    }
    
    return true;
}

/** Computes the field gradient of element eid with respect to a single field dof. */
bool functional_numericalfieldgradentry(vm *v, objectmesh *mesh, elementid eid, objectfield *field, grade g, elementid i, int indx, int nv, int *vid, functional_integrand *integrand, void *ref, objectfield *grad) {
    double fr, fl, eps=1e-6;
    unsigned int nentries;
    double *entry, *gentry;

    if (!field_getelementaslist(field, g, i, indx, &nentries, &entry)) return false;
    if (!field_getelementaslist(grad, g, i, indx, &nentries, &gentry)) return false;

    for (unsigned int j=0; j<nentries; j++) {
        double f0=entry[j];
        eps=functional_fdstepsize(f0, 1);

        entry[j]=f0+eps;
        if (!(*integrand)(v, mesh, eid, nv, vid, ref, &fr)) return false;

        entry[j]=f0-eps;
        if (!(*integrand)(v, mesh, eid, nv, vid, ref, &fl)) return false;

        entry[j]=f0;
        gentry[j]+=(fr-fl)/(2*eps);
    }

    return true;
}

typedef struct {
    functional_mapinfo *info;
    objectfield *field;
    functional_integrand *integrand;
    fespace *disc;
    objectsparse *conn;
    void *ref;
} functional_numericalfieldgradientref;

/** Reevaluate a dependent element for one perturbed field dof. */
static bool functional_numericalremotefieldgrad(vm *v, functional_numericalfieldgradientref *tref, elementid remoteid, grade g, elementid i, int indx, objectfield *grad) {
    int nv=(tref->info->g==0 ? 1 : 0), *vid=(tref->info->g==0 ? &remoteid : NULL);

    if (tref->conn) {
        if (!sparseccs_getrowindices(&tref->conn->ccs, remoteid, &nv, &vid)) return false;
    }

    return functional_numericalfieldgradentry(v, tref->info->mesh, remoteid, tref->field, g, i, indx, nv, vid, tref->integrand, tref->ref, grad);
}

/** Computes the gradient of element id with respect to its constituent vertices and any dependencies */
bool functional_numericalfieldgradientmapfn(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, void *out) {
    functional_numericalfieldgradientref *tref=(functional_numericalfieldgradientref *) ref;
    
    if (tref->disc) {
        int nnodes=tref->disc->nnodes;
        fieldindx findx[nnodes];
        
        if (fespace_doftofieldindx(tref->field, tref->disc, nv, vid, findx)) {
            for (int k=0; k<nnodes; k++) {
                if (!functional_numericalfieldgradentry(v, mesh, id, tref->field, findx[k].g, findx[k].id, findx[k].indx, nv, vid, tref->integrand, tref->ref, out)) return false;

                if (tref->info->dependencies) {
                    varray_elementid dependencies;
                    varray_elementidinit(&dependencies);
                    if ((tref->info->dependencies)(tref->info, id, &dependencies)) {
                        for (int j=0; j<dependencies.count; j++) {
                            if (!functional_numericalremotefieldgrad(v, tref, dependencies.data[j], findx[k].g, findx[k].id, findx[k].indx, out)) {
                                varray_elementidclear(&dependencies);
                                return false;
                            }
                        }
                    }
                    varray_elementidclear(&dependencies);
                }
            }
        }
        
    } else {
        for (elementid k=0; k<nv; k++) {
            if (!functional_numericalfieldgrad(v, mesh, id, tref->field, MESH_GRADE_VERTEX, vid[k], nv, vid, tref->integrand, tref->ref, out)) return false;

            if (tref->info->dependencies) {
                varray_elementid dependencies;
                varray_elementidinit(&dependencies);
                if ((tref->info->dependencies)(tref->info, id, &dependencies)) {
                    for (int j=0; j<dependencies.count; j++) {
                        if (!functional_numericalremotefieldgrad(v, tref, dependencies.data[j], MESH_GRADE_VERTEX, vid[k], 0, out)) {
                            varray_elementidclear(&dependencies);
                            return false;
                        }
                    }
                }
                varray_elementidclear(&dependencies);
            }
        }
    }
    
    return true;
}

/** Compute the field gradient numerically */
bool functional_mapnumericalfieldgradient(vm *v, functional_mapinfo *info, value *out) {
    int success=false;
    int ntask=functional_ntasks();
    functional_task task[ntask];
    
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    objectfield *new[ntask]; // Create an output field for each thread
    objectfield *fieldclones[ntask]; // Clones of the field for each worker
    functional_numericalfieldgradientref tref[ntask];
    for (int i=0; i<ntask; i++) {
        new[i]=NULL; fieldclones[i]=NULL;
        tref[i].ref=NULL;
    }
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    
    for (int i=0; i<ntask; i++) {
        // Create one output field per thread
        new[i]=object_newfield(info->mesh, info->field->prototype, info->field->fnspc, info->field->dof);
        if (!new[i]) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_mapfieldgradient_cleanup; }
        field_zero(new[i]);
        
        tref[i].info=info;
        tref[i].integrand=info->integrand;
        tref[i].conn=mesh_getconnectivityelement(info->mesh, 0, info->g);
        tref[i].disc=NULL;
        
        // Serial maps perturb the original field in place; clone only for workers
        if (ntask>1) {
            fieldclones[i]=field_clone(info->field);
            tref[i].field=fieldclones[i];
            tref[i].ref=info->ref;
            if (info->cloneref) {
                tref[i].ref=(info->cloneref) (info->ref, info->field, fieldclones[i]);
            } else UNREACHABLE("Functional calls numericalfieldgradient but doesn't provide cloneref");
        } else {
            tref[i].field=info->field;
            tref[i].ref=info->ref;
        }
        
        if (MORPHO_ISFESPACE(tref[i].field->fnspc)) {
            tref[i].disc=MORPHO_GETFESPACE(tref[i].field->fnspc)->fespace;
            if (info->g<tref[i].disc->grade) {
                if (!fespace_lower(tref[i].disc, info->g, &tref[i].disc)) {
                    functional_fespaceerror(v, tref[i].field, info->g);
                    goto functional_mapfieldgradient_cleanup;
                }
            }
        }
        
        task[i].ref=(void *) &tref[i]; // Use this to pass the info structure
        task[i].mapfn=functional_numericalfieldgradientmapfn;
        task[i].result=(void *) new[i];
    }
    
    if (!functional_map(ntask, task)) goto functional_mapfieldgradient_cleanup;
    
    /* Then add up all the fields */
    for (int i=1; i<ntask; i++) matrix_axpy(1.0, &new[i]->data, &new[0]->data);
    
    success=true;
    
    // ...and return the result
    *out = MORPHO_OBJECT(new[0]);
    
functional_mapfieldgradient_cleanup:
    for (int i=0; i<ntask; i++) {
        if (!fieldclones[i]) continue;
        
        // Free any cloned references
        if (info->freeref) (info->freeref) (tref[i].ref);
        else if (info->cloneref) MORPHO_FREE(tref[i].ref);
        
        // Free the temporary copies of the fields
        object_free((object *) fieldclones[i]);
    }
    
    // Free spare output matrices
    for (int i=1; i<ntask; i++) if (new[i]) object_free((object *) new[i]);
    if (!success && new[0]) object_free((object *) new[0]);
    
    functional_cleanuptasks(v, ntask, task, &imageids);
    
    return success;
}

typedef struct {
    functional_mapinfo *info;
    objectfield *field;
    fespace *disc;
    objectsparse *conn;
    objectsparse *parentvertices;
    jumpref *ref;
} jump_numericalfieldgradientref;

static bool jump_getelementvertices(objectsparse *conn, grade g, elementid id, int *nv, int **vid) {
    if (conn) return sparseccs_getrowindices(&conn->ccs, id, nv, vid);
    if (g==0) {
        *nv=1;
        *vid=&id;
        return true;
    }
    return false;
}

static bool jump_collectparentfieldentries(jump_numericalfieldgradientref *tref, elementid interfaceid, fieldindx *findxout, int *nentries) {
    jumpref *ref=tref->ref;
    int nparents=0, *parents=NULL;
    int plusnv=0, minusnv=0, *plusvid=NULL, *minusvid=NULL;

    *nentries=0;
    if (!jump_getadjacentparents(ref, interfaceid, &nparents, &parents)) return false;
    if (nparents!=2) return true;

    elementid plusid, minusid;
    jump_orderparents(parents, &plusid, &minusid);

    if (!mesh_getconnectivity(tref->parentvertices, plusid, &plusnv, &plusvid)) return false;
    if (!mesh_getconnectivity(tref->parentvertices, minusid, &minusnv, &minusvid)) return false;

    fieldindx findx[tref->disc->nnodes];
    if (!fespace_doftofieldindx(tref->field, tref->disc, plusnv, plusvid, findx)) return false;
    for (int i=0; i<tref->disc->nnodes; i++) {
        bool found=false;
        for (int j=0; j<*nentries; j++) {
            if (findxout[j].g==findx[i].g && findxout[j].id==findx[i].id && findxout[j].indx==findx[i].indx) { found=true; break; }
        }
        if (!found) {
            findxout[*nentries]=findx[i];
            (*nentries)++;
        }
    }

    if (!fespace_doftofieldindx(tref->field, tref->disc, minusnv, minusvid, findx)) return false;
    for (int i=0; i<tref->disc->nnodes; i++) {
        bool found=false;
        for (int j=0; j<*nentries; j++) {
            if (findxout[j].g==findx[i].g && findxout[j].id==findx[i].id && findxout[j].indx==findx[i].indx) { found=true; break; }
        }
        if (!found) {
            findxout[*nentries]=findx[i];
            (*nentries)++;
        }
    }

    return true;
}

static bool jump_numericalfieldgradientmapfn(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, void *out) {
    jump_numericalfieldgradientref *tref=(jump_numericalfieldgradientref *) ref;
    fieldindx findx[2*tref->disc->nnodes];
    int nentries=0;

    if (!jump_collectparentfieldentries(tref, id, findx, &nentries)) return false;

    for (int k=0; k<nentries; k++) {
        if (!functional_numericalfieldgradentry(v, mesh, id, tref->field, findx[k].g, findx[k].id, findx[k].indx, nv, vid, tref->info->integrand, tref->ref, out)) return false;

        if (tref->info->dependencies) {
            varray_elementid dependencies;
            varray_elementidinit(&dependencies);
            if ((tref->info->dependencies)(tref->info, id, &dependencies)) {
                for (int j=0; j<dependencies.count; j++) {
                    int rnv=0, *rvid=NULL;
                    if (!jump_getelementvertices(tref->conn, tref->info->g, dependencies.data[j], &rnv, &rvid)) {
                        varray_elementidclear(&dependencies);
                        return false;
                    }
                    if (!functional_numericalfieldgradentry(v, mesh, dependencies.data[j], tref->field, findx[k].g, findx[k].id, findx[k].indx, rnv, rvid, tref->info->integrand, tref->ref, out)) {
                        varray_elementidclear(&dependencies);
                        return false;
                    }
                }
            }
            varray_elementidclear(&dependencies);
        }
    }

    return true;
}

static bool functional_mapjumpnumericalfieldgradient(vm *v, functional_mapinfo *info, objectsparse *parentvertices, void *baseref, value *out) {
    int success=false;
    int ntask=functional_ntasks();
    functional_task task[ntask];

    varray_elementid imageids;
    varray_elementidinit(&imageids);

    objectfield *new[ntask];
    objectfield *fieldclones[ntask];
    jump_numericalfieldgradientref tref[ntask];
    for (int i=0; i<ntask; i++) {
        new[i]=NULL; fieldclones[i]=NULL;
        tref[i].ref=NULL;
    }

    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;

    for (int i=0; i<ntask; i++) {
        new[i]=object_newfield(info->mesh, info->field->prototype, info->field->fnspc, info->field->dof);
        if (!new[i]) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_mapjumpfieldgradient_cleanup; }
        field_zero(new[i]);

        tref[i].info=info;
        tref[i].conn=mesh_getconnectivityelement(info->mesh, 0, info->g);
        tref[i].parentvertices=parentvertices;
        tref[i].disc=NULL;

        // Serial maps perturb the original field in place; clone only for workers
        if (ntask>1) {
            fieldclones[i]=field_clone(info->field);
            tref[i].field=fieldclones[i];
            tref[i].ref=(jumpref *) ((info->cloneref) ? (info->cloneref)(baseref, info->field, fieldclones[i]) : baseref);
        } else {
            tref[i].field=info->field;
            tref[i].ref=(jumpref *) baseref;
        }
        if (!tref[i].ref) goto functional_mapjumpfieldgradient_cleanup;
        if (MORPHO_ISFESPACE(tref[i].field->fnspc)) tref[i].disc=MORPHO_GETFESPACE(tref[i].field->fnspc)->fespace;
        if (!tref[i].disc) goto functional_mapjumpfieldgradient_cleanup;

        task[i].ref=(void *) &tref[i];
        task[i].mapfn=jump_numericalfieldgradientmapfn;
        task[i].result=(void *) new[i];
    }

    if (!functional_map(ntask, task)) goto functional_mapjumpfieldgradient_cleanup;

    for (int i=1; i<ntask; i++) matrix_axpy(1.0, &new[i]->data, &new[0]->data);;
    
    success=true;
    *out=MORPHO_OBJECT(new[0]);

functional_mapjumpfieldgradient_cleanup:
    for (int i=0; i<ntask; i++) {
        if (fieldclones[i]) {
            if (info->freeref && tref[i].ref) (info->freeref)(tref[i].ref);
            else if (info->cloneref && tref[i].ref) MORPHO_FREE(tref[i].ref);
            object_free((object *) fieldclones[i]);
        }
        if (i>0 && new[i]) object_free((object *) new[i]);
    }
    if (!success && new[0]) object_free((object *) new[0]);
    functional_cleanuptasks(v, ntask, task, &imageids);
    return success;
}

/* ----------------------------
 * Map numerical hessians
 * ---------------------------- */

/** Adds a value to an element of a sparse matrix */
bool functional_sparseaccumulate(objectsparse *A, int i, int j, double val) {
    double f0 = 0.0;
    value h0;
    if (sparsedok_get(&A->dok, i, j, &h0)) {
        if (!morpho_valuetofloat(h0, &f0)) return false;
    }
    
    sparsedok_insert(&A->dok, i, j, MORPHO_FLOAT(f0+val));
    return true;
}

/** Computes the contribution to the hessian of element eid with respect to vertices i and j */
bool functional_numericalhess(vm *v, objectmesh *mesh, elementid eid, elementid i, elementid j, int nv, int *vid, functional_integrand *integrand, void *ref, objectsparse *hess) {
    double x0,y0,epsx=1e-4,epsy=1e-4;
    
    for (unsigned int k=0; k<mesh->dim; k++) { // Loop over coordinates in vertex i
        matrix_getelement(mesh->vert, k, i, &x0);
        epsx=functional_fdstepsize(x0, 2);
        
        if (i==j) { // Use a special formula for diagonal elements
            double fc, fr, fl;
            if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fc)) return false;
            
            matrix_setelement(mesh->vert, k, i, x0+epsx);
            if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fr)) return false;
            
            matrix_setelement(mesh->vert, k, i, x0-epsx);
            if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fl)) return false;
            
            matrix_setelement(mesh->vert, k, i, x0); // Restore vertex to original position
            
            functional_sparseaccumulate(hess, i*mesh->dim+k, i*mesh->dim+k, (fr + fl - 2*fc)/(epsx*epsx));
        }
        
        // Loop over coordinates in vertex j
        for (unsigned int l=0; //(i==j? k+1 : k); // Detect whether we're in an off diagonal block
             l<mesh->dim; l++) {
            if (i==j && k==l) continue;
            double fll,frr,flr,frl;
            
            matrix_getelement(mesh->vert, l, j, &y0);
            epsy=functional_fdstepsize(y0, 2);
            
            matrix_setelement(mesh->vert, k, i, x0+epsx);
            matrix_setelement(mesh->vert, l, j, y0+epsy);
            if (!(*integrand) (v, mesh, eid, nv, vid, ref, &frr)) return false;
            
            matrix_setelement(mesh->vert, l, j, y0-epsy);
            if (!(*integrand) (v, mesh, eid, nv, vid, ref, &frl)) return false;
            
            matrix_setelement(mesh->vert, k, i, x0-epsx);
            if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fll)) return false;
            
            matrix_setelement(mesh->vert, l, j, y0+epsy);
            if (!(*integrand) (v, mesh, eid, nv, vid, ref, &flr)) return false;
            
            matrix_setelement(mesh->vert, k, i, x0); // Restore vertices to original position
            matrix_setelement(mesh->vert, l, j, y0);
            
            functional_sparseaccumulate(hess, i*mesh->dim+k, j*mesh->dim+l, (frr + fll - flr - frl)/(4*epsx*epsy));
            //functional_sparseaccumulate(hess, j*mesh->dim+l, i*mesh->dim+k, (frr + fll - flr - frl)/(4*epsx*epsy));
        }
    }
    
    return true;
}

/** Computes the gradient of element id with respect to its constituent vertices and any dependencies */
bool functional_numericalhessianmapfn(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, void *out) {
    bool success=true;
    functional_mapinfo *info=(functional_mapinfo *) ref;
    
    // TODO: Exploit symmetry of hessian to reduce work
    
    for (int i=0; i<nv; i++) {
        for (int j=0; j<nv; j++) {
            if (!functional_numericalhess(v, mesh, id, vid[i], vid[j], nv, vid, info->integrand, info->ref, out)) return false;
        }
    }
    
    // Now handle dependencies
    if (info->dependencies) {
        varray_elementid dependencies;
        varray_elementidinit(&dependencies);
        
        // Get list of vertices this element depends on
        if ((info->dependencies) (info, id, &dependencies)) {
            for (int i=0; i<dependencies.count; i++) {
                for (int j=0; j<dependencies.count; j++) {
                    if (functional_containsvertex(nv, vid, dependencies.data[i]) && functional_containsvertex(nv, vid, dependencies.data[j])) continue;
                    if (!functional_numericalhess(v, mesh, id, dependencies.data[i], dependencies.data[j], nv, vid, info->integrand, info->ref, out)) success=false;
                }
            }
        }
        
        varray_elementidclear(&dependencies);
    }
    
    return success;
}

static int _sparsecmp(const void *a, const void *b) {
    objectsparse *aa = *(objectsparse **) a;
    objectsparse *bb = *(objectsparse **) b;
    return bb->dok.dict.count - aa->dok.dict.count;
}

/** Compute the hessian numerically */
bool functional_mapnumericalhessian(vm *v, functional_mapinfo *info, value *out) {
    int success=false;
    int ntask=functional_ntasks();
    functional_task task[ntask];
    
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    objectsparse *new[ntask]; // Create an output matrix for each thread
    objectmesh meshclones[ntask]; // Shallow clones with private vertex matrices (parallel only)
    
    for (int i=0; i<ntask; i++) {
        new[i]=NULL;
        meshclones[i].vert=NULL;
    }
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    
    for (int i=0; i<ntask; i++) {
        int N = info->mesh->dim*mesh_nvertices(info->mesh);
        
        // Create one output matrix per thread
        new[i]=object_newsparse(&N, &N);
        if (!new[i]) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_maphessian_cleanup; }
        
        // Serial maps perturb the original vertices in place; clone only for workers
        if (ntask>1) {
            objectmatrix *vert=matrix_clone(info->mesh->vert);
            if (!vert) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_maphessian_cleanup; }
            meshclones[i]=*info->mesh;
            meshclones[i].vert=vert;
            task[i].mesh=&meshclones[i];
        }
        
        task[i].ref=(void *) info; // Use this to pass the info structure
        task[i].mapfn=functional_numericalhessianmapfn;
        task[i].result=(void *) new[i];
    }
    
    if (!functional_map(ntask, task)) goto functional_maphessian_cleanup;
    
    qsort(new, ntask, sizeof(objectsparse *), _sparsecmp);
    
    if (!sparse_checkformat(new[0], SPARSE_CCS, true, true)) {
        morpho_runtimeerror(v, SPARSE_OPFAILEDERR);
        goto functional_maphessian_cleanup;
    }
    
    /* Then add up all the matrices */
    for (int i=1; i<ntask; i++) {
        if (!new[i]->dok.dict.count) continue;
        objectsparse out = MORPHO_STATICSPARSE();
        sparsedok_init(&out.dok);
        sparseccs_init(&out.ccs);
        
        if (sparse_add(new[0], new[i], 1.0, 1.0, &out)==SPARSE_OK) {
            sparseccs_clear(&new[0]->ccs);
            new[0]->ccs = out.ccs;
        } else {
            morpho_runtimeerror(v, SPARSE_OPFAILEDERR);
            goto functional_maphessian_cleanup;
        }
    }
    success=true;
    
    // Use symmetry actions
    //if (info->sym==SYMMETRY_ADD) functional_symmetrysumforces(info->mesh, new[0]);
    
    sparsedok_clear(&new[0]->dok); // Remove dok info
    
    // ...and return the result
    *out = MORPHO_OBJECT(new[0]);
    
functional_maphessian_cleanup:
    // Free the temporary copies of the vertex matrices
    for (int i=0; i<ntask; i++) if (meshclones[i].vert) object_free((object *) meshclones[i].vert);
    // Free spare output matrices
    for (int i=1; i<ntask; i++) if (new[i]) object_free((object *) new[i]);
    if (!success && new[0]) object_free((object *) new[0]);
    
    functional_cleanuptasks(v, ntask, task, &imageids);
    
    return success;
}


/* **********************************************************************
 * Common library functions
 * ********************************************************************** */

/** Calculate the difference of two vectors */
void functional_vecadd(unsigned int n, double *a, double *b, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=a[i]+b[i];
}

/** Add with scale */
void functional_vecaddscale(unsigned int n, double *a, double lambda, double *b, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=a[i]+lambda*b[i];
}

/** Calculate the difference of two vectors */
void functional_vecsub(unsigned int n, double *a, double *b, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=a[i]-b[i];
}

/** Scale a vector */
void functional_vecscale(unsigned int n, double lambda, double *a, double *out) {
    for (unsigned int i=0; i<n; i++) out[i]=lambda*a[i];
}

/** Calculate the norm of a vector */
double functional_vecnorm(unsigned int n, double *a) {
    return cblas_dnrm2(n, a, 1);
}

/** Dot product of two vectors */
double functional_vecdot(unsigned int n, double *a, double *b) {
    return cblas_ddot(n, a, 1, b, 1);
}

/** 3D cross product  */
void functional_veccross(double *a, double *b, double *out) {
    out[0]=a[1]*b[2]-a[2]*b[1];
    out[1]=a[2]*b[0]-a[0]*b[2];
    out[2]=a[0]*b[1]-a[1]*b[0];
}

/** 2D cross product  */
void functional_veccross2d(double *a, double *b, double *out) {
    *out=a[0]*b[1]-a[1]*b[0];
}

bool length_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out);
bool area_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out);
bool volume_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out);

/** Calculate element size */
bool functional_elementsize(vm *v, objectmesh *mesh, grade g, elementid id, int nv, int *vid, double *out) {
    switch (g) {
        case 1: return length_integrand(v, mesh, id, nv, vid, NULL, out);
        case 2: return area_integrand(v, mesh, id, nv, vid, NULL, out);
        case 3: return volume_integrand(v, mesh, id, nv, vid, NULL, out);
    }
    return false;
}

bool length_gradient_scale(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc, double scale);
bool area_gradient_scale(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc, double scale);
bool volume_gradient_scale(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc, double scale);

bool length_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc);
bool area_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc);
bool volume_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc);

/** Calculate a scaled element gradient */
bool functional_elementgradient_scale(vm *v, objectmesh *mesh, grade g, elementid id, int nv, int *vid, objectmatrix *frc, double scale) {
    switch (g) {
        case 1: return length_gradient_scale(v, mesh, id, nv, vid, NULL, frc, scale);
        case 2: return area_gradient_scale(v, mesh, id, nv, vid, NULL, frc, scale);
        case 3: return volume_gradient_scale(v, mesh, id, nv, vid, NULL, frc, scale);
    }
    return false;
}

/** Calculate element gradient */
bool functional_elementgradient(vm *v, objectmesh *mesh, grade g, elementid id, int nv, int *vid, objectmatrix *frc) {
    switch (g) {
        case 1: return length_gradient(v, mesh, id, nv, vid, NULL, frc);
        case 2: return area_gradient(v, mesh, id, nv, vid, NULL, frc);
        case 3: return volume_gradient(v, mesh, id, nv, vid, NULL, frc);
    }
    return false;
}

/* ----------------------------------------------
 * Length
 * ---------------------------------------------- */

/** Calculate area */
bool length_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    if (nv!=2) { *out=0; return true; }
    double *x[nv], s0[mesh->dim];
    for (int j=0; j<nv; j++) matrix_getcolumnptr(mesh->vert, vid[j], &x[j]);

    functional_vecsub(mesh->dim, x[1], x[0], s0);

    *out=functional_vecnorm(mesh->dim, s0);
    return true;
}

/** Calculate scaled gradient */
bool length_gradient_scale(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc, double scale) {
    double *x[nv], s0[mesh->dim], norm;
    for (int j=0; j<nv; j++) matrix_getcolumnptr(mesh->vert, vid[j], &x[j]);

    functional_vecsub(mesh->dim, x[1], x[0], s0);
    norm=functional_vecnorm(mesh->dim, s0);
    if (norm<MORPHO_EPS) return true;
    
    if (matrix_addtocolumnptr(frc, vid[0], -1.0/norm*scale, s0)!=LINALGERR_OK) return false;
    if (matrix_addtocolumnptr(frc, vid[1], 1./norm*scale, s0)!=LINALGERR_OK) return false;

    return true;
}

/** Calculate gradient */
bool length_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    return length_gradient_scale(v, mesh, id, nv, vid, NULL, frc, 1.0);
}

FUNCTIONAL_INIT(Length, MESH_GRADE_LINE)
FUNCTIONAL_MD_INTEGRAND(Length, MESH_GRADE_LINE, length_integrand)
FUNCTIONAL_MD_TOTAL(Length, MESH_GRADE_LINE, length_integrand)
FUNCTIONAL_MD_GRADIENT(Length, MESH_GRADE_LINE, length_gradient, SYMMETRY_ADD)
FUNCTIONAL_MD_HESSIAN(Length, MESH_GRADE_LINE, length_integrand)

MORPHO_BEGINCLASS(Length)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", Length_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(Length),
FUNCTIONAL_MD_TOTAL_METHODS(Length),
FUNCTIONAL_MD_GRADIENT_METHODS(Length),
FUNCTIONAL_MD_HESSIAN_METHODS(Length)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Enclosed area
 * ---------------------------------------------- */

/** Calculate area enclosed */
bool areaenclosed_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double *x[nv], cx[mesh->dim], normcx;
    for (int j=0; j<nv; j++) matrix_getcolumnptr(mesh->vert, vid[j], &x[j]);

    if (mesh->dim==2) {
        functional_veccross2d(x[0], x[1], cx);
        normcx=fabs(cx[0]);
    } else {
        functional_veccross(x[0], x[1], cx);
        normcx=functional_vecnorm(mesh->dim, cx);
    }

    *out=0.5*normcx;

    return true;
}

FUNCTIONAL_INIT(AreaEnclosed, MESH_GRADE_LINE)
FUNCTIONAL_MD_INTEGRAND(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_integrand)
FUNCTIONAL_MD_TOTAL(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_integrand)
FUNCTIONAL_MD_NUMERICALGRADIENT(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_integrand, SYMMETRY_ADD)
FUNCTIONAL_MD_HESSIAN(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_integrand)

MORPHO_BEGINCLASS(AreaEnclosed)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", AreaEnclosed_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(AreaEnclosed),
FUNCTIONAL_MD_TOTAL_METHODS(AreaEnclosed),
FUNCTIONAL_MD_GRADIENT_METHODS(AreaEnclosed),
FUNCTIONAL_MD_HESSIAN_METHODS(AreaEnclosed)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Area
 * ---------------------------------------------- */

/** Calculate area */
bool area_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    if (nv!=3) { *out=0; return true; }
    double *x[nv], s0[3], s1[3], cx[3];
    for (int j=0; j<3; j++) { s0[j]=0; s1[j]=0; cx[j]=0; }
    for (int j=0; j<nv; j++) matrix_getcolumnptr(mesh->vert, vid[j], &x[j]);

    functional_vecsub(mesh->dim, x[1], x[0], s0);
    functional_vecsub(mesh->dim, x[2], x[1], s1);

    functional_veccross(s0, s1, cx);
    *out=0.5*functional_vecnorm(3, cx);
    
    return true;
}

/** Calculate scaled gradient */
bool area_gradient_scale(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc, double scale) {
    double *x[nv], s0[3], s1[3], s01[3], s010[3], s011[3];
    double norm;
    for (int j=0; j<3; j++) { s0[j]=0; s1[j]=0; s01[j]=0; s010[j]=0; s011[j]=0; }
    for (int j=0; j<nv; j++) if (matrix_getcolumnptr(mesh->vert, vid[j], &x[j])!=LINALGERR_OK) return false;

    functional_vecsub(mesh->dim, x[1], x[0], s0);
    functional_vecsub(mesh->dim, x[2], x[1], s1);

    functional_veccross(s0, s1, s01);
    norm=functional_vecnorm(3, s01);
    if (norm<MORPHO_EPS) return true;

    functional_veccross(s01, s0, s010);
    functional_veccross(s01, s1, s011);

    if (matrix_addtocolumnptr(frc, vid[0], 0.5/norm*scale, s011)!=LINALGERR_OK) return false;
    if (matrix_addtocolumnptr(frc, vid[2], 0.5/norm*scale, s010)!=LINALGERR_OK) return false;

    functional_vecadd(mesh->dim, s010, s011, s0);

    if (matrix_addtocolumnptr(frc, vid[1], -0.5/norm*scale, s0)!=LINALGERR_OK) return false;

    return true;
}

/** Calculate gradient */
bool area_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    return area_gradient_scale(v, mesh, id, nv, vid, NULL, frc, 1.0);
}

FUNCTIONAL_INIT(Area, MESH_GRADE_AREA)
FUNCTIONAL_MD_INTEGRAND(Area, MESH_GRADE_AREA, area_integrand)
FUNCTIONAL_MD_TOTAL(Area, MESH_GRADE_AREA, area_integrand)
FUNCTIONAL_MD_GRADIENT(Area, MESH_GRADE_AREA, area_gradient, SYMMETRY_ADD)
FUNCTIONAL_MD_HESSIAN(Area, MESH_GRADE_AREA, area_integrand)

MORPHO_BEGINCLASS(Area)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", Area_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(Area),
FUNCTIONAL_MD_TOTAL_METHODS(Area),
FUNCTIONAL_MD_GRADIENT_METHODS(Area),
FUNCTIONAL_MD_HESSIAN_METHODS(Area)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Enclosed volume
 * ---------------------------------------------- */

/** Calculate enclosed volume */
bool volumeenclosed_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double *x[nv], cx[mesh->dim];
    for (int j=0; j<nv; j++) if (matrix_getcolumnptr(mesh->vert, vid[j], &x[j])!=LINALGERR_OK) return false;

    functional_veccross(x[0], x[1], cx);

    *out=fabs(functional_vecdot(mesh->dim, cx, x[2]))/6.0;
    return true;
}

/** Calculate gradient */
bool volumeenclosed_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    double *x[nv], cx[mesh->dim], dot;
    for (int j=0; j<nv; j++) if (matrix_getcolumnptr(mesh->vert, vid[j], &x[j])!=LINALGERR_OK) return false;

    functional_veccross(x[0], x[1], cx);
    dot=functional_vecdot(mesh->dim, cx, x[2]);
    if (fabs(dot)<=DBL_MIN) MORPHO_FAIL(v, VOLUMEENCLOSED_ZERO);
    
    dot/=fabs(dot);

    if (matrix_addtocolumnptr(frc, vid[2], dot/6.0, cx)!=LINALGERR_OK) return false;

    functional_veccross(x[1], x[2], cx);
    if (matrix_addtocolumnptr(frc, vid[0], dot/6.0, cx)!=LINALGERR_OK) return false;

    functional_veccross(x[2], x[0], cx);
    if (matrix_addtocolumnptr(frc, vid[1], dot/6.0, cx)!=LINALGERR_OK) return false;

    return true;
}

FUNCTIONAL_INIT(VolumeEnclosed, MESH_GRADE_AREA)
FUNCTIONAL_MD_INTEGRAND(VolumeEnclosed, MESH_GRADE_AREA, volumeenclosed_integrand)
FUNCTIONAL_MD_TOTAL(VolumeEnclosed, MESH_GRADE_AREA, volumeenclosed_integrand)
FUNCTIONAL_MD_GRADIENT(VolumeEnclosed, MESH_GRADE_AREA, volumeenclosed_gradient, SYMMETRY_ADD)
FUNCTIONAL_MD_HESSIAN(VolumeEnclosed, MESH_GRADE_AREA, volumeenclosed_integrand)

MORPHO_BEGINCLASS(VolumeEnclosed)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", VolumeEnclosed_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(VolumeEnclosed),
FUNCTIONAL_MD_TOTAL_METHODS(VolumeEnclosed),
FUNCTIONAL_MD_GRADIENT_METHODS(VolumeEnclosed),
FUNCTIONAL_MD_HESSIAN_METHODS(VolumeEnclosed)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Volume
 * ---------------------------------------------- */

/** Calculate enclosed volume */
bool volume_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double *x[nv], s10[mesh->dim], s20[mesh->dim], s30[mesh->dim], cx[mesh->dim];
    for (int j=0; j<nv; j++) matrix_getcolumnptr(mesh->vert, vid[j], &x[j]);

    functional_vecsub(mesh->dim, x[1], x[0], s10);
    functional_vecsub(mesh->dim, x[2], x[0], s20);
    functional_vecsub(mesh->dim, x[3], x[0], s30);

    functional_veccross(s20, s30, cx);

    *out=fabs(functional_vecdot(mesh->dim, s10, cx))/6.0;
    return true;
}

/** Calculate scaled gradient */
bool volume_gradient_scale(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc, double scale) {
    double *x[nv], s10[mesh->dim], s20[mesh->dim], s30[mesh->dim];
    double s31[mesh->dim], s21[mesh->dim], cx[mesh->dim], uu;
    for (int j=0; j<nv; j++) matrix_getcolumnptr(mesh->vert, vid[j], &x[j]);

    functional_vecsub(mesh->dim, x[1], x[0], s10);
    functional_vecsub(mesh->dim, x[2], x[0], s20);
    functional_vecsub(mesh->dim, x[3], x[0], s30);
    functional_vecsub(mesh->dim, x[3], x[1], s31);
    functional_vecsub(mesh->dim, x[2], x[1], s21);

    functional_veccross(s20, s30, cx);
    uu=functional_vecdot(mesh->dim, s10, cx);
    uu=(uu>0 ? 1.0 : -1.0);

    if (matrix_addtocolumnptr(frc, vid[1], uu/6.0*scale, cx)!=LINALGERR_OK) return false;

    functional_veccross(s31, s21, cx);
    if (matrix_addtocolumnptr(frc, vid[0], uu/6.0*scale, cx)!=LINALGERR_OK) return false;

    functional_veccross(s30, s10, cx);
    if (matrix_addtocolumnptr(frc, vid[2], uu/6.0*scale, cx)!=LINALGERR_OK) return false;

    functional_veccross(s10, s20, cx);
    if (matrix_addtocolumnptr(frc, vid[3], uu/6.0*scale, cx)!=LINALGERR_OK) return false;

    return true;
}

/** Calculate gradient */
bool volume_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    return volume_gradient_scale(v, mesh, id, nv, vid, NULL, frc, 1.0);
}

FUNCTIONAL_INIT(Volume, MESH_GRADE_VOLUME)
FUNCTIONAL_MD_INTEGRAND(Volume, MESH_GRADE_VOLUME, volume_integrand)
FUNCTIONAL_MD_TOTAL(Volume, MESH_GRADE_VOLUME, volume_integrand)
FUNCTIONAL_MD_GRADIENT(Volume, MESH_GRADE_VOLUME, volume_gradient, SYMMETRY_ADD)
FUNCTIONAL_MD_HESSIAN(Volume, MESH_GRADE_VOLUME, volume_integrand)

MORPHO_BEGINCLASS(Volume)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", Volume_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(Volume),
FUNCTIONAL_MD_TOTAL_METHODS(Volume),
FUNCTIONAL_MD_GRADIENT_METHODS(Volume),
FUNCTIONAL_MD_HESSIAN_METHODS(Volume)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Scalar potential
 * ---------------------------------------------- */

static value scalarpotential_functionproperty;
static value scalarpotential_gradfunctionproperty;

typedef struct {
    value fn;
} scalarpotentialref;

bool scalarpotential_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, scalarpotentialref *ref) {
    ref->fn=MORPHO_NIL;
    return (objectinstance_getpropertyinterned(self, scalarpotential_functionproperty, &ref->fn) &&
            MORPHO_ISCALLABLE(ref->fn));
}

/** Evaluate the scalar potential */
bool scalarpotential_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double *x;
    value fn = ((scalarpotentialref *) ref)->fn;
    value args[mesh->dim];
    value ret;

    if (matrix_getcolumnptr(mesh->vert, id, &x)!=LINALGERR_OK) return false;
    for (int i=0; i<mesh->dim; i++) args[i]=MORPHO_FLOAT(x[i]);

    if (morpho_call(v, fn, mesh->dim, args, &ret)) {
        return morpho_valuetofloat(ret, out);
    }

    return false;
}

/** Evaluate the gradient of the scalar potential */
bool scalarpotential_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    double *x;
    value fn = ((scalarpotentialref *) ref)->fn;
    value args[mesh->dim];
    value ret;

    if (matrix_getcolumnptr(mesh->vert, id, &x)!=LINALGERR_OK) return false;
    for (int i=0; i<mesh->dim; i++) args[i]=MORPHO_FLOAT(x[i]);

    if (morpho_call(v, fn, mesh->dim, args, &ret)) {
        if (MORPHO_ISMATRIX(ret)) {
            objectmatrix *vf=MORPHO_GETMATRIX(ret);

            if (vf->nrows*vf->ncols==frc->nrows) {
                return (matrix_addtocolumnptr(frc, id, 1.0, vf->elements)==LINALGERR_OK);
            }
        }
    }

    return false;
}

value ScalarPotential_init(vm *v, int nargs, value *args) {
    functional_setgrade(MORPHO_GETINSTANCE(MORPHO_SELF(args)), MESH_GRADE_VERTEX);
    return MORPHO_NIL;
}

value ScalarPotential_init__fn(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    functional_setgrade(self, MESH_GRADE_VERTEX);
    objectinstance_setproperty(self, scalarpotential_functionproperty, MORPHO_GETARG(args, 0));
    return MORPHO_NIL;
}

value ScalarPotential_init__fn_fn(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    functional_setgrade(self, MESH_GRADE_VERTEX);
    objectinstance_setproperty(self, scalarpotential_functionproperty, MORPHO_GETARG(args, 0));
    objectinstance_setproperty(self, scalarpotential_gradfunctionproperty, MORPHO_GETARG(args, 1));
    return MORPHO_NIL;
}

static bool _scalarpotential_bindref(vm *v, objectinstance *self, functional_mapinfo *info, scalarpotentialref *ref) {
    if (info->g < 0) info->g = MESH_GRADE_VERTEX;
    if (!scalarpotential_prepareref(self, info->mesh, info->g, info->sel, ref)) {
        morpho_runtimeerror(v, SCALARPOTENTIAL_FNCLLBL);
        return false;
    }
    info->ref = ref;
    info->integrand = scalarpotential_integrand;
    return true;
}

static value _scalarpotential_integrand(vm *v, objectinstance *self, functional_mapinfo *info) {
    scalarpotentialref ref;
    if (!_scalarpotential_bindref(v, self, info, &ref)) return MORPHO_NIL;
    return _functional_run(v, info, MESH_GRADE_VERTEX, functional_mapintegrand, true);
}

value ScalarPotential_integrand__mesh(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    _functional_mapinfo(&info, MORPHO_GETMESH(MORPHO_GETARG(args, 0)), NULL, NULL);
    return _scalarpotential_integrand(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info);
}

value ScalarPotential_integrand__mesh_sel(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    _functional_mapinfo(&info, MORPHO_GETMESH(MORPHO_GETARG(args, 0)), MORPHO_GETSELECTION(MORPHO_GETARG(args, 1)), NULL);
    return _scalarpotential_integrand(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info);
}

static value _scalarpotential_integrand_elem(vm *v, objectinstance *self, functional_mapinfo *info) {
    scalarpotentialref ref;
    if (!_scalarpotential_bindref(v, self, info, &ref)) return MORPHO_NIL;
    return _functional_run(v, info, MESH_GRADE_VERTEX, functional_mapintegrandforelement, false);
}

value ScalarPotential_integrand__mesh_int(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    _functional_mapinfo(&info, MORPHO_GETMESH(MORPHO_GETARG(args, 0)), NULL, NULL);
    info.id = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    return _scalarpotential_integrand_elem(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info);
}

value ScalarPotential_integrand__mesh_int_int(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    _functional_mapinfo(&info, MORPHO_GETMESH(MORPHO_GETARG(args, 0)), NULL, NULL);
    info.g = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1));
    info.id = MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 2));
    return _scalarpotential_integrand_elem(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info);
}

static value _scalarpotential_total(vm *v, objectinstance *self, functional_mapinfo *info) {
    scalarpotentialref ref;
    if (!_scalarpotential_bindref(v, self, info, &ref)) return MORPHO_NIL;
    return _functional_run(v, info, MESH_GRADE_VERTEX, functional_sumintegrand, false);
}

value ScalarPotential_total__mesh(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    _functional_mapinfo(&info, MORPHO_GETMESH(MORPHO_GETARG(args, 0)), NULL, NULL);
    return _scalarpotential_total(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info);
}

value ScalarPotential_total__mesh_sel(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    _functional_mapinfo(&info, MORPHO_GETMESH(MORPHO_GETARG(args, 0)), MORPHO_GETSELECTION(MORPHO_GETARG(args, 1)), NULL);
    return _scalarpotential_total(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info);
}

static value _scalarpotential_gradient(vm *v, objectinstance *self, functional_mapinfo *info) {
    scalarpotentialref ref;
    value fn;

    if (objectinstance_getpropertyinterned(self, scalarpotential_gradfunctionproperty, &fn)) {
        if (!MORPHO_ISCALLABLE(fn)) {
            morpho_runtimeerror(v, SCALARPOTENTIAL_FNCLLBL);
            return MORPHO_NIL;
        }
        ref.fn = fn;
        info->ref = &ref;
        info->grad = scalarpotential_gradient;
        return _functional_run(v, info, MESH_GRADE_VERTEX, functional_mapgradient, true);
    }

    if (!_scalarpotential_bindref(v, self, info, &ref)) return MORPHO_NIL;
    return _functional_run(v, info, MESH_GRADE_VERTEX, functional_mapnumericalgradient, true);
}

value ScalarPotential_gradient__mesh(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    _functional_mapinfo(&info, MORPHO_GETMESH(MORPHO_GETARG(args, 0)), NULL, NULL);
    return _scalarpotential_gradient(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info);
}

value ScalarPotential_gradient__mesh_sel(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    _functional_mapinfo(&info, MORPHO_GETMESH(MORPHO_GETARG(args, 0)), MORPHO_GETSELECTION(MORPHO_GETARG(args, 1)), NULL);
    return _scalarpotential_gradient(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info);
}

static value _scalarpotential_hessian(vm *v, objectinstance *self, functional_mapinfo *info) {
    scalarpotentialref ref;
    if (!_scalarpotential_bindref(v, self, info, &ref)) return MORPHO_NIL;
    return _functional_run(v, info, MESH_GRADE_VERTEX, functional_mapnumericalhessian, true);
}

value ScalarPotential_hessian__mesh(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    _functional_mapinfo(&info, MORPHO_GETMESH(MORPHO_GETARG(args, 0)), NULL, NULL);
    return _scalarpotential_hessian(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info);
}

value ScalarPotential_hessian__mesh_sel(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    _functional_mapinfo(&info, MORPHO_GETMESH(MORPHO_GETARG(args, 0)), MORPHO_GETSELECTION(MORPHO_GETARG(args, 1)), NULL);
    return _scalarpotential_hessian(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), &info);
}

#define SP_MAPFLAGS (MORPHO_FN_REENTRANT|FUNCTIONAL_MD_MAPFLAGS)
#define SP_TOTALFLAGS (MORPHO_FN_REENTRANT|FUNCTIONAL_MD_TOTALFLAGS)
#define SP_ELEMFLAGS (MORPHO_FN_REENTRANT|FUNCTIONAL_MD_ELEMFLAGS)

MORPHO_BEGINCLASS(ScalarPotential)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", ScalarPotential_init, MORPHO_FN_MUTATES),
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Callable)", ScalarPotential_init__fn, MORPHO_FN_MUTATES),
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Callable, Callable)", ScalarPotential_init__fn_fn, MORPHO_FN_MUTATES),

MORPHO_METHOD_SIGNATURE(FUNCTIONAL_INTEGRAND_METHOD, "Matrix (Mesh)", ScalarPotential_integrand__mesh, SP_MAPFLAGS),
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_INTEGRAND_METHOD, "Matrix (Mesh, Selection)", ScalarPotential_integrand__mesh_sel, SP_MAPFLAGS),
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_INTEGRAND_METHOD, "Float (Mesh, Int)", ScalarPotential_integrand__mesh_int, SP_ELEMFLAGS),
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_INTEGRAND_METHOD, "Float (Mesh, Int, Int)", ScalarPotential_integrand__mesh_int_int, SP_ELEMFLAGS),

MORPHO_METHOD_SIGNATURE(FUNCTIONAL_TOTAL_METHOD, "Float (Mesh)", ScalarPotential_total__mesh, SP_TOTALFLAGS),
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_TOTAL_METHOD, "Float (Mesh, Selection)", ScalarPotential_total__mesh_sel, SP_TOTALFLAGS),

MORPHO_METHOD_SIGNATURE(FUNCTIONAL_GRADIENT_METHOD, "Matrix (Mesh)", ScalarPotential_gradient__mesh, SP_MAPFLAGS),
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_GRADIENT_METHOD, "Matrix (Mesh, Selection)", ScalarPotential_gradient__mesh_sel, SP_MAPFLAGS),

MORPHO_METHOD_SIGNATURE(FUNCTIONAL_HESSIAN_METHOD, "Sparse (Mesh)", ScalarPotential_hessian__mesh, SP_MAPFLAGS),
MORPHO_METHOD_SIGNATURE(FUNCTIONAL_HESSIAN_METHOD, "Sparse (Mesh, Selection)", ScalarPotential_hessian__mesh_sel, SP_MAPFLAGS)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Linear Elasticity
 * ---------------------------------------------- */

static value linearelasticity_referenceproperty;
static value linearelasticity_weightbyreferenceproperty;
static value linearelasticity_poissonproperty;

typedef struct {
    objectmesh *refmesh;
    grade grade;
    double lambda; // Lamé coefficients
    double mu;     //
} linearelasticityref;

/** Calculates the Gram matrix */
void linearelasticity_calculategram(objectmatrix *vert, int dim, int nv, int *vid, objectmatrix *gram) {
    int gdim=nv-1; // Dimension of Gram matrix
    double *x[nv], // Positions of vertices
            s[gdim][nv]; // Side vectors

    for (int j=0; j<nv; j++) matrix_getcolumnptr(vert, vid[j], &x[j]); // Get vertices
    for (int j=1; j<nv; j++) functional_vecsub(dim, x[j], x[0], s[j-1]); // u_i = X_i - X_0
    // <u_i, u_j>
    for (int i=0; i<nv-1; i++) for (int j=0; j<nv-1; j++) gram->elements[i+j*gdim]=functional_vecdot(dim, s[i], s[j]);
}

/** Calculate the linear elastic energy */
bool linearelasticity_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    double weight=0.0;
    linearelasticityref *info = (linearelasticityref *) ref;
    int gdim=nv-1; // Dimension of Gram matrix

    /* Construct static matrices */
    double gramrefel[gdim*gdim], gramdefel[gdim*gdim], qel[gdim*gdim], rel[gdim*gdim], cgel[gdim*gdim];
    objectmatrix gramref = MORPHO_STATICMATRIX(gramrefel, gdim, gdim); // Gram matrices
    objectmatrix gramdef = MORPHO_STATICMATRIX(gramdefel, gdim, gdim); //
    objectmatrix q = MORPHO_STATICMATRIX(qel, gdim, gdim); // Inverse of Gram in source domain
    objectmatrix r = MORPHO_STATICMATRIX(rel, gdim, gdim); // Intermediate calculations
    objectmatrix cg = MORPHO_STATICMATRIX(cgel, gdim, gdim); // Cauchy-Green strain tensor

    linearelasticity_calculategram(info->refmesh->vert, mesh->dim, nv, vid, &gramref);
    linearelasticity_calculategram(mesh->vert, mesh->dim, nv, vid, &gramdef);

    if (matrix_copy(&gramref, &q)!=LINALGERR_OK) return false;
    if (matrix_inverse(&q)!=LINALGERR_OK) return false;
    if (matrix_mul(&gramdef, &q, &r)!=LINALGERR_OK) return false;

    if (matrix_identity(&cg)!=LINALGERR_OK) return false;
    matrix_scale(&cg, -0.5);
    matrix_axpy(0.5, &r, &cg);         //  y <- alpha*x + y

    double trcg=0.0, trcgcg=0.0;
    matrix_trace(&cg, &trcg);
    
    matrix_mul(&cg, &cg, &r);
    matrix_trace(&r, &trcgcg);

    if (!functional_elementsize(v, info->refmesh, info->grade, id, nv, vid, &weight)) return false;

    *out=weight*(info->mu*trcgcg + 0.5*info->lambda*trcg*trcg);

    return true;
}

/** Prepares the reference structure from the LinearElasticity object's properties */
bool linearelasticity_prepareref(objectinstance *self, linearelasticityref *ref) {
    bool success=false;
    value refmesh=MORPHO_NIL;
    value grade=MORPHO_NIL;
    value poisson=MORPHO_NIL;

    if (objectinstance_getpropertyinterned(self, linearelasticity_referenceproperty, &refmesh) &&
        MORPHO_ISMESH(refmesh) &&
        objectinstance_getpropertyinterned(self, functional_gradeproperty, &grade) &&
        MORPHO_ISINTEGER(grade) &&
        objectinstance_getpropertyinterned(self, linearelasticity_poissonproperty, &poisson) &&
        MORPHO_ISNUMBER(poisson)) {
        ref->refmesh=MORPHO_GETMESH(refmesh);
        ref->grade=MORPHO_GETINTEGERVALUE(grade);

        double nu = MORPHO_GETFLOATVALUE(poisson);

        ref->mu=0.5/(1+nu);
        ref->lambda=nu/(1+nu)/(1-2*nu);
        success=true;
    }
    return success;
}

value LinearElasticity_init(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    /* First argument is the reference mesh */
    if (nargs>0) {
        if (MORPHO_ISMESH(MORPHO_GETARG(args, 0))) {
            objectinstance_setproperty(self, linearelasticity_referenceproperty, MORPHO_GETARG(args, 0));
            objectmesh *mesh = MORPHO_GETMESH(MORPHO_GETARG(args, 0));

            objectinstance_setproperty(self, functional_gradeproperty, MORPHO_INTEGER(mesh_maxgrade(mesh)));
            objectinstance_setproperty(self, linearelasticity_poissonproperty, MORPHO_FLOAT(0.3));
        } else morpho_runtimeerror(v, FUNCTIONAL_ARGS);
    } else morpho_runtimeerror(v, FUNCTIONAL_ARGS);

    /* Second (optional) argument is the grade to act on */
    if (nargs>1) {
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 1))) {
            objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), functional_gradeproperty, MORPHO_GETARG(args, 1));
        }
    }

    return MORPHO_NIL;
}

/** Integrand function */
value LinearElasticity_integrand(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    linearelasticityref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (linearelasticity_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), &ref)) {
            info.g = ref.grade;
            info.integrand = linearelasticity_integrand;
            info.ref = &ref;
            functional_runmap(v, &info, functional_mapintegrand, &out);
        } else morpho_runtimeerror(v, FUNCTIONAL_ARGS);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

/** Total function */
value LinearElasticity_total(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    linearelasticityref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (linearelasticity_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), &ref)) {
            info.g = ref.grade;
            info.integrand = linearelasticity_integrand;
            info.ref = &ref;
            functional_runmap(v, &info, functional_sumintegrand, &out);
        } else morpho_runtimeerror(v, FUNCTIONAL_ARGS);
    }
    return out;
}

/** Integrand function */
value LinearElasticity_gradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    linearelasticityref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (linearelasticity_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), &ref)) {
            info.g = ref.grade;
            info.integrand = linearelasticity_integrand;
            info.ref = &ref;
            info.sym = SYMMETRY_ADD;
            functional_runmap(v, &info, functional_mapnumericalgradient, &out);
        } else morpho_runtimeerror(v, FUNCTIONAL_ARGS);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

MORPHO_BEGINCLASS(LinearElasticity)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, LinearElasticity_init, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, LinearElasticity_integrand, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, LinearElasticity_total, MORPHO_FN_PUREFN|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, LinearElasticity_gradient, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED)
MORPHO_ENDCLASS

/* ----------------------------------------------
* Hydrogel
* ---------------------------------------------- */

static value hydrogel_aproperty;
static value hydrogel_bproperty;
static value hydrogel_cproperty;
static value hydrogel_dproperty;
static value hydrogel_phirefproperty;
static value hydrogel_phi0property;

typedef struct {
    objectmesh *refmesh;
    grade grade;
    double a, b, c, d, phiref; // Hydrogel coefficients
    value phi0; // Can be a number or a field. (Ensuring flexibility for supplying a phi0 field in the future)
} hydrogelref;

/** Prepares the reference structure from the object's properties */
bool hydrogel_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, hydrogelref *ref) {
    bool success=false;
    value refmesh=MORPHO_NIL, grade=MORPHO_NIL, phi0=MORPHO_NIL;
    value a=MORPHO_NIL, b=MORPHO_NIL, c=MORPHO_NIL, d=MORPHO_NIL, phiref=MORPHO_NIL;

    if (objectinstance_getpropertyinterned(self, linearelasticity_referenceproperty, &refmesh) &&
        MORPHO_ISMESH(refmesh) &&
        objectinstance_getpropertyinterned(self, functional_gradeproperty, &grade) &&
        MORPHO_ISINTEGER(grade) &&
        objectinstance_getpropertyinterned(self, hydrogel_aproperty, &a) &&
        MORPHO_ISNUMBER(a) &&
        objectinstance_getpropertyinterned(self, hydrogel_bproperty, &b) &&
        MORPHO_ISNUMBER(b) &&
        objectinstance_getpropertyinterned(self, hydrogel_cproperty, &c) &&
        MORPHO_ISNUMBER(c) &&
        objectinstance_getpropertyinterned(self, hydrogel_dproperty, &d) &&
        MORPHO_ISNUMBER(d) &&
        objectinstance_getpropertyinterned(self, hydrogel_phirefproperty, &phiref) &&
        MORPHO_ISNUMBER(phiref) &&
        objectinstance_getpropertyinterned(self, hydrogel_phi0property, &phi0) &&
        (MORPHO_ISNUMBER(phi0) || MORPHO_ISFIELD(phi0))) {
        ref->refmesh=MORPHO_GETMESH(refmesh);
        ref->grade=MORPHO_GETINTEGERVALUE(grade);

        if (ref->grade<0) ref->grade=mesh_maxgrade(mesh);

        if (morpho_valuetofloat(a, &ref->a) &&
            morpho_valuetofloat(b, &ref->b) &&
            morpho_valuetofloat(c, &ref->c) &&
            morpho_valuetofloat(d, &ref->d) &&
            morpho_valuetofloat(phiref, &ref->phiref)) {
            ref->phi0 = phi0;
            success=true;
        }
    }
    return success;
}

/** Calculate the Hydrogel energy */
bool hydrogel_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    hydrogelref *info = (hydrogelref *) ref;
    value vphi0 = info->phi0;
    double V=0.0, V0=0.0, phi0=0.0;

    if (!functional_elementsize(v, info->refmesh, info->grade, id, nv, vid, &V0)) return false;
    if (!functional_elementsize(v, mesh, info->grade, id, nv, vid, &V)) return false;

    if (V0<1e-8) {
        morpho_runtimewarning(v, HYDROGEL_ZEEROREFELEMENT, id, V, V0);
    }

    if (fabs(V)<MORPHO_EPS) {
        *out = 0;
        return true;
    }

    // Determine phi0 either as a number or by looking up something in a field
    if (MORPHO_ISFIELD(info->phi0)) {
        objectfield *p = MORPHO_GETFIELD(info->phi0);
        if (!field_getelement(p, info->grade, id, 0, &vphi0)) MORPHO_FAILVARGS(v, HYDROGEL_FLDGRD, (unsigned int) info->grade);
    }
    if (MORPHO_ISNUMBER(vphi0)) {
        if (!morpho_valuetofloat(vphi0, &phi0)) return false;
    }

    double phi = phi0/(V/V0);
    double pr = info->phiref;
    if (phi<0 || 1-phi<0) {
        morpho_runtimewarning(v, HYDROGEL_BNDS, id, V, V0, phi, 1-phi);
    }

    if (phi>1-MORPHO_EPS) phi = 1-MORPHO_EPS;
    if (phi<MORPHO_EPS) phi = MORPHO_EPS;

    *out = (info->a * phi*log(phi) +
            info->b * (1-phi)*log(1-phi) +
            info->c * phi*(1-phi))*V +
            info->d * (log(pr/phi)/3.0 - pow((pr/phi), (2.0/3)) + 1.0)*V0;

    return true;
}

/** Calculate gradient */
bool hydrogel_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {

    hydrogelref *info = (hydrogelref *) ref;
    value vphi0 = info->phi0;
    double V=0.0, V0=0.0, phi0=0.0;

    if (!functional_elementsize(v, info->refmesh, info->grade, id, nv, vid, &V0)) return false;
    if (!functional_elementsize(v, mesh, info->grade, id, nv, vid, &V)) return false;

    if (V0<1e-8) {
        morpho_runtimewarning(v, HYDROGEL_ZEEROREFELEMENT, id, V, V0);
    }

    if (fabs(V)<MORPHO_EPS) return true;

    // Determine phi0 either as a number or by looking up something in a field
    if (MORPHO_ISFIELD(info->phi0)) {
        objectfield *p = MORPHO_GETFIELD(info->phi0);
        if (!field_getelement(p, info->grade, id, 0, &vphi0)) MORPHO_FAILVARGS(v, HYDROGEL_FLDGRD, (unsigned int) info->grade);
    }
    if (MORPHO_ISNUMBER(vphi0)) {
        if (!morpho_valuetofloat(vphi0, &phi0)) return false;
    }

    double phi = phi0/(V/V0);
    double pr = info->phiref;
    if (phi<0 || 1-phi<0) {
        morpho_runtimewarning(v, HYDROGEL_BNDS, id, V, V0, phi, 1-phi);
    }

    if (phi>1-MORPHO_EPS) phi = 1-MORPHO_EPS;
    if (phi<MORPHO_EPS) phi = MORPHO_EPS;

    double grad = (-info->a * phi +
            info->b * ( phi + log(1-phi) ) +
            info->c * phi*phi +
            info->d * (pr/phi0) * ((phi/pr)/3.0 - (2.0/3) * pow((phi/pr), (1.0/3)) ) );

    // Compute grad * element gradient
    if (!functional_elementgradient_scale(v, mesh, info->grade, id, nv, vid, frc, grad)) return false;

    return true;
}

/** Evaluate a gradient */
value Hydrogel_gradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    value out=MORPHO_NIL;
    hydrogelref ref;
    if (functional_validateargs(v, nargs, args, &info)) {
        if (hydrogel_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, -1, info.sel, &ref)) {
            info.g = ref.grade;
            info.grad = hydrogel_gradient;
            info.ref = &ref;
            info.sym = SYMMETRY_ADD;
            functional_runmap(v, &info, functional_mapgradient, &out);
        } else morpho_runtimeerror(v, FUNCTIONAL_ARGS);
    }

    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);

    return out;
}


value Hydrogel_init(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    int nfixed;
    value grade=MORPHO_INTEGER(-1);
    value a=MORPHO_NIL, b=MORPHO_NIL, c=MORPHO_NIL, d=MORPHO_NIL, phiref=MORPHO_NIL, phi0=MORPHO_NIL;

    if (builtin_options(v, nargs, args, &nfixed, 7,
                        hydrogel_aproperty, &a,
                        hydrogel_bproperty, &b,
                        hydrogel_cproperty, &c,
                        hydrogel_dproperty, &d,
                        hydrogel_phirefproperty, &phiref,
                        hydrogel_phi0property, &phi0,
                        functional_gradeproperty, &grade)) {

        objectinstance_setproperty(self, hydrogel_aproperty, a);
        objectinstance_setproperty(self, hydrogel_bproperty, b);
        objectinstance_setproperty(self, hydrogel_cproperty, c);
        objectinstance_setproperty(self, hydrogel_dproperty, d);
        objectinstance_setproperty(self, hydrogel_phirefproperty, phiref);
        objectinstance_setproperty(self, hydrogel_phi0property, phi0);
        objectinstance_setproperty(self, functional_gradeproperty, grade);

        if (nfixed==1 && MORPHO_ISMESH(MORPHO_GETARG(args, 0))) {
            objectinstance_setproperty(self, linearelasticity_referenceproperty, MORPHO_GETARG(args, 0));
            if (MORPHO_ISINTEGER(grade) && MORPHO_GETINTEGERVALUE(grade)<0) {
                objectinstance_setproperty(self, functional_gradeproperty, MORPHO_INTEGER(mesh_maxgrade(MORPHO_GETMESH(MORPHO_GETARG(args, 0)))));
            }
        } else morpho_runtimeerror(v, FUNCTIONAL_ARGS);
    } else morpho_runtimeerror(v, FUNCTIONAL_ARGS);

    return MORPHO_NIL;
}

FUNCTIONAL_METHOD(Hydrogel, integrand, (ref.grade), hydrogelref, hydrogel_prepareref, functional_mapintegrand, hydrogel_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)

FUNCTIONAL_METHOD(Hydrogel, total, (ref.grade), hydrogelref, hydrogel_prepareref, functional_sumintegrand, hydrogel_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)

MORPHO_BEGINCLASS(Hydrogel)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, Hydrogel_init, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, Hydrogel_integrand, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, Hydrogel_total, MORPHO_FN_PUREFN|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, Hydrogel_gradient, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Equielement
 * ---------------------------------------------- */

static value equielement_weightproperty;

typedef struct {
    grade grade;
    objectsparse *vtoel; // Connect vertices to elements
    objectsparse *eltov; // Connect elements to vertices
    objectmatrix *weight; // Weight field
    double mean;
} equielementref;

/** Prepares the reference structure from the Equielement object's properties */
bool equielement_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, equielementref *ref) {
    bool success=false;
    value grade=MORPHO_NIL;
    value weight=MORPHO_NIL;

    if (objectinstance_getpropertyinterned(self, functional_gradeproperty, &grade) &&
        MORPHO_ISINTEGER(grade) ) {
        ref->grade=MORPHO_GETINTEGERVALUE(grade);
        ref->weight=NULL;

        int maxgrade=mesh_maxgrade(mesh);
        if (ref->grade<0 || ref->grade>maxgrade) ref->grade = maxgrade;

        ref->vtoel=mesh_addconnectivityelement(mesh, ref->grade, 0);
        ref->eltov=mesh_addconnectivityelement(mesh, 0, ref->grade);

        if (ref->vtoel && ref->eltov) success=true;
    }

    if (objectinstance_getpropertyinterned(self, equielement_weightproperty, &weight) &&
        MORPHO_ISMATRIX(weight) ) {
        ref->weight=MORPHO_GETMATRIX(weight);
        if (ref->weight) {
            double sum[ref->weight->nvals];
            matrix_sum(ref->weight, sum);
            ref->mean = sum[0];
            ref->mean/=ref->weight->ncols;
        }
    }

    return success;
}


bool equielement_contains(varray_elementid *nbrs, elementid id) {
    for (unsigned int i=0; i<nbrs->count; i++) {
        if (nbrs->data[i]==id) return true;
    }
    return false;
}

/** Finds the points that a point depends on  */
bool equielement_dependencies(functional_mapinfo *info, elementid id, varray_elementid *out) {
    objectmesh *mesh = info->mesh;
    equielementref *eref = info->ref;
    bool success=false;
    varray_elementid nbrs;
    varray_elementidinit(&nbrs);

    // varray_elementidwrite(out, id); // EquiElement is a vertex element, and hence depends on itself
    
    if (mesh_findneighbors(mesh, MESH_GRADE_VERTEX, id, eref->grade, &nbrs)>0) {
        for (unsigned int i=0; i<nbrs.count; i++) {
            int nentries, *entries; // Get the vertices for this element
            if (!sparseccs_getrowindices(&eref->eltov->ccs, nbrs.data[i], &nentries, &entries)) goto equieleement_dependencies_cleanup;

            for (unsigned int j=0; j<nentries; j++) {
                if (entries[j]==id) continue;
                if (equielement_contains(out, entries[j])) continue;
                varray_elementidwrite(out, entries[j]);
            }
        }
    }
    success=true;

equieleement_dependencies_cleanup:
    varray_elementidclear(&nbrs);

    return success;
}

/** Calculate the equielement energy */
bool equielement_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *r, double *out) {
    equielementref *ref = (equielementref *) r;
    int nconn, *conn;

    if (sparseccs_getrowindices(&ref->vtoel->ccs, id, &nconn, &conn)) {
        if (nconn==1) { *out = 0; return true; }

        double size[nconn], mean=0.0, total=0.0;

        for (int i=0; i<nconn; i++) {
            int nv, *vid;
            sparseccs_getrowindices(&ref->eltov->ccs, conn[i], &nv, &vid);
            functional_elementsize(v, mesh, ref->grade, conn[i], nv, vid, &size[i]);
            mean+=size[i];
        }

        mean /= ((double) nconn);

        if (fabs(mean)<MORPHO_EPS) { *out = 0; return true; }

        /* Now evaluate the functional at this vertex */
        if (!ref->weight || fabs(ref->mean)<MORPHO_EPS) {
            for (unsigned int i=0; i<nconn; i++) total+=(1.0-size[i]/mean)*(1.0-size[i]/mean);
        } else {
            double weight[nconn], wmean=0.0;

            for (int i=0; i<nconn; i++) {
                weight[i]=1.0;
                matrix_getelement(ref->weight, 0, conn[i], &weight[i]);
                wmean+=weight[i];
            }

            wmean /= ((double) nconn);
            if (fabs(wmean)<MORPHO_EPS) wmean = 1.0;

            for (unsigned int i=0; i<nconn; i++) {
                double term = (1.0-weight[i]*size[i]/mean/wmean);
                total+=term*term;
            }
        }

        *out = total;
    }

    return true;
}

value EquiElement_init(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    int nfixed;
    value grade=MORPHO_INTEGER(-1);
    value weight=MORPHO_NIL;

    if (builtin_options(v, nargs, args, &nfixed, 2, equielement_weightproperty, &weight, functional_gradeproperty, &grade)) {
        objectinstance_setproperty(self, equielement_weightproperty, weight);
        objectinstance_setproperty(self, functional_gradeproperty, grade);
    } else morpho_runtimeerror(v, FUNCTIONAL_ARGS);

    return MORPHO_NIL;
}

FUNCTIONAL_METHOD(EquiElement, integrand, MESH_GRADE_VERTEX, equielementref, equielement_prepareref, functional_mapintegrand, equielement_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)

FUNCTIONAL_METHOD(EquiElement, total, MESH_GRADE_VERTEX, equielementref, equielement_prepareref, functional_sumintegrand, equielement_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)

FUNCTIONAL_METHOD(EquiElement, gradient, MESH_GRADE_VERTEX, equielementref, equielement_prepareref, functional_mapnumericalgradient, equielement_integrand, equielement_dependencies, FUNCTIONAL_ARGS, SYMMETRY_ADD)

FUNCTIONAL_METHOD(EquiElement, hessian, MESH_GRADE_VERTEX, equielementref, equielement_prepareref, functional_mapnumericalhessian, equielement_integrand, equielement_dependencies, FUNCTIONAL_ARGS, SYMMETRY_ADD)

MORPHO_BEGINCLASS(EquiElement)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, EquiElement_init, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, EquiElement_integrand, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, EquiElement_total, MORPHO_FN_PUREFN|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, EquiElement_gradient, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_HESSIAN_METHOD, EquiElement_hessian, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED)
MORPHO_ENDCLASS

/* **********************************************************************
 * Curvatures
 * ********************************************************************** */

/* ----------------------------------------------
 * LineCurvatureSq
 * ---------------------------------------------- */

static value curvature_integrandonlyproperty;

typedef struct {
    objectsparse *lineel; // Lines
    objectselection *selection; // Selection
    bool integrandonly; // Output integrated curvature or 'bare' curvature.
} curvatureref;

bool curvature_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, curvatureref *ref) {
    bool success = true;

    ref->selection=sel;

    ref->lineel = mesh_getconnectivityelement(mesh, MESH_GRADE_VERTEX, MESH_GRADE_LINE);
    if (ref->lineel) success=sparse_checkformat(ref->lineel, SPARSE_CCS, true, false);

    if (success) {
        objectsparse *s = mesh_getconnectivityelement(mesh, MESH_GRADE_LINE, MESH_GRADE_VERTEX);
        if (!s) s=mesh_addconnectivityelement(mesh, MESH_GRADE_LINE, MESH_GRADE_VERTEX);
        success=s;
    }

    if (success) {
        value integrandonly=MORPHO_FALSE;
        objectinstance_getpropertyinterned(self, curvature_integrandonlyproperty, &integrandonly);
        ref->integrandonly=MORPHO_ISTRUE(integrandonly);
    }

    return success;
}

/** Finds the points that a point depends on  */
bool linecurvsq_dependencies(functional_mapinfo *info, elementid id, varray_elementid *out) {
    objectmesh *mesh = info->mesh;
    curvatureref *cref = info->ref;
    bool success=false;
    varray_elementid nbrs;
    varray_elementidinit(&nbrs);

    varray_elementidwrite(out, id); // LinecurvSq is a vertex element, and hence depends on itself
    
    if (mesh_findneighbors(mesh, MESH_GRADE_VERTEX, id, MESH_GRADE_LINE, &nbrs)>0) {
        for (unsigned int i=0; i<nbrs.count; i++) {
            int nentries, *entries; // Get the vertices for this edge
            if (!sparseccs_getrowindices(&cref->lineel->ccs, nbrs.data[i], &nentries, &entries)) goto linecurvsq_dependencies_cleanup;
            for (unsigned int j=0; j<nentries; j++) {
                if (entries[j]==id) continue;
                varray_elementidwrite(out, entries[j]);
            }
        }
    }
    success=true;

linecurvsq_dependencies_cleanup:
    varray_elementidclear(&nbrs);

    return success;
}

/** Calculate the integral of the curvature squared  */
bool linecurvsq_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    curvatureref *cref = (curvatureref *) ref;
    double result = 0.0;
    varray_elementid nbrs;
    varray_elementid synid;
    varray_elementidinit(&nbrs);
    varray_elementidinit(&synid);

    double s0[mesh->dim], s1[mesh->dim], *s[2] = { s0, s1}, sgn=-1.0;

    if (mesh_findneighbors(mesh, MESH_GRADE_VERTEX, id, MESH_GRADE_LINE, &nbrs)>0 &&
        mesh_getsynonyms(mesh, MESH_GRADE_VERTEX, id, &synid)) {
        if (nbrs.count!=2) goto linecurvsq_integrand_cleanup;

        for (unsigned int i=0; i<2; i++) {
            int nentries, *entries; // Get the vertices for this edge
            if (!sparseccs_getrowindices(&cref->lineel->ccs, nbrs.data[i], &nentries, &entries)) break;

            double *x0, *x1;
            if (mesh_getvertexcoordinatesaslist(mesh, entries[0], &x0) &&
                mesh_getvertexcoordinatesaslist(mesh, entries[1], &x1)) {
                functional_vecsub(mesh->dim, x0, x1, s[i]);
            }
            if (!(entries[0]==id || functional_inlist(&synid, entries[0]))) sgn*=-1;
        }

        double s0s0=functional_vecdot(mesh->dim, s0, s0),
               s0s1=functional_vecdot(mesh->dim, s0, s1),
               s1s1=functional_vecdot(mesh->dim, s1, s1);

        s0s0=sqrt(s0s0); s1s1=sqrt(s1s1);

        if (s0s0<MORPHO_EPS || s1s1<MORPHO_EPS) goto linecurvsq_integrand_cleanup;

        double u=sgn*s0s1/s0s0/s1s1,
               len=0.5*(s0s0+s1s1);

        if (u<1) u=acos(u); else u=0;

        result = u*u/len;
        if (cref->integrandonly) result /= len; // Get the bare curvature.
    }

linecurvsq_integrand_cleanup:

    *out = result;
    varray_elementidclear(&nbrs);
    varray_elementidclear(&synid);

    return true;
}

FUNCTIONAL_INIT(LineCurvatureSq, MESH_GRADE_VERTEX)
FUNCTIONAL_METHOD(LineCurvatureSq, integrand, MESH_GRADE_VERTEX, curvatureref, curvature_prepareref, functional_mapintegrand, linecurvsq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(LineCurvatureSq, integrandForElement, MESH_GRADE_VERTEX, curvatureref, curvature_prepareref, functional_mapintegrandforelement, linecurvsq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(LineCurvatureSq, total, MESH_GRADE_VERTEX, curvatureref, curvature_prepareref, functional_sumintegrand, linecurvsq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(LineCurvatureSq, gradient, MESH_GRADE_VERTEX, curvatureref, curvature_prepareref, functional_mapnumericalgradient, linecurvsq_integrand, linecurvsq_dependencies, FUNCTIONAL_ARGS, SYMMETRY_ADD)
FUNCTIONAL_METHOD(LineCurvatureSq, hessian, MESH_GRADE_VERTEX, curvatureref, curvature_prepareref, functional_mapnumericalhessian, linecurvsq_integrand, linecurvsq_dependencies, FUNCTIONAL_ARGS, SYMMETRY_ADD)

MORPHO_BEGINCLASS(LineCurvatureSq)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, LineCurvatureSq_init, MORPHO_FN_MUTATES),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, LineCurvatureSq_integrand, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_INTEGRANDFORELEMENT_METHOD, LineCurvatureSq_integrandForElement, MORPHO_FN_PUREFN|MORPHO_FN_THROWS),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, LineCurvatureSq_gradient, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, LineCurvatureSq_total, MORPHO_FN_PUREFN|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_HESSIAN_METHOD, LineCurvatureSq_hessian, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * LineTorsionSq
 * ---------------------------------------------- */

/** Return a list of vertices that an element depends on  */
bool linetorsionsq_dependencies(functional_mapinfo *info, elementid id, varray_elementid *out) {
    objectmesh *mesh = info->mesh;
    curvatureref *cref = info->ref;
    bool success=false;
    varray_elementid nbrs;
    varray_elementid synid;

    varray_elementidinit(&nbrs);
    varray_elementidinit(&synid);

    if (mesh_findneighbors(mesh, MESH_GRADE_LINE, id, MESH_GRADE_LINE, &nbrs)>0) {
        for (unsigned int i=0; i<nbrs.count; i++) {
            int nentries, *entries; // Get the vertices for this edge
            if (!sparseccs_getrowindices(&cref->lineel->ccs, nbrs.data[i], &nentries, &entries)) goto linetorsionsq_dependencies_cleanup;
            for (unsigned int j=0; j<nentries; j++) {
                varray_elementidwriteunique(out, entries[j]);
            }
        }
    }
    success=true;

linetorsionsq_dependencies_cleanup:
    varray_elementidclear(&nbrs);
    varray_elementidclear(&synid);

    return success;
}


/** Calculate the integral of the torsion squared  */
bool linetorsionsq_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    curvatureref *cref = (curvatureref *) ref;
    int tmpi; elementid tmpid;
    bool success=false;

    //double result = 0.0;
    varray_elementid nbrs;
    varray_elementid synid;
    varray_elementidinit(&nbrs);
    varray_elementidinit(&synid);
    elementid vlist[6]; // List of vertices in order  n
    int type[6];
    for (unsigned int i=0; i<6; i++) type[i]=-1;

    /* We want an ordered list of vertex indices:
     *               v the element
     *    0 --- 1/2 --- 3/4 --- 5
     * Where 1/2 and 3/4 are the same vertex, but could have different indices due to symmetries */
     vlist[2] = vid[0]; vlist[3] = vid[1]; // Copy the current element into place

    /* First identify neighbors and get the vertex ids for each element */
    if (mesh_findneighbors(mesh, MESH_GRADE_LINE, id, MESH_GRADE_LINE, &nbrs)>0) {
        if (nbrs.count<2) {
            *out = 0; success=true;
            goto linecurvsq_torsion_cleanup;
        }

        for (unsigned int i=0; i<nbrs.count; i++) {
            int nentries, *entries; // Get the vertices for this edge
            if (!sparseccs_getrowindices(&cref->lineel->ccs, nbrs.data[i], &nentries, &entries)) goto linecurvsq_torsion_cleanup;
            for (unsigned int j=0; j<nentries; j++) { // Copy the vertexids
                vlist[4*i+j] = entries[j];
            }
        }
    }

    /* The vertex ids are not yet in the right order. Let's identify which vertex is which */
    for (int i=0; i<2; i++) {
        if (mesh_getsynonyms(mesh, 0, vid[i], &synid)) {
            for (int j=0; j<6; j++) if (vlist[j]==vid[i] || functional_inlist(&synid, vlist[j])) type[j]=i;
        }
    }
    /* The type array now contains either 0,1 depending on which vertex we have, or -1 if the vertex is not a synonym for the element's vertices */
#define SWAP(var, i, j, tmp) { tmp=var[i]; var[i]=var[j]; var[j]=tmp; }
    if (type[0]==1 || type[1]==1) { // Make sure the first segment corresponds to the first vertex
        SWAP(vlist, 0, 4, tmpid); SWAP(vlist, 1, 5, tmpid);
        SWAP(type, 0, 4, tmpi); SWAP(type, 1, 5, tmpi);
    }

    if (type[1]==-1) { // Check order of first segment
        SWAP(vlist, 0, 1, tmpid);
        SWAP(type, 0, 1, tmpi);
    }

    if (type[4]==-1) { // Check order of first segment
        SWAP(vlist, 4, 5, tmpid);
        SWAP(type, 4, 5, tmpi);
    }
#undef SWAP

    /* We now have an ordered list of vertices.
       Get the vertex positions */
    double *x[6];
    for (int i=0; i<6; i++) matrix_getcolumnptr(mesh->vert, vlist[i], &x[i]);

    double A[3], B[3], C[3], crossAB[3], crossBC[3];
    functional_vecsub(3, x[1], x[0], A);
    functional_vecsub(3, x[3], x[2], B);
    functional_vecsub(3, x[5], x[4], C);

    functional_veccross(A, B, crossAB);
    functional_veccross(B, C, crossBC);

    double normB=functional_vecnorm(3, B),
           normAB=functional_vecnorm(3, crossAB),
           normBC=functional_vecnorm(3, crossBC);

    double S = functional_vecdot(3, A, crossBC)*normB;
    if (normAB>MORPHO_EPS) S/=normAB;
    if (normBC>MORPHO_EPS) S/=normBC;

    S=asin(S);
    *out=S*S/normB;
    success=true;

linecurvsq_torsion_cleanup:
    varray_elementidclear(&nbrs);
    varray_elementidclear(&synid);

    return success;
}

FUNCTIONAL_INIT(LineTorsionSq, MESH_GRADE_LINE)
FUNCTIONAL_METHOD(LineTorsionSq, integrand, MESH_GRADE_LINE, curvatureref, curvature_prepareref, functional_mapintegrand, linetorsionsq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(LineTorsionSq, total, MESH_GRADE_LINE, curvatureref, curvature_prepareref, functional_sumintegrand, linetorsionsq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(LineTorsionSq, gradient, MESH_GRADE_LINE, curvatureref, curvature_prepareref, functional_mapnumericalgradient, linetorsionsq_integrand, linetorsionsq_dependencies, FUNCTIONAL_ARGS, SYMMETRY_ADD)
FUNCTIONAL_METHOD(LineTorsionSq, hessian, MESH_GRADE_LINE, curvatureref, curvature_prepareref, functional_mapnumericalhessian, linetorsionsq_integrand, linetorsionsq_dependencies, FUNCTIONAL_ARGS, SYMMETRY_ADD)

MORPHO_BEGINCLASS(LineTorsionSq)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, LineTorsionSq_init, MORPHO_FN_MUTATES),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, LineTorsionSq_integrand, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, LineTorsionSq_gradient, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, LineTorsionSq_total, MORPHO_FN_PUREFN|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_HESSIAN_METHOD, LineTorsionSq_hessian, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * MeanCurvatureSq
 * ---------------------------------------------- */

static value curvature_geodesicproperty;

typedef struct {
    objectsparse *areael; // Areas
    objectselection *selection; // Selection
    bool integrandonly; // Output integrated curvature or 'bare' curvature.
    bool geodesic; // Compute the geodesic curvature instead of the Gauss curvature (see https://cuhkmath.wordpress.com/2016/06/21/the-discrete-gauss-bonnet-theorem/)
} areacurvatureref;

bool areacurvature_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, areacurvatureref *ref) {
    bool success = true;

    ref->selection=sel;

    ref->areael = mesh_getconnectivityelement(mesh, MESH_GRADE_VERTEX, MESH_GRADE_AREA);
    if (ref->areael) success=sparse_checkformat(ref->areael, SPARSE_CCS, true, false);

    if (success) {
        objectsparse *s = mesh_getconnectivityelement(mesh, MESH_GRADE_AREA, MESH_GRADE_VERTEX);
        if (!s) s=mesh_addconnectivityelement(mesh, MESH_GRADE_AREA, MESH_GRADE_VERTEX);
        success=s;
    }

    if (success) {
        value integrandonly=MORPHO_FALSE;
        objectinstance_getpropertyinterned(self, curvature_integrandonlyproperty, &integrandonly);
        ref->integrandonly=MORPHO_ISTRUE(integrandonly);

        value geodesic=MORPHO_FALSE;
        objectinstance_getpropertyinterned(self, curvature_geodesicproperty, &geodesic);
        ref->geodesic=MORPHO_ISTRUE(geodesic);
    }

    return success;
}

/** Return a list of vertices that an element depends on  */
bool meancurvaturesq_dependencies(functional_mapinfo *info, elementid id, varray_elementid *out) {
    objectmesh *mesh = info->mesh;
    areacurvatureref *cref = info->ref;
    bool success=false;
    varray_elementid nbrs;
    varray_elementid synid;

    varray_elementidinit(&nbrs);
    varray_elementidinit(&synid);

    mesh_getsynonyms(mesh, MESH_GRADE_VERTEX, id, &synid);
    varray_elementidwriteunique(&synid, id);

    /* Loop over synonyms of the element id */
    mesh_findneighbors(mesh, MESH_GRADE_VERTEX, id, MESH_GRADE_AREA, &nbrs);

    for (unsigned int i=0; i<nbrs.count; i++) { /* Loop over adjacent triangles */
        int nvert, *vids;
        if (!sparseccs_getrowindices(&cref->areael->ccs, nbrs.data[i], &nvert, &vids)) goto meancurvsq_dependencies_cleanup;

        for (unsigned int j=0; j<nvert; j++) {
            if (vids[j]==id) continue;
            varray_elementidwriteunique(out, vids[j]);
        }
    }
    success=true;

meancurvsq_dependencies_cleanup:
    varray_elementidclear(&nbrs);
    varray_elementidclear(&synid);

    return success;
}

/** Orders the vertices in the list vids so that the vertex in synid is first */
bool curvature_ordervertices(varray_elementid *synid, int nv, int *vids) {
    int posn=-1;
    for (unsigned int i=0; i<nv && posn<0; i++) {
        for (unsigned int k=0; k<synid->count; k++) if (synid->data[k]==vids[i]) { posn = i; break; }
    }

    if (posn>0) { // If the desired vertex isn't in first position, move it there.
        int tmp=vids[posn];
        vids[posn]=vids[0]; vids[0]=tmp;
    }

    return (posn>=0);
}

/** Calculate the integral of the mean curvature squared  */
bool meancurvaturesq_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    areacurvatureref *cref = (areacurvatureref *) ref;
    double areasum = 0;
    bool success=false;

    varray_elementid nbrs;
    varray_elementid synid;
    varray_elementidinit(&nbrs);
    varray_elementidinit(&synid);

    mesh_getsynonyms(mesh, MESH_GRADE_VERTEX, id, &synid);
    varray_elementidwriteunique(&synid, id);

    double frc[mesh->dim]; // This will hold the total force due to the triangles present
    for (unsigned int i=0; i<mesh->dim; i++) frc[i]=0.0;

    mesh_findneighbors(mesh, MESH_GRADE_VERTEX, id, MESH_GRADE_AREA, &nbrs);

    for (unsigned int i=0; i<nbrs.count; i++) { /* Loop over adjacent triangles */
        int nvert, *ovids;
        if (!sparseccs_getrowindices(&cref->areael->ccs, nbrs.data[i], &nvert, &ovids)) goto meancurvsq_cleanup;

        int vids[nvert]; // Copy so we can reorder
        for (int j=0; j<nvert; j++) vids[j]=ovids[j];
        
        /* Order the vertices */
        if (!curvature_ordervertices(&synid, nvert, vids)) goto meancurvsq_cleanup;

        double *x[3], s0[3], s1[3], s01[3], s101[3];
        double norm;
        for (int j=0; j<3; j++) matrix_getcolumnptr(mesh->vert, vids[j], &x[j]);

        /* s0 = x1-x0; s1 = x2-x1 */
        functional_vecsub(mesh->dim, x[1], x[0], s0);
        functional_vecsub(mesh->dim, x[2], x[1], s1);

        /* F(v0) = (s1 x s0 x s1)/|s0 x x1|/2 */
        functional_veccross(s0, s1, s01);
        norm=functional_vecnorm(mesh->dim, s01);
        if (norm<MORPHO_EPS) goto meancurvsq_cleanup;

        areasum+=norm/2;
        functional_veccross(s1, s01, s101);

        functional_vecaddscale(mesh->dim, frc, 0.5/norm, s101, frc);
    }

    *out = functional_vecdot(mesh->dim, frc, frc)/(areasum/3.0)/4.0;
    if (cref->integrandonly) *out /= (areasum/3.0);
    success=true;

meancurvsq_cleanup:
    varray_elementidclear(&nbrs);
    varray_elementidclear(&synid);

    return success;
}

FUNCTIONAL_INIT(MeanCurvatureSq, MESH_GRADE_VERTEX)
FUNCTIONAL_METHOD(MeanCurvatureSq, integrand, MESH_GRADE_VERTEX, areacurvatureref, areacurvature_prepareref, functional_mapintegrand, meancurvaturesq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(MeanCurvatureSq, total, MESH_GRADE_VERTEX, areacurvatureref, areacurvature_prepareref, functional_sumintegrand, meancurvaturesq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(MeanCurvatureSq, gradient, MESH_GRADE_VERTEX, areacurvatureref, areacurvature_prepareref, functional_mapnumericalgradient, meancurvaturesq_integrand, meancurvaturesq_dependencies, FUNCTIONAL_ARGS, SYMMETRY_ADD)

MORPHO_BEGINCLASS(MeanCurvatureSq)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, MeanCurvatureSq_init, MORPHO_FN_MUTATES),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, MeanCurvatureSq_integrand, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, MeanCurvatureSq_gradient, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, MeanCurvatureSq_total, MORPHO_FN_PUREFN|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * GaussCurvature
 * ---------------------------------------------- */

/** Calculate the integral of the gaussian curvature  */
bool gausscurvature_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    areacurvatureref *cref = (areacurvatureref *) ref;
    double anglesum = 0, areasum = 0;
    bool success=false;

    varray_elementid nbrs;
    varray_elementid synid;
    varray_elementidinit(&nbrs);
    varray_elementidinit(&synid);

    mesh_getsynonyms(mesh, MESH_GRADE_VERTEX, id, &synid);
    varray_elementidwriteunique(&synid, id);

    double frc[mesh->dim]; // This will hold the total force due to the triangles present
    for (unsigned int i=0; i<mesh->dim; i++) frc[i]=0.0;

    mesh_findneighbors(mesh, MESH_GRADE_VERTEX, id, MESH_GRADE_AREA, &nbrs);

    for (unsigned int i=0; i<nbrs.count; i++) { /* Loop over adjacent triangles */
        int nvert, *ovids;
        if (!sparseccs_getrowindices(&cref->areael->ccs, nbrs.data[i], &nvert, &ovids)) goto gausscurv_cleanup;
        
        int vids[nvert]; // Copy so we can reorder
        for (int j=0; j<nvert; j++) vids[j]=ovids[j];

        /* Order the vertices */
        if (!curvature_ordervertices(&synid, nvert, vids)) goto gausscurv_cleanup;

        double *x[3], s0[3], s1[3], s01[3];
        for (int j=0; j<3; j++) matrix_getcolumnptr(mesh->vert, vids[j], &x[j]);

        /* s0 = x1-x0; s1 = x2-x0 */
        functional_vecsub(mesh->dim, x[1], x[0], s0);
        functional_vecsub(mesh->dim, x[2], x[0], s1);

        functional_veccross(s0, s1, s01);
        double area = functional_vecnorm(mesh->dim, s01);
        anglesum+=atan2(area, functional_vecdot(mesh->dim, s0, s1));

        areasum+=area/2;
    }

    *out = 2*M_PI-anglesum;
    if (cref->geodesic) *out = M_PI-anglesum;
    if (cref->integrandonly) *out /= (areasum/3.0);
    success=true;

gausscurv_cleanup:
    varray_elementidclear(&nbrs);
    varray_elementidclear(&synid);

    return success;
}

FUNCTIONAL_INIT(GaussCurvature, MESH_GRADE_VERTEX)
FUNCTIONAL_METHOD(GaussCurvature, integrand, MESH_GRADE_VERTEX, areacurvatureref, areacurvature_prepareref, functional_mapintegrand, gausscurvature_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(GaussCurvature, total, MESH_GRADE_VERTEX, areacurvatureref, areacurvature_prepareref, functional_sumintegrand, gausscurvature_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE)
FUNCTIONAL_METHOD(GaussCurvature, gradient, MESH_GRADE_VERTEX, areacurvatureref, areacurvature_prepareref, functional_mapnumericalgradient, gausscurvature_integrand, meancurvaturesq_dependencies, FUNCTIONAL_ARGS, SYMMETRY_ADD)

MORPHO_BEGINCLASS(GaussCurvature)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, GaussCurvature_init, MORPHO_FN_MUTATES),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, GaussCurvature_integrand, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, GaussCurvature_gradient, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, GaussCurvature_total, MORPHO_FN_PUREFN|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED)
MORPHO_ENDCLASS

/* **********************************************************************
 * Fields
 * ********************************************************************** */

typedef struct {
    objectfield *field;
    grade grade;
} fieldref;

static void functional_fespaceerror(vm *v, objectfield *field, grade g) {
    char *name = (field ? MORPHO_GETFESPACENAME(field->fnspc) : NULL);

    morpho_runtimeerror(v, FUNC_FESPACE,
                        name ? " with finite element space " : "",
                        name ? name : "",
                        (unsigned int) g);
}

static bool functional_preparefespacefield(vm *v, objectfield *field, grade g) {
    if (!field || !MORPHO_ISFESPACE(field->fnspc)) return true;

    fespace *disc = MORPHO_GETFESPACE(field->fnspc)->fespace;
    /* Revisit: CG1→vertices is not a missing trace. Honest FnctlFESpc tests
       wait on CG0 (no boundary restriction) and refusing g > disc->grade
       (no implicit raise, e.g. AreaIntegral of a line-grade space). */
    if (g<disc->grade) {
        if (!fespace_lower(disc, g, &disc)) {
            functional_fespaceerror(v, field, g);
            return false;
        }
    }

    objectsparse *conn = mesh_getconnectivityelement(field->mesh, 0, disc->grade);
    if (!conn) conn = mesh_addconnectivityelement(field->mesh, 0, disc->grade);
    if (!conn) MORPHO_FAILVARGS(v, FUNC_ELNTFND, (unsigned int) disc->grade);

    for (grade i=0; i<=disc->grade; i++) {
        objectsparse *vmatrix = mesh_addconnectivityelement(field->mesh, i, 0);
        if (!vmatrix && i>0 && disc->shape[i]>0) {
            if (!mesh_addgrade(field->mesh, i)) MORPHO_FAILVARGS(v, FUNC_ELNTFND, (unsigned int) i);
            vmatrix = mesh_addconnectivityelement(field->mesh, i, 0);
        }
        if (!vmatrix && i>0 && disc->shape[i]>0) MORPHO_FAILVARGS(v, FUNC_ELNTFND, (unsigned int) i);
    }

    return true;
}

static bool functional_preparefieldlist(vm *v, value *fields, int nfields, grade g) {
    for (int i=0; i<nfields; i++) {
        if (MORPHO_ISFIELD(fields[i]) &&
            !functional_preparefespacefield(v, MORPHO_GETFIELD(fields[i]), g)) return false;
    }

    return true;
}

static bool fieldref_startfn(vm *v, functional_mapinfo *info) {
    fieldref *ref = (fieldref *) info->ref;
    return functional_preparefespacefield(v, ref->field, info->g);
}

/* ----------------------------------------------
 * GradSq
 * ---------------------------------------------- */

bool gradsq_computeperpendicular(unsigned int n, double *s1, double *s2, double *out) {
    double s1s2, s2s2, sout;

    /* Compute s1 - (s1.s2) s2 / (s2.2) */
    s1s2 = functional_vecdot(n, s1, s2);
    s2s2 = functional_vecdot(n, s2, s2);
    if (fabs(s2s2)<MORPHO_EPS) return false; // Check for side of zero weight

    double temp[n];
    functional_vecscale(n, s1s2/s2s2, s2, temp);
    functional_vecsub(n, s1, temp, out);

    /* Scale by 1/|t|^2 */
    sout = functional_vecnorm(n, out);
    if (fabs(sout)<MORPHO_EPS) return false; // Check for side of zero weight

    functional_vecscale(n, 1/(sout*sout), out, out);
    return true;
}

/** Evaluates the gradient of a field quantity
 @param[in] mesh - object to use
 @param[in] field - field to compute gradient of
 @param[in] nv - number of vertices
 @param[in] vid - vertex ids
 @param[out] out - should be field->psize * mesh->dim units of storage */
bool gradsq_evaluategradient(objectmesh *mesh, objectfield *field, int nv, int *vid, double *out) {    double *f[nv]; // Field value lists
    double *x[nv]; // Vertex coordinates
    unsigned int nentries=0;

    // Get field values and vertex coordinates
    for (unsigned int i=0; i<nv; i++) {
        if (!mesh_getvertexcoordinatesaslist(mesh, vid[i], &x[i])) return false;
        if (!field_getelementaslist(field, MESH_GRADE_VERTEX, vid[i], 0, &nentries, &f[i])) return false;
    }

    double s[3][mesh->dim], t[3][mesh->dim];

    /* Vector sides */
    functional_vecsub(mesh->dim, x[1], x[0], s[0]);
    functional_vecsub(mesh->dim, x[2], x[1], s[1]);
    functional_vecsub(mesh->dim, x[0], x[2], s[2]);

    /* Perpendicular vectors. Collapsed sides contribute zero gradient. */
    if (!gradsq_computeperpendicular(mesh->dim, s[2], s[1], t[0]) ||
        !gradsq_computeperpendicular(mesh->dim, s[0], s[2], t[1]) ||
        !gradsq_computeperpendicular(mesh->dim, s[1], s[0], t[2])) {
        for (unsigned int i=0; i<mesh->dim*nentries; i++) out[i]=0;
        return true;
    }

    /* Compute the gradient */
    for (unsigned int i=0; i<mesh->dim*nentries; i++) out[i]=0;
    for (unsigned int j=0; j<nv; j++) {
        for (unsigned int i=0; i<nentries; i++) {
            functional_vecaddscale(mesh->dim, &out[i*mesh->dim], f[j][i], t[j], &out[i*mesh->dim]);
        }
    }

    return true;
}

/** Evaluates the gradient of a field quantity in 3D
 @param[in] mesh - object to use
 @param[in] field - field to compute gradient of
 @param[in] nv - number of vertices
 @param[in] vid - vertex ids
 @param[out] out - should be field->psize * mesh->dim units of storage */
bool gradsq_evaluategradient3d(objectmesh *mesh, objectfield *field, int nv, int *vid, double *out) {
    double *f[nv]; // Field value lists
    double *x[nv]; // Vertex coordinates
    double xarray[nv*mesh->dim]; // Vertex coordinates
    double xtarray[nv*mesh->dim]; // Vertex coordinates
    unsigned int nentries=0;

    // Get field values and vertex coordinates
    for (unsigned int i=0; i<nv; i++) {
        if (!mesh_getvertexcoordinatesaslist(mesh, vid[i], &x[i])) return false;
        if (!field_getelementaslist(field, MESH_GRADE_VERTEX, vid[i], 0, &nentries, &f[i])) return false;
    }

    // Build a matrix such that the columns are x_i - x_0
    for (unsigned int i=1; i<nv; i++) {
        functional_vecsub(mesh->dim, x[i], x[0], &xarray[(i-1)*mesh->dim]);
    }

    for (unsigned int i=0; i<mesh->dim*nentries; i++) out[i]=0;

    objectmatrix M = MORPHO_STATICMATRIX(xarray, mesh->dim, mesh->dim);
    objectmatrix Mt = MORPHO_STATICMATRIX(xtarray, mesh->dim, mesh->dim);
    matrix_transpose(&M, &Mt);

    objectmatrix grad = MORPHO_STATICMATRIX(out, mesh->dim, nentries);

    // Loop over elements of the field
    for (unsigned int i=0; i<nentries; i++) {
        // Copy across the field values to form the rhs
        for (unsigned int j=0; j<mesh->dim; j++) out[i*mesh->dim+j] = f[j+1][i]-f[0][i];
    }

    // Solve to obtain the gradient of each element
    matrix_solvesmall(&Mt, &grad);

    return true;
}

/** Prepares the gradsq reference */
bool gradsq_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, fieldref *ref) {
    bool success=false, grdset=false;
    value field=MORPHO_NIL, grd=MORPHO_NIL;

    if (objectinstance_getpropertyinterned(self, functional_fieldproperty, &field) &&
        MORPHO_ISFIELD(field)) {
        ref->field=MORPHO_GETFIELD(field);
        success=true;
    }

    if (objectinstance_getpropertyinterned(self, functional_gradeproperty, &grd) &&
        MORPHO_ISINTEGER(grd)) {
        ref->grade=MORPHO_GETINTEGERVALUE(grd);
        if (ref->grade>0) grdset=true;
    }
    if (!grdset) ref->grade=mesh_maxgrade(mesh);

    return success;
}

/** Clones the nematic reference with a given substitute field */
void *gradsq_cloneref(void *ref, objectfield *field, objectfield *sub) {
    fieldref *nref = (fieldref *) ref;
    fieldref *clone = MORPHO_MALLOC(sizeof(fieldref));
    
    if (clone) {
        *clone = *nref;
        if (clone->field==field) clone->field=sub;
    }
    
    return clone;
}

/** Calculate the |grad q|^2 energy */
bool gradsq_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    fieldref *eref = ref;
    double size=0; // Length area or volume of the element
    double grad[eref->field->psize*mesh->dim];

    if (!functional_elementsize(v, mesh, eref->grade, id, nv, vid, &size)) return false;

    if (eref->grade==2) {
        if (!gradsq_evaluategradient(mesh, eref->field, nv, vid, grad)) return false;
    } else if (eref->grade==3) {
        if (!gradsq_evaluategradient3d(mesh, eref->field, nv, vid, grad)) return false;
    } else {
        MORPHO_FAILVARGS(v, FUNC_ELNTFND, (unsigned int) eref->grade);
    }

    double gradnrm=functional_vecnorm(eref->field->psize*mesh->dim, grad);
    *out = gradnrm*gradnrm*size;

    return true;
}

/** Initialize a GradSq object */
value GradSq_init(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));

    if (nargs>0 && MORPHO_ISFIELD(MORPHO_GETARG(args, 0))) {
        objectinstance_setproperty(self, functional_fieldproperty, MORPHO_GETARG(args, 0));
        objectinstance_setproperty(self, functional_gradeproperty, MORPHO_INTEGER(mesh_maxgrade(MORPHO_GETFIELD(MORPHO_GETARG(args, 0))->mesh)));
    } else {
        morpho_runtimeerror(v, FUNCTIONAL_ARGS);
        return MORPHO_FALSE;
    }

    /* Second (optional) argument is the grade to act on */
    if (nargs>1) {
        if (MORPHO_ISINTEGER(MORPHO_GETARG(args, 1))) {
            objectinstance_setproperty(self, functional_gradeproperty, MORPHO_GETARG(args, 1));
        }
    }

    return MORPHO_NIL;
}

FUNCTIONAL_METHOD_START(GradSq, integrand, (ref.grade), fieldref, gradsq_prepareref, fieldref_startfn, NULL, functional_mapintegrand, gradsq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD_START(GradSq, total, (ref.grade), fieldref, gradsq_prepareref, fieldref_startfn, NULL, functional_sumintegrand, gradsq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD_START(GradSq, gradient, (ref.grade), fieldref, gradsq_prepareref, fieldref_startfn, NULL, functional_mapnumericalgradient, gradsq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_ADD);

value GradSq_fieldgradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    fieldref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (gradsq_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, MESH_GRADE_AREA, info.sel, &ref)) {
            info.g = ref.grade;
            info.field = ref.field;
            info.integrand = gradsq_integrand;
            info.start = fieldref_startfn;
            info.cloneref = gradsq_cloneref;
            info.ref = &ref;
            functional_runmap(v, &info, functional_mapnumericalfieldgradient, &out);
        } else morpho_runtimeerror(v, FUNCTIONAL_ARGS);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

MORPHO_BEGINCLASS(GradSq)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, GradSq_init, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, GradSq_integrand, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, GradSq_total, MORPHO_FN_PUREFN|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, GradSq_gradient, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_FIELDGRADIENT_METHOD, GradSq_fieldgradient, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Nematic
 * ---------------------------------------------- */

static value nematic_ksplayproperty;
static value nematic_ktwistproperty;
static value nematic_kbendproperty;
static value nematic_pitchproperty;

typedef struct {
    double ksplay,ktwist,kbend,pitch;
    bool haspitch;
    objectfield *field;
    grade grade;
} nematicref;

static bool nematic_startfn(vm *v, functional_mapinfo *info) {
    nematicref *ref = (nematicref *) info->ref;
    return functional_preparefespacefield(v, ref->field, info->g);
}

/** Prepares the nematic reference */
bool nematic_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, nematicref *ref) {
    bool success=false, grdset=false;
    value field=MORPHO_NIL, grd=MORPHO_NIL;
    value val=MORPHO_NIL;
    ref->ksplay=1.0; ref->ktwist=1.0; ref->kbend=1.0; ref->pitch=0.0;
    ref->haspitch=false;

    if (objectinstance_getpropertyinterned(self, functional_fieldproperty, &field) &&
        MORPHO_ISFIELD(field)) {
        ref->field=MORPHO_GETFIELD(field);
        success=true;
    }
    if (objectinstance_getpropertyinterned(self, nematic_ksplayproperty, &val) && MORPHO_ISNUMBER(val)) {
        morpho_valuetofloat(val, &ref->ksplay);
    }
    if (objectinstance_getpropertyinterned(self, nematic_ktwistproperty, &val) && MORPHO_ISNUMBER(val)) {
        morpho_valuetofloat(val, &ref->ktwist);
    }
    if (objectinstance_getpropertyinterned(self, nematic_kbendproperty, &val) && MORPHO_ISNUMBER(val)) {
        morpho_valuetofloat(val, &ref->kbend);
    }
    if (objectinstance_getpropertyinterned(self, nematic_pitchproperty, &val) && MORPHO_ISNUMBER(val)) {
        morpho_valuetofloat(val, &ref->pitch);
        ref->haspitch=true;
    }

    if (objectinstance_getpropertyinterned(self, functional_gradeproperty, &grd) &&
        MORPHO_ISINTEGER(grd)) {
        ref->grade=MORPHO_GETINTEGERVALUE(grd);
        if (ref->grade>0) grdset=true;
    }
    if (!grdset) ref->grade=mesh_maxgrade(mesh);

    return success;
}

/** Clones the nematic reference with a given substitute field */
void *nematic_cloneref(void *ref, objectfield *field, objectfield *sub) {
    nematicref *nref = (nematicref *) ref;
    nematicref *clone = MORPHO_MALLOC(sizeof(nematicref));
    
    if (clone) {
        *clone = *nref;
        if (clone->field==field) clone->field=sub;
    }
    
    return clone;
}

/* Integrates two linear functions with values at vertices f[0]...f[2] and g[0]...g[2] */
double nematic_bcint(double *f, double *g) {
    return (f[0]*(2*g[0]+g[1]+g[2]) + f[1]*(g[0]+2*g[1]+g[2]) + f[2]*(g[0]+g[1]+2*g[2]))/12;
}

/* Integrates a linear vector function with values at vertices f[0]...f[2] */
double nematic_bcint1(double *f) {
    return (f[0] + f[1] + f[2])/3;
}

/* Integrates a linear vector function with values at vertices f[0]...f[n]
   Works for dimensions 1-3 at least */
double nematic_bcintf(unsigned int n, double *f) {
    double sum = 0;
    for (unsigned int i=0; i<n; i++) sum+=f[i];
    return sum/n;
}

/* Integrates a product of two linear functions with values at vertices
   f[0]...f[n] and g[0]...g[n].
   Works for dimensions 1-3 at least */
double nematic_bcintfg(unsigned int n, double *f, double *g) {
    double sum = 0;
    for (unsigned int i=0; i<n; i++) {
        for (unsigned int j=0; j<n; j++) sum+=f[i]*g[j];
        sum+=f[i]*g[i];
    }
    return sum/(n*(n+1));
}

/** Calculate the nematic energy */
bool nematic_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    nematicref *eref = ref;
    double size=0; // Length area or volume of the element
    double gradnnraw[eref->field->psize*3];
    double gradnn[eref->field->psize*3];
    double divnn, curlnn[3] = { 0.0, 0.0, 0.0 };
    
    for (int i=0; i<eref->field->psize*3; i++) { gradnn[i]=0.0; gradnnraw[i]=0.0; }

    if (!functional_elementsize(v, mesh, eref->grade, id, nv, vid, &size)) return false;

    // Get nematic director components
    double *nn[nv]; // Field value lists
    unsigned int nentries=0;
    for (unsigned int i=0; i<nv; i++) {
        if (!field_getelementaslist(eref->field, MESH_GRADE_VERTEX, vid[i], 0, &nentries, &nn[i])) return false;
    }

    // Evaluate gradients of the director
    if (eref->grade==2) {
        if (!gradsq_evaluategradient(mesh, eref->field, nv, vid, gradnnraw)) return
            false;
    } else if (eref->grade==3) {
        if (!gradsq_evaluategradient3d(mesh, eref->field, nv, vid, gradnnraw)) return
            false;
    }
    
    // Copy into 3x3 matrix
    for (int j=0; j<3; j++) for (int i=0; i<mesh->dim; i++) gradnn[3*j+i] = gradnnraw[mesh->dim*j+i];
    
    // Output of this is the matrix:
    // [ nx,x ny,x nz,x ] [ 0 3 6 ] <- indices
    // [ nx,y ny,y nz,y ] [ 1 4 7 ]
    // [ nx,z ny,z nz,z ] [ 2 5 8 ]
    objectmatrix gradnnmat = MORPHO_STATICMATRIX(gradnn, 3, 3);

    matrix_trace(&gradnnmat, &divnn);
    curlnn[0]=gradnn[7]-gradnn[5]; // nz,y - ny,z
    curlnn[1]=gradnn[2]-gradnn[6]; // nx,z - nz,x
    curlnn[2]=gradnn[3]-gradnn[1]; // ny,x - nx,y

    /* From components of the curl, construct the coefficients that go in front of integrals of
           nx^2, ny^2, nz^2, nx*ny, ny*nz, and nz*nx over the element. */
    double ctwst[6] = { curlnn[0]*curlnn[0], curlnn[1]*curlnn[1], curlnn[2]*curlnn[2],
                        2*curlnn[0]*curlnn[1], 2*curlnn[1]*curlnn[2], 2*curlnn[2]*curlnn[0]};

    double cbnd[6] = { ctwst[1] + ctwst[2], ctwst[0] + ctwst[2], ctwst[0] + ctwst[1],
                       -ctwst[3], -ctwst[4], -ctwst[5] };

    /* Calculate integrals of nx^2, ny^2, nz^2, nx*ny, ny*nz, and nz*nx over the element */
    double nnt[3][nv]; // The transpose of nn
    for (unsigned int i=0; i<nv; i++)
        for (unsigned int j=0; j<3; j++) nnt[j][i]=nn[i][j];

    double integrals[] = {  nematic_bcintfg(nv, nnt[0], nnt[0]),
                            nematic_bcintfg(nv, nnt[1], nnt[1]),
                            nematic_bcintfg(nv, nnt[2], nnt[2]),
                            nematic_bcintfg(nv, nnt[0], nnt[1]),
                            nematic_bcintfg(nv, nnt[1], nnt[2]),
                            nematic_bcintfg(nv, nnt[2], nnt[0])
    };

    /* Now we can calculate the components of splay, twist and bend */
    double splay=0.0, twist=0.0, bend=0.0, chol=0.0;

    /* Evaluate the three contributions to the integral */
    splay = 0.5*eref->ksplay*size*divnn*divnn;
    for (unsigned int i=0; i<6; i++) {
        twist += ctwst[i]*integrals[i];
        bend += cbnd[i]*integrals[i];
    }
    twist *= 0.5*eref->ktwist*size;
    bend *= 0.5*eref->kbend*size;

    if (eref->haspitch) {
        /* Cholesteric terms: 0.5 * k22 * [- 2 q (cx <nx> + cy <ny> + cz <nz>) + q^2] */
        for (unsigned i=0; i<3; i++) {
            chol += -2*curlnn[i]*nematic_bcintf(nv, nnt[i])*eref->pitch;
        }
        chol += (eref->pitch*eref->pitch);
        chol *= 0.5*eref->ktwist*size;
    }

    *out = splay+twist+bend+chol;

    return true;
}

/** Initialize a Nematic object */
value Nematic_init(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));

    int nfixed=nargs;
    value ksplay=MORPHO_FLOAT(1.0),
          ktwist=MORPHO_FLOAT(1.0),
          kbend=MORPHO_FLOAT(1.0);
    value pitch=MORPHO_NIL;

    if (builtin_options(v, nargs, args, &nfixed, 4,
                        nematic_ksplayproperty, &ksplay,
                        nematic_ktwistproperty, &ktwist,
                        nematic_kbendproperty, &kbend,
                        nematic_pitchproperty, &pitch)) {
        objectinstance_setproperty(self, nematic_ksplayproperty, ksplay);
        objectinstance_setproperty(self, nematic_ktwistproperty, ktwist);
        objectinstance_setproperty(self, nematic_kbendproperty, kbend);
        objectinstance_setproperty(self, nematic_pitchproperty, pitch);
    } else morpho_runtimeerror(v, FUNCTIONAL_ARGS);

    if (nfixed==1 && MORPHO_ISFIELD(MORPHO_GETARG(args, 0))) {
        objectinstance_setproperty(self, functional_fieldproperty, MORPHO_GETARG(args, 0));
        objectinstance_setproperty(self, functional_gradeproperty, MORPHO_INTEGER(mesh_maxgrade(MORPHO_GETFIELD(MORPHO_GETARG(args, 0))->mesh)));
    } else morpho_runtimeerror(v, FUNCTIONAL_ARGS);

    return MORPHO_NIL;
}

FUNCTIONAL_METHOD_START(Nematic, integrand, (ref.grade), nematicref, nematic_prepareref, nematic_startfn, NULL, functional_mapintegrand, nematic_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD_START(Nematic, total, (ref.grade), nematicref, nematic_prepareref, nematic_startfn, NULL, functional_sumintegrand, nematic_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD_START(Nematic, gradient, (ref.grade), nematicref, nematic_prepareref, nematic_startfn, NULL, functional_mapnumericalgradient, nematic_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE);

value Nematic_fieldgradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    nematicref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (nematic_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, MESH_GRADE_AREA, info.sel, &ref)) {
            info.g=ref.grade;
            info.integrand=nematic_integrand;
            info.start=nematic_startfn;
            info.ref=&ref;
            info.cloneref=nematic_cloneref;
            functional_runmap(v, &info, functional_mapnumericalfieldgradient, &out);
        } else morpho_runtimeerror(v, FUNCTIONAL_ARGS);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

MORPHO_BEGINCLASS(Nematic)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, Nematic_init, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, Nematic_integrand, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, Nematic_total, MORPHO_FN_PUREFN|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, Nematic_gradient, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_FIELDGRADIENT_METHOD, Nematic_fieldgradient, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * NematicElectric
 * ---------------------------------------------- */

typedef struct {
    objectfield *director;
    value field;
    grade grade;
} nematicelectricref;

static bool nematicelectric_startfn(vm *v, functional_mapinfo *info) {
    nematicelectricref *ref = (nematicelectricref *) info->ref;

    if (!functional_preparefespacefield(v, ref->director, info->g)) return false;
    if (MORPHO_ISFIELD(ref->field) &&
        !functional_preparefespacefield(v, MORPHO_GETFIELD(ref->field), info->g)) return false;

    return true;
}

/** Prepares the nematicelectric reference */
bool nematicelectric_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, nematicelectricref *ref) {
    bool success=false, grdset=false;
    ref->field=MORPHO_NIL;
    value fieldlist=MORPHO_NIL, grd=MORPHO_NIL;

    if (objectinstance_getpropertyinterned(self, functional_fieldproperty, &fieldlist) &&
        MORPHO_ISLIST(fieldlist)) {
        objectlist *lst = MORPHO_GETLIST(fieldlist);
        value director = MORPHO_NIL;
        list_getelement(lst, 0, &director);
        list_getelement(lst, 1, &ref->field);

        if (MORPHO_ISFIELD(director)) ref->director=MORPHO_GETFIELD(director);

        if (MORPHO_ISFIELD(ref->field) || MORPHO_ISMATRIX(ref->field)) success=true;
    }

    if (objectinstance_getpropertyinterned(self, functional_gradeproperty, &grd) &&
        MORPHO_ISINTEGER(grd)) {
        ref->grade=MORPHO_GETINTEGERVALUE(grd);
        if (ref->grade>0) grdset=true;
    }
    if (!grdset) ref->grade=mesh_maxgrade(mesh);

    return success;
}

/** Clones the nematic reference with a given substitute field */
void *nematicelectric_cloneref(void *ref, objectfield *field, objectfield *sub) {
    nematicelectricref *nref = (nematicelectricref *) ref;
    nematicelectricref *clone = MORPHO_MALLOC(sizeof(nematicelectricref));
    
    if (clone) {
        *clone = *nref;
        if (clone->director==field) clone->director=sub;
        if (MORPHO_ISFIELD(clone->field) &&
            MORPHO_GETFIELD(clone->field)==field) {
            clone->field=MORPHO_OBJECT(sub);
        }
    }
    
    return clone;
}

/** Calculate the integral (n.E)^2 energy, where E is calculated from the electric potential */
bool nematicelectric_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    nematicelectricref *eref = ref;
    double size=0; // Length area or volume of the element

    if (!functional_elementsize(v, mesh, eref->grade, id, nv, vid, &size)) return false;

    // Get nematic director components
    double *nn[nv]; // Field value lists
    unsigned int nentries=0;
    for (unsigned int i=0; i<nv; i++) {
        if (!field_getelementaslist(eref->director, MESH_GRADE_VERTEX, vid[i], 0, &nentries, &nn[i])) return false;
    }

    // The electric field ends up being constant over the element
    double ee[mesh->dim];
    if (MORPHO_ISFIELD(eref->field)) {
        if (eref->grade==2) {
            if (!gradsq_evaluategradient(mesh, MORPHO_GETFIELD(eref->field), nv, vid, ee)) return false;
        } else if (eref->grade==3) {
            if (!gradsq_evaluategradient3d(mesh, MORPHO_GETFIELD(eref->field), nv, vid, ee)) return false;
        }
    }

    /* Calculate integrals of nx^2, ny^2, nz^2, nx*ny, ny*nz, and nz*nx over the element */
    double nnt[mesh->dim][nv]; // The transpose of nn
    for (unsigned int i=0; i<nv; i++)
        for (unsigned int j=0; j<mesh->dim; j++) nnt[j][i]=nn[i][j];

    /* Calculate integral (n.e)^2 using the above results */
    double total = ee[0]*ee[0]*nematic_bcintfg(nv, nnt[0], nnt[0])+
                   ee[1]*ee[1]*nematic_bcintfg(nv, nnt[1], nnt[1])+
                   ee[2]*ee[2]*nematic_bcintfg(nv, nnt[2], nnt[2])+
                   2*ee[0]*ee[1]*nematic_bcintfg(nv, nnt[0], nnt[1])+
                   2*ee[1]*ee[2]*nematic_bcintfg(nv, nnt[1], nnt[2])+
                   2*ee[2]*ee[0]*nematic_bcintfg(nv, nnt[2], nnt[0]);

    *out = size*total;

    return true;
}

/** Initialize a NematicElectric object */
value NematicElectric_init(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));

    if (nargs==2 && MORPHO_ISFIELD(MORPHO_GETARG(args, 0)) &&
        MORPHO_ISFIELD(MORPHO_GETARG(args, 1))) {
        objectlist *new = object_newlist(2, &MORPHO_GETARG(args, 0));
        if (new) {
            value lst = MORPHO_OBJECT(new);
            objectinstance_setproperty(self, functional_fieldproperty, lst);
            morpho_bindobjects(v, 1, &lst);
            objectinstance_setproperty(self, functional_gradeproperty, MORPHO_INTEGER(mesh_maxgrade(MORPHO_GETFIELD(MORPHO_GETARG(args, 0))->mesh)));
        }
    } else morpho_runtimeerror(v, FUNCTIONAL_ARGS);

    return MORPHO_NIL;
}

FUNCTIONAL_METHOD_START(NematicElectric, integrand, (ref.grade), nematicelectricref, nematicelectric_prepareref, nematicelectric_startfn, NULL, functional_mapintegrand, nematicelectric_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD_START(NematicElectric, total, (ref.grade), nematicelectricref, nematicelectric_prepareref, nematicelectric_startfn, NULL, functional_sumintegrand, nematicelectric_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD_START(NematicElectric, gradient, (ref.grade), nematicelectricref, nematicelectric_prepareref, nematicelectric_startfn, NULL, functional_mapnumericalgradient, nematicelectric_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE);

value NematicElectric_fieldgradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    nematicelectricref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (nematicelectric_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, MESH_GRADE_AREA, info.sel, &ref)) {
            info.g=ref.grade;
            info.integrand=nematicelectric_integrand;
            info.start=nematicelectric_startfn;
            info.cloneref=nematicelectric_cloneref;
            info.ref=&ref;
            functional_runmap(v, &info, functional_mapnumericalfieldgradient, &out);
        } else morpho_runtimeerror(v, FUNCTIONAL_ARGS);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

MORPHO_BEGINCLASS(NematicElectric)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, NematicElectric_init, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, NematicElectric_integrand, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, NematicElectric_total, MORPHO_FN_PUREFN|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, NematicElectric_gradient, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_FIELDGRADIENT_METHOD, NematicElectric_fieldgradient, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * NormSq
 * ---------------------------------------------- */

/** Calculate the norm squared of a field quantity */
bool normsq_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    fieldref *eref = ref;
    unsigned int nentries;
    double *entries;

    if (field_getelementaslist(eref->field, MESH_GRADE_VERTEX, id, 0, &nentries, &entries)) {
        *out = functional_vecdot(nentries, entries, entries);
        return true;
    }

    return false;
}

FUNCTIONAL_METHOD_START(NormSq, integrand, MESH_GRADE_VERTEX, fieldref, gradsq_prepareref, fieldref_startfn, NULL, functional_mapintegrand, normsq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD_START(NormSq, total, MESH_GRADE_VERTEX, fieldref, gradsq_prepareref, fieldref_startfn, NULL, functional_sumintegrand, normsq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE);

FUNCTIONAL_METHOD_START(NormSq, gradient, MESH_GRADE_VERTEX, fieldref, gradsq_prepareref, fieldref_startfn, NULL, functional_mapnumericalgradient, normsq_integrand, NULL, FUNCTIONAL_ARGS, SYMMETRY_NONE);

value NormSq_init(vm *v, int nargs, value *args) {
    GradSq_init(v, nargs, args);
    objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), functional_gradeproperty, MORPHO_INTEGER(MESH_GRADE_VERTEX));
    return MORPHO_NIL;
}

value NormSq_fieldgradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    fieldref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (gradsq_prepareref(MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, MESH_GRADE_VERTEX, info.sel, &ref)) {
            info.g=MESH_GRADE_VERTEX;
            info.ref=&ref;
            info.field=ref.field;
            info.integrand=normsq_integrand;
            info.start=fieldref_startfn;
            info.cloneref=gradsq_cloneref;
            functional_runmap(v, &info, functional_mapnumericalfieldgradient, &out);
        } else morpho_runtimeerror(v, FUNCTIONAL_ARGS);
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

MORPHO_BEGINCLASS(NormSq)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, NormSq_init, MORPHO_FN_MUTATES|MORPHO_FN_THROWS),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, NormSq_integrand, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, NormSq_total, MORPHO_FN_PUREFN|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, NormSq_gradient, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED),
MORPHO_METHOD(FUNCTIONAL_FIELDGRADIENT_METHOD, NormSq_fieldgradient, MORPHO_FN_PUREFN|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED)
MORPHO_ENDCLASS

/* **********************************************************************
 * Integrals
 * ********************************************************************** */

/** Integral references
 @brief Used to pass through the functional element mapping system.
 A thread local copy is made with cloned fields */

typedef struct {
    value integrand;
    
    int nfields;
    value *fields;
    value *originalfields; // Original fields
    value method; // Method dictionary
    objectmesh *mref; // Reference mesh
    vm *v;
    grade g; // Grade to integrate over
    bool weightbyref; // Use reference mesh for the element
} integralref;

static bool integral_startfn(vm *v, functional_mapinfo *info) {
    integralref *ref = (integralref *) info->ref;
    return functional_preparefieldlist(v, ref->fields, ref->nfields, info->g);
}

typedef struct jumpref_s jumpref;

typedef enum {
    JUMP_STRATEGY_CENTROID_MODE,
    JUMP_STRATEGY_QUADRATURE_MODE
} jumpstrategy;

typedef struct {
    int nv;
    int *vid;
    quantity *quantities;
} jumpside;

/* ----------------------------------------------
 * Integrand functions
 * ---------------------------------------------- */

/** Integral element references
 @brief used to store information about the current element in thread-local storage. We wrap them in an object so that they can be safely stored in a value.
 Guaranteed to be thread local */

typedef struct {
    object obj;
    objectmesh *mesh;    // The current mesh object
    
    integralref *iref;   // The current integral ref structure
    
// Information about the element
    grade g;             // Current grade
    elementid id;        // Current element
    int nv;              // Number of vertices
    int *vid;            // Vertex ids
    double **vertexposn; // List of vertex positions
    double elementsize;  // Size of the element
    
// Interpolated quantities:
    double *lambda;      // Barycentric coordinates
    double *posn;        // Position in physical space
    
    quantity *quantities; // Original quantities obtained for the element
    objectmatrix *invj;   // Inverse jacobian for the element
    
    value *qgrad;        // Gradients
    value *qhess;        // Hessians
    value *qinterpolated; // List of interpolated quantities (this allows us to identify operators on fields
} objectintegralelementref;

size_t objectintegralelementref_sizefn(object *obj) {
    return sizeof(objectintegralelementref);
}

void objectintegralelementref_printfn(object *obj, void *v) {
    morpho_printf(v, "<Elementref>");
}

objecttypedefn objectintegralelementrefdefn = {
    .printfn=objectintegralelementref_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectintegralelementref_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

objecttype objectintegralelementreftype;
#define OBJECT_INTEGRALELEMENTREF objectintegralelementreftype

/** Tests whether an object is an element ref */
#define MORPHO_ISINTEGRALELEMENTREF(val) object_istype(val, OBJECT_INTEGRALELEMENTREF)

/** Gets the object as an element ref */
#define MORPHO_GETINTEGRALELEMENTREF(val) ((objectintegralelementref *) MORPHO_GETOBJECT(val))

/** Static element ref */
#define MORPHO_STATICINTEGRALELEMENTREF(mesh, grade, id, nv, vid)      { .obj.type=OBJECT_INTEGRALELEMENTREF, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .g=grade, .mesh=mesh, .id=id, .nv=nv, .vid=vid, .qinterpolated=NULL }

int elementhandle;

/** Get the current element ref from thread-local storage in the VM */
objectintegralelementref *integral_getelementref(vm *v) {
    value elref=MORPHO_NIL;
    vm_gettlvar(v, elementhandle, &elref);
    if (MORPHO_ISINTEGRALELEMENTREF(elref)) return MORPHO_GETINTEGRALELEMENTREF(elref);
    
    return NULL;
}

/* ----------------------------------------------
 * Jump interface references
 * ---------------------------------------------- */

/** Thread-local interface context for Jump functionals. */
typedef struct {
    object obj;
    objectmesh *mesh;      // The current mesh object
    vm *v;                 // Worker VM for callbacks on this interface

    jumpref *jref;         // Shared Jump reference

    grade g;               // Interface grade
    elementid id;          // Interface id
    int nv;                // Number of interface vertices
    int *vid;              // Interface vertex ids
    double **vertexposn;   // Interface vertex positions
    double interfacesize;  // Size/measure of the interface

    elementid plusid;      // Canonical + parent element id
    elementid minusid;     // Canonical - parent element id

    int plusnv, minusnv;   // Number of vertices in parent elements
    int *plusvid, *minusvid; // Vertex ids in parent elements

    jumpside qplus;
    jumpside qminus;

    objectmatrix *normal;  // Canonical interface normal

    double *pluslambda;    // Parent-element barycentric coordinates on + side
    double *minuslambda;   // Parent-element barycentric coordinates on - side
    double *posn;          // Current physical position
    value *qinterpolated;  // Current interpolated quantities passed to the integrand

    value *qplusgrad;      // Per-side cached gradients, later
    value *qminusgrad;
    value *qplushess;      // Per-side cached Hessians, later
    value *qminushess;
} objectjumpinterfaceref;

size_t objectjumpinterfaceref_sizefn(object *obj) {
    return sizeof(objectjumpinterfaceref);
}

void objectjumpinterfaceref_printfn(object *obj, void *v) {
    morpho_printf(v, "<JumpInterfaceRef>");
}

objecttypedefn objectjumpinterfacerefdefn = {
    .printfn=objectjumpinterfaceref_printfn,
    .markfn=NULL,
    .freefn=NULL,
    .sizefn=objectjumpinterfaceref_sizefn,
    .hashfn=NULL,
    .cmpfn=NULL
};

objecttype objectjumpinterfacereftype;
#define OBJECT_JUMPINTERFACEREF objectjumpinterfacereftype

#define MORPHO_ISJUMPINTERFACEREF(val) object_istype(val, OBJECT_JUMPINTERFACEREF)
#define MORPHO_GETJUMPINTERFACEREF(val) ((objectjumpinterfaceref *) MORPHO_GETOBJECT(val))
#define MORPHO_STATICJUMPINTERFACEREF(mesh, grade, id, nv, vid) { .obj.type=OBJECT_JUMPINTERFACEREF, .obj.status=OBJECT_ISUNMANAGED, .obj.next=NULL, .mesh=mesh, .v=NULL, .g=grade, .id=id, .nv=nv, .vid=vid, .qplus={0}, .qminus={0}, .normal=NULL, .pluslambda=NULL, .minuslambda=NULL, .posn=NULL, .qinterpolated=NULL, .qplusgrad=NULL, .qminusgrad=NULL, .qplushess=NULL, .qminushess=NULL }

int jumpinterfacehandle;

static objectjumpinterfaceref *jump_getinterfaceref(vm *v) {
    value iref=MORPHO_NIL;
    vm_gettlvar(v, jumpinterfacehandle, &iref);
    if (MORPHO_ISJUMPINTERFACEREF(iref)) return MORPHO_GETJUMPINTERFACEREF(iref);
    
    return NULL;
}

/* ---------
 * Elementid
 * --------- */

static value integral_elementid(vm *v, int nargs, value *args) {
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref) MORPHO_RAISEVARGS(v, INTEGRAL_SPCLFN, ELEMENTID_FUNCTION);
    
    return MORPHO_INTEGER(elref->id);
}

/* --------
 * Tangent
 * -------- */

int tangenthandle; // TL storage handle for tangent vectors

/** Evaluate the tangent vector */
void integral_evaluatetangent(vm *v, value *out) {
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref || elref->g!=1) {
        morpho_runtimeerror(v, INTEGRAL_SPCLFN, TANGENT_FUNCTION);
        return;
    }
    
    int dim = elref->mesh->dim;
    
    objectmatrix *mtangent = matrix_new(dim, 1, false);
    if (!mtangent) {
        morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
        return;
    }
    
    functional_vecsub(dim, elref->vertexposn[1], elref->vertexposn[0], mtangent->elements);

    double tnorm=functional_vecnorm(dim, mtangent->elements);
    if (fabs(tnorm)>MORPHO_EPS) functional_vecscale(dim, 1.0/tnorm, mtangent->elements, mtangent->elements);
    
    vm_settlvar(v, tangenthandle, MORPHO_OBJECT(mtangent));
    *out = MORPHO_OBJECT(mtangent);
}

static value integral_tangent(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    
    vm_gettlvar(v, tangenthandle, &out);
    if (MORPHO_ISNIL(out)) integral_evaluatetangent(v, &out);
    
    return out;
}

/* --------
 * Normal
 * -------- */

int normlhandle; // TL storage handle for normal vectors

/** Evaluates the normal vector */
void integral_evaluatenormal(vm *v, value *out) {
    objectintegralelementref *elref = integral_getelementref(v);
    
    if (!elref || elref->g!=2) {
        morpho_runtimeerror(v, INTEGRAL_SPCLFN, NORMAL_FUNCTION);
        return;
    }
    
    int dim = elref->mesh->dim;
    double s0[dim], s1[dim];
    
    objectmatrix *mnormal = matrix_new(dim, 1, false);
    if (!mnormal) {
        morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
        return;
    }
    
    functional_vecsub(dim, elref->vertexposn[1], elref->vertexposn[0], s0);
    functional_vecsub(dim, elref->vertexposn[2], elref->vertexposn[1], s1);
    functional_veccross(s0, s1, mnormal->elements);
    
    double nnorm=functional_vecnorm(dim, mnormal->elements);
    if (fabs(nnorm)>MORPHO_EPS) functional_vecscale(dim, 1.0/nnorm, mnormal->elements, mnormal->elements);
    
    vm_settlvar(v, normlhandle, MORPHO_OBJECT(mnormal));
    *out = MORPHO_OBJECT(mnormal);
}

static value integral_normal(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;

    vm_gettlvar(v, normlhandle, &out);
    if (MORPHO_ISNIL(out)) integral_evaluatenormal(v, &out);
    
    return out;
}

/* --------
 * Gradient
 * -------- */

bool integrator_sumquantityweighted(int n, double *wts, value *q, value *out);

/** @brief Prepares an inverse jacobian matrix.
    @param[in] dim - dimension of physical space
    @param[in] g - grade of the object
    @param[in] x - list of vertex positions (grade+1 entries, each of length dim)
    @param[out] invj - inverse jacobian for the transformation (dim*g entries) */
bool integral_prepareinvjacobian(unsigned int dim, grade g, double **x, objectmatrix *invj) {
    bool success=false;
    
    // Construct the (dim x g) matrix of edge vectors
    double s[dim*g];
    for (int i=0; i<g; i++) functional_vecsub(dim, x[i+1], x[0], s + i*dim);
    
    if (g==dim) {
        objectmatrix smat = MORPHO_STATICMATRIX(s, dim, dim);
        success=(matrix_copy(&smat, invj)==LINALGERR_OK &&
                 matrix_inverse(invj)==LINALGERR_OK);
    } else if (g==1) {
        double s01norm = functional_vecdot(dim, s, s);
        if (s01norm>0) {
            functional_vecscale(dim, 1.0/s01norm, s, invj->elements);
            success=true;
        }
    } else if (g==2 && dim==3) {
        double *s0 = s, *s1 = s+dim, s0xs1[dim], u[dim], v[dim*g];
        functional_veccross(s0, s1, s0xs1);
        double s0xs1norm = functional_vecnorm(dim, s0xs1);
        if (s0xs1norm>0) {
            double invs0xs1norm = 1/(s0xs1norm*s0xs1norm);
            functional_veccross(s1, s0xs1, u);
            functional_vecscale(dim, invs0xs1norm, u, v);
            functional_veccross(s0xs1, s0, u);
            functional_vecscale(dim, invs0xs1norm, u, v+dim);
            
            objectmatrix invjt = MORPHO_STATICMATRIX(v, dim, g);
            
            matrix_transpose(&invjt, invj);
            
            success=true;
        }
    }
    return success;
}

/** Allocate suitable storage for the gradient */
bool integral_gradalloc(int dim, value prototype, value *out) {
    if (MORPHO_ISNIL(prototype)) { // Scalar
        objectmatrix *mgrad=matrix_new(dim, 1, false);
        if (mgrad) *out = MORPHO_OBJECT(mgrad);
        return mgrad;
    } else if (MORPHO_ISMATRIX(prototype)) {
        objectlist *mlst = object_newlist(0, NULL);
        if (mlst) *out = MORPHO_OBJECT(mlst);
        return mlst;
    } else UNREACHABLE("Field type not supported in grad");
    return false;
}

/** Allocate suitable storage for the hessian */
bool integral_hessalloc(int dim, value prototype, value *out) {
    if (MORPHO_ISNIL(prototype)) { // Scalar
        objectmatrix *mhess=matrix_new(dim, dim, false);
        if (mhess) *out = MORPHO_OBJECT(mhess);
        return mhess;
    } else if (MORPHO_ISMATRIX(prototype)) {
        objectlist *mlst = object_newlist(0, NULL);
        if (mlst) *out = MORPHO_OBJECT(mlst);
        return mlst;
    } else UNREACHABLE("Field type not supported in hess");
    return false;
}

/** Prepares the gradient sum to hold the component of the gradient */
bool integral_gradsuminit(int i, value prototype, value dest, value *sum) {
    if (MORPHO_ISLIST(dest)) {
        objectlist *lst = MORPHO_GETLIST(dest);
        
        if (i>=list_length(lst)) {
            objectmatrix *prmat = MORPHO_GETMATRIX(prototype);
            objectmatrix *new = matrix_new(prmat->nrows, prmat->ncols, true);
            if (!new) return false;
            *sum = MORPHO_OBJECT(new);
            list_append(lst, *sum);
        } else {
            matrix_zero(MORPHO_GETMATRIX(lst->val.data[i]));
            *sum = lst->val.data[i];
        }
    }
    return true;
}
 
/** Copies the component of the gradient into the relevant destination if needed */
bool integral_gradsumcopy(int i, value sum, value dest) {
    if (MORPHO_ISMATRIX(dest)) {
        return morpho_valuetofloat(sum, &MORPHO_GETMATRIX(dest)->elements[i]);
    } else return true;
}

/** Prepares the hessian sum to hold a component of the hessian */
bool integral_hesssuminit(int c, value prototype, value dest, value *sum) {
    if (MORPHO_ISLIST(dest)) {
        objectlist *lst = MORPHO_GETLIST(dest);
        
        if (c>=list_length(lst)) {
            objectmatrix *prmat = MORPHO_GETMATRIX(prototype);
            objectmatrix *new = matrix_new(prmat->nrows, prmat->ncols, true);
            if (!new) return false;
            *sum = MORPHO_OBJECT(new);
            list_append(lst, *sum);
        } else {
            matrix_zero(MORPHO_GETMATRIX(lst->val.data[c]));
            *sum = lst->val.data[c];
        }
    }
    return true;
}

/** Copies the component of the hessian into the relevant destination if needed */
bool integral_hesssumcopy(int i, int j, value sum, value dest) {
    if (MORPHO_ISMATRIX(dest)) {
        return morpho_valuetofloat(sum, &MORPHO_GETMATRIX(dest)->elements[j*MORPHO_GETMATRIX(dest)->nrows+i]);
    } else return true;
}

/** Copies the component of the gradient into the relevant destination */
bool integral_oldgradcopy(int dim, int ndof, double *grad, value prototype, value dest) {
    bool success=false;
    if (MORPHO_ISMATRIX(dest)) {
        objectmatrix *mdest = MORPHO_GETMATRIX(dest);
        memcpy(mdest->elements, grad, sizeof(double)*dim);
        success=true;
    } else if (MORPHO_ISLIST(dest)) {
        objectlist *lst = MORPHO_GETLIST(dest);
        objectmatrix *proto = MORPHO_GETMATRIX(prototype);
        for (int i=0; i<dim; i++) {
            objectmatrix *mgrad=NULL;
            value el;
            
            if (i>=list_length(lst)) {
                mgrad=matrix_new(proto->nrows, proto->ncols, false); // Should copy prototype dimensions!
                if (mgrad) {
                    for (int k=0; k<ndof; k++) mgrad->elements[k]=grad[k*dim+i];
                    list_append(lst, MORPHO_OBJECT(mgrad));
                    success=true;
                }
            }
        }
    }
    return success;
}

/** Evaluates the gradient of a field */
bool integral_evaluategradient(vm *v, value q, value *out) {
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref) MORPHO_FAILVARGS(v, INTEGRAL_SPCLFN, GRAD_FUNCTION);
    
    /* Identify the field being referred to */
    int ifld, xfld=-1;
    for (ifld=0; ifld<elref->iref->nfields; ifld++) {
        if (MORPHO_ISFIELD(q) && MORPHO_ISSAME(elref->iref->originalfields[ifld], q)) break;
        else if (MORPHO_ISSAME(elref->qinterpolated[ifld], q)) {
            if (xfld>=0) MORPHO_FAIL(v, INTEGRAL_FLD);
            // @warning: This will fail if two fields happen to have the same value(!)
            xfld=ifld;
        }
    }
    if (xfld>=0) ifld = xfld;
    
    // Raise an error if we couldn't find it
    if (ifld>=elref->iref->nfields) MORPHO_FAIL(v, INTEGRAL_FLD);
    
    // Extract information from the field
    objectfield *fld = MORPHO_GETFIELD(elref->iref->fields[ifld]);
    int dim = elref->mesh->dim;
    
    // Allocate objects if need be. Don't bind these; these will be freed when the elref is cleared.
    if (!MORPHO_ISOBJECT(elref->qgrad[ifld])) {
        if (!integral_gradalloc(dim, fld->prototype, &elref->qgrad[ifld])) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    }
    
    bool success=false;
    
    // Evaluate gradient
    if (MORPHO_ISFESPACE(fld->fnspc)) {
        if (!elref->invj) {
            elref->invj=matrix_new(elref->g, elref->mesh->dim, false);
            
            if (elref->invj) {
                integral_prepareinvjacobian(elref->mesh->dim, elref->g, elref->vertexposn, elref->invj);
            } else MORPHO_FAIL(v, INTEGRAL_DFFEVL);
        }
        
        fespace *disc = MORPHO_GETFESPACE(fld->fnspc)->fespace;
        if (!FESPACE_HASGRADIENT(disc)) MORPHO_FAIL(v, INTEGRAL_DFFEVL);
        
        int nnodes = disc->nnodes;
        double gdata[nnodes * elref->g];
        objectmatrix gmat = MORPHO_STATICMATRIX(gdata, nnodes, elref->g);
        
        // Compute gradient in reference frame
        fespace_gradient(disc, elref->lambda, &gmat);
        
        // Compute matrix
        double fmatdata[nnodes * dim];
        objectmatrix fmat = MORPHO_STATICMATRIX(fmatdata, nnodes, dim);
        
        if (matrix_mul(&gmat, elref->invj, &fmat)!=LINALGERR_OK) MORPHO_FAIL(v, INTEGRAL_DFFEVL);
        
        for (int i=0; i<dim; i++) {
            value sum;
            
            if (integral_gradsuminit(i, fld->prototype, elref->qgrad[ifld], &sum) &&
                integrator_sumquantityweighted(nnodes, fmat.elements+i*nnodes, elref->quantities[ifld].vals, &sum)) {
                integral_gradsumcopy(i, sum, elref->qgrad[ifld]);
            } else MORPHO_FAIL(v, INTEGRAL_DFFEVL);
        }
        
        success=true;
    } else { // Old gradient calculation
        int ndof = fld->psize; // Number of degrees of freedom per element
        double grad[ndof*dim]; // Storage for gradient
        
        // Evaluate correct gradient
        if (elref->g==2) success=gradsq_evaluategradient(elref->mesh, fld, elref->nv, elref->vid, grad);
        else if (elref->g==3) success=gradsq_evaluategradient3d(elref->mesh, fld, elref->nv, elref->vid, grad);
        
        integral_oldgradcopy(dim, ndof, grad, fld->prototype, elref->qgrad[ifld]);
        success=true;
    }
    
    // Store for further use
    if (success) *out=elref->qgrad[ifld];
    
    return success;
}

static value integral_gradfn(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    if (nargs==1) {
        integral_evaluategradient(v, MORPHO_GETARG(args, 0), &out);
    } else morpho_runtimeerror(v, INTEGRAL_FLD);
    
    return out;
}

/** Evaluates the hessian of a field */
bool integral_evaluatehessian(vm *v, value q, value *out) {
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref) MORPHO_FAILVARGS(v, INTEGRAL_SPCLFN, HESS_FUNCTION);
    
    int ifld, xfld=-1;
    for (ifld=0; ifld<elref->iref->nfields; ifld++) {
        if (MORPHO_ISFIELD(q) && MORPHO_ISSAME(elref->iref->originalfields[ifld], q)) break;
        else if (MORPHO_ISSAME(elref->qinterpolated[ifld], q)) {
            if (xfld>=0) MORPHO_FAIL(v, INTEGRAL_FLD);
            xfld=ifld;
        }
    }
    if (xfld>=0) ifld = xfld;
    
    if (ifld>=elref->iref->nfields) MORPHO_FAIL(v, INTEGRAL_FLD);
    
    objectfield *fld = MORPHO_GETFIELD(elref->iref->fields[ifld]);
    int dim = elref->mesh->dim;
    
    if (!MORPHO_ISOBJECT(elref->qhess[ifld])) {
        if (!integral_hessalloc(dim, fld->prototype, &elref->qhess[ifld])) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    }
    
    if (MORPHO_ISFESPACE(fld->fnspc)) {
        if (!elref->invj) {
            elref->invj=matrix_new(elref->g, elref->mesh->dim, false);
            if (elref->invj) {
                integral_prepareinvjacobian(elref->mesh->dim, elref->g, elref->vertexposn, elref->invj);
            } else MORPHO_FAIL(v, INTEGRAL_DFFEVL);
        }
        
        fespace *disc = MORPHO_GETFESPACE(fld->fnspc)->fespace;
        if (!FESPACE_HASHESSIAN(disc)) MORPHO_FAIL(v, INTEGRAL_DFFEVL);
        int nnodes = disc->nnodes;
        double hdata[nnodes * elref->g * elref->g];
        objectmatrix hmat = MORPHO_STATICMATRIX(hdata, nnodes, elref->g*elref->g);
        
        fespace_hessian(disc, elref->lambda, &hmat);
        
        double fdata[nnodes * dim * dim];
        for (int j=0; j<dim; j++) {
            for (int i=0; i<dim; i++) {
                double *outcol = fdata + (j*dim+i)*nnodes;
                for (int n=0; n<nnodes; n++) {
                    double sum=0.0;
                    for (int qref=0; qref<elref->g; qref++) {
                        for (int pref=0; pref<elref->g; pref++) {
                            sum += hdata[(qref*elref->g+pref)*nnodes+n] *
                                   elref->invj->elements[i*elref->g+pref] *
                                   elref->invj->elements[j*elref->g+qref];
                        }
                    }
                    outcol[n]=sum;
                }
            }
        }
        
        for (int j=0; j<dim; j++) {
            for (int i=0; i<dim; i++) {
                int c = j*dim+i;
                value sum;
                
                if (integral_hesssuminit(c, fld->prototype, elref->qhess[ifld], &sum) &&
                    integrator_sumquantityweighted(nnodes, fdata+c*nnodes, elref->quantities[ifld].vals, &sum)) {
                    integral_hesssumcopy(i, j, sum, elref->qhess[ifld]);
                } else MORPHO_FAIL(v, INTEGRAL_DFFEVL);
            }
        }
        
        *out=elref->qhess[ifld];
        return true;
    }
    
    MORPHO_FAIL(v, INTEGRAL_DFFEVL);
}

static value integral_hessfn(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;
    if (nargs==1) {
        integral_evaluatehessian(v, MORPHO_GETARG(args, 0), &out);
    } else morpho_runtimeerror(v, INTEGRAL_FLD);
    
    return out;
}

/* -------------------
 * Cauchy green strain
 * ------------------- */

int cauchygreenhandle; // TL storage handle for CG tensor

/** Evaluates the cg strain tensor */
void integral_evaluatecg(vm *v, value *out) {
    objectintegralelementref *elref = integral_getelementref(v);
    
    if (!elref || !elref->iref->mref) {
        morpho_runtimeerror(v, INTEGRAL_SPCLFN, CGTENSOR_FUNCTION); return;
    }
    
    int gdim=elref->nv-1; // Dimension of Gram matrix
    
    objectmatrix *cg=matrix_new(gdim, gdim, true);
    if (!cg) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return; }
    
    double gramrefel[gdim*gdim], gramdefel[gdim*gdim], qel[gdim*gdim], rel[gdim*gdim];
    objectmatrix gramref = MORPHO_STATICMATRIX(gramrefel, gdim, gdim); // Gram matrices
    objectmatrix gramdef = MORPHO_STATICMATRIX(gramdefel, gdim, gdim); //
    objectmatrix q = MORPHO_STATICMATRIX(qel, gdim, gdim); // Inverse of Gram in source domain
    objectmatrix r = MORPHO_STATICMATRIX(rel, gdim, gdim); // Intermediate calculations
    
    linearelasticity_calculategram(elref->iref->mref->vert, elref->mesh->dim, elref->nv, elref->vid, &gramref);
    linearelasticity_calculategram(elref->mesh->vert, elref->mesh->dim, elref->nv, elref->vid, &gramdef);
    
    if (matrix_copy(&gramref, &q)!=LINALGERR_OK) return;
    if (matrix_inverse(&q)!=LINALGERR_OK) return;
    if (matrix_mul(&gramdef, &q, &r)!=LINALGERR_OK) return;

    if (matrix_identity(cg)!=LINALGERR_OK) return;
    matrix_scale(cg, -0.5);
    matrix_axpy(0.5, &r, cg);
    
    vm_settlvar(v, cauchygreenhandle, MORPHO_OBJECT(cg));
    *out = MORPHO_OBJECT(cg);
}

static value integral_cgfn(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;

    vm_gettlvar(v, cauchygreenhandle, &out);
    if (MORPHO_ISNIL(out)) integral_evaluatecg(v, &out);
    
    return out;
}

/* -------------------
 * Jacobian
 * ------------------- */

/*
 * A reference triangle is mapped to a target triangle through a
 * linear transformation (the pushforward); an inverse transformation (the pullback)
 * exists if the triangle is not degenerate. This function computes the forward
 * and inverse jacobians.
 */

int jacobianhandle; // TL storage handle for Jacobian
int invjacobianhandle; // TL storage handle for inverse Jacobian

void _fetchvertices(objectintegralelementref *elref, objectmesh *mesh, int nv, elementid *vid, double **x) {
    // Fetch reference vertices
    for (int j=0; j<nv; j++) matrix_getcolumnptr(elref->iref->mref->vert, vid[j], &x[j]);
}

void _edgevectors(grade g, int dim, double **x, double *out) {
    for (int i=0; i<g; i++) functional_vecsub(dim, x[i+1], x[0], out + i*dim);
}

/** Evaluates the jacobian and inverse jacobian; returns either of these as requested */
void integral_evaluatejacobian(vm *v, value *jac, value *invjac) {
    objectintegralelementref *elref = integral_getelementref(v);
    
    if (!elref) {
        morpho_runtimeerror(v, INTEGRAL_SPCLFN, JACOBIAN_FUNCTION); return;
    }
    
    int dim = elref->mesh->dim;     // Dimension of the mesh
    
    // Allocate matrices
    objectmatrix *J=matrix_new(dim, dim, true);
    objectmatrix *Jinv=matrix_new(dim, dim, true);
    
    if (J) vm_settlvar(v, jacobianhandle, MORPHO_OBJECT(J));
    if (Jinv) vm_settlvar(v, invjacobianhandle, MORPHO_OBJECT(Jinv));
    
    if (!J || !Jinv) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return; }
    
    // Now compute them
    grade g = elref->g;             // Grade of the element
    int nv = elref->nv;             //
    
    double **X = elref->vertexposn; // Vertex positions of the target element
    double *x[nv];                  // Vertex positions of the reference element
    
    objectmesh *mref = elref->iref->mref; // Reference mesh
    if (mref) _fetchvertices(elref, mref, nv, elref->vid, x);
    
    // Construct matrix of edge vectors for target and reference elements
    double starget[dim*dim], sinv[dim*dim];
    objectmatrix St = MORPHO_STATICMATRIX(starget, dim, dim),
                 Sinv = MORPHO_STATICMATRIX(sinv, dim, dim);
    
    _edgevectors(g, dim, X, starget);
    if (mref) {
        _edgevectors(g, dim, x, sinv);
        matrix_inverse(&Sinv);
    } else {
        matrix_identity(&Sinv); // If no reference, the reference is the unit triangle
    }
    
    matrix_mul(&St, &Sinv, J); // J = S . s^-1
    matrix_copy(J, Jinv);
    matrix_inverse(Jinv); // Compute J^-1
    
    if (jac) *jac = MORPHO_OBJECT(J);
    if (invjac) *invjac = MORPHO_OBJECT(Jinv);
}

static value integral_jacobian(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;

    vm_gettlvar(v, jacobianhandle, &out);
    if (MORPHO_ISNIL(out)) integral_evaluatejacobian(v, &out, NULL);
    
    return out;
}

static value integral_invjacobian(vm *v, int nargs, value *args) {
    value out=MORPHO_NIL;

    vm_gettlvar(v, invjacobianhandle, &out);
    if (MORPHO_ISNIL(out)) integral_evaluatejacobian(v, NULL, &out);
    
    return out;
}

/* ----------------------
 * General initialization
 * ---------------------- */

/** Clears threadlocal storage */
void integral_cleartlvars(vm *v) {
    int handles[] = { elementhandle, normlhandle, tangenthandle, cauchygreenhandle, jacobianhandle, invjacobianhandle, -1 };
    
    for (int i=0; handles[i]>=0; i++) {
        vm_settlvar(v, handles[i], MORPHO_NIL);
    }
}

void integral_freetlvars(vm *v) {
    int handles[] = { normlhandle, tangenthandle, cauchygreenhandle,jacobianhandle, invjacobianhandle, -1 };
    
    for (int i=0; handles[i]>=0; i++) {
        value val;
        vm_gettlvar(v, handles[i], &val);
        if (MORPHO_ISOBJECT(val)) morpho_freeobject(val);
    }
    
    integral_cleartlvars(v);
}

/* ----------------------------------------------
 * Generic integral support functions
 * ---------------------------------------------- */

value functional_methodproperty;

/** Prepares an integral reference */
bool integral_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, integralref *ref) {
    bool success=false;
    value func=MORPHO_NIL;
    value mref=MORPHO_NIL;
    value wtbyref=MORPHO_NIL;
    value field=MORPHO_NIL;
    value method=MORPHO_NIL;
    ref->v=NULL;
    ref->nfields=0;
    ref->method=MORPHO_NIL;
    ref->mref=NULL;
    ref->g=g;
    ref->weightbyref=false;

    if (objectinstance_getpropertyinterned(self, scalarpotential_functionproperty, &func) &&
        MORPHO_ISCALLABLE(func)) {
        ref->integrand=func;
        success=true;
    }
    if (objectinstance_getpropertyinterned(self, linearelasticity_referenceproperty, &mref) &&
        MORPHO_ISMESH(mref)) {
        ref->mref=MORPHO_GETMESH(mref);
    }
    if (objectinstance_getpropertyinterned(self, linearelasticity_weightbyreferenceproperty, &wtbyref)) {
        ref->weightbyref=!morpho_isfalse(wtbyref);
    }
    if (objectinstance_getpropertyinterned(self, functional_methodproperty, &method)) {
        ref->method=method;
    }
    if (objectinstance_getpropertyinterned(self, functional_fieldproperty, &field) &&
        MORPHO_ISLIST(field)) {
        objectlist *list = MORPHO_GETLIST(field);
        ref->nfields=list->val.count;
        ref->fields=list->val.data;
        ref->originalfields=list->val.data;
        
        for (int i=0; i<ref->nfields; i++) {
            if (MORPHO_ISFIELD(ref->fields[i])) {
                objectfield *fld = MORPHO_GETFIELD(ref->fields[i]);
                field_addpool(fld);
            }
        }
    }
    return success;
}

/** Clones the integral reference with a given substitute field */
void *integral_cloneref(void *ref, objectfield *field, objectfield *sub) {
    integralref *nref = (integralref *) ref;
    integralref *clone = MORPHO_MALLOC(sizeof(integralref));
    
    if (clone) {
        *clone = *nref;
        clone->originalfields=nref->originalfields;
        clone->fields=MORPHO_MALLOC(sizeof(value)*clone->nfields);
        if (!clone->fields) { MORPHO_FREE(clone); return NULL; }
        
        for (int i=0; i<clone->nfields; i++) {
            clone->fields[i]=nref->fields[i];
            if (MORPHO_ISFIELD(nref->fields[i]) &&
                MORPHO_GETFIELD(nref->fields[i])==field) {
                clone->fields[i]=MORPHO_OBJECT(sub);
            }
        }
    }
    
    return clone;
}

/** Frees a reference */
void integral_freeref(void *ref) {
    integralref *nref = (integralref *) ref;
    MORPHO_FREE(nref->fields);
    MORPHO_FREE(ref);
}

/** Clears any data in an element ref */
void integral_clearelref(objectintegralelementref *elref) {
    if (elref->invj) object_free((object *) elref->invj);
}

/** Free cached field gradients and hessians for an element */
static void integral_freegradhess(int nfields, value *qgrad, value *qhess) {
    for (int i=0; i<nfields; i++) {
        if (MORPHO_ISLIST(qgrad[i])) {
            objectlist *l = MORPHO_GETLIST(qgrad[i]);
            for (int j=0; j<l->val.count; j++) morpho_freeobject(l->val.data[j]);
        }
        morpho_freeobject(qgrad[i]);
        if (MORPHO_ISLIST(qhess[i])) {
            objectlist *l = MORPHO_GETLIST(qhess[i]);
            for (int j=0; j<l->val.count; j++) morpho_freeobject(l->val.data[j]);
        }
        morpho_freeobject(qhess[i]);
    }
}

/** Prepares quantity list */
bool integral_preparequantities(integralref *iref, int nv, int *vid, quantity *quantities) {
    bool success=true;
    for (int k=0; k<iref->nfields; k++) {
        objectfield *f=MORPHO_GETFIELD(iref->fields[k]);
        quantities[k].vals=NULL;
        
        if (MORPHO_ISFESPACE(f->fnspc)) {
            fespace *disc=MORPHO_GETFESPACE(f->fnspc)->fespace;
            if (nv-1<disc->grade) {
                if (!fespace_lower(disc, nv-1, &disc)) return false;
            }
            
            quantities[k].nnodes=disc->nnodes;
            quantities[k].ifn=disc->ifn;
            
            fieldindx findx[disc->nnodes];
            if (!fespace_doftofieldindx(f, disc, nv, vid, findx)) return false;
            
            quantities[k].vals=MORPHO_MALLOC(sizeof(value)*disc->nnodes);
            if (!quantities[k].vals) return false;
            for (int i=0; i<disc->nnodes; i++) {
                int dof;
                if (!field_getindex(f, findx[i].g, findx[i].id, findx[i].indx, &dof)) return false;
                if (!field_getelementwithindex(f, dof, &quantities[k].vals[i])) return false;
            }
            success=true;
        } else {
            quantities[k].nnodes=nv;
            quantities[k].ifn=NULL;
            quantities[k].vals=MORPHO_MALLOC(sizeof(value)*nv);
            if (!quantities[k].vals) return false;
            for (unsigned int i=0; i<nv; i++) {
                if (!field_getelement(f, MESH_GRADE_VERTEX, vid[i], 0, &quantities[k].vals[i])) return false;
            }
            success=true; 
        }
    }
    return success;
}

/** Clears a list of quantities */
void integral_clearquantities(int nq, quantity *quantities) {
    for (int k=0; k<nq; k++) {
        if (quantities[k].vals) MORPHO_FREE(quantities[k].vals);
    }
}

bool integral_integrandfn(unsigned int dim, double *t, double *x, unsigned int nquantity, value *quantity, void *ref, double *fout) {
    integralref *iref = ref;
    objectmatrix posn = MORPHO_STATICMATRIX(x, dim, 1);
    value args[nquantity+1], out;

    // The integrand function is called with the position and then interpolated quantities.
    args[0]=MORPHO_OBJECT(&posn);
    for (unsigned int i=0; i<nquantity; i++) args[i+1]=quantity[i];
    
    objectintegralelementref *elref = integral_getelementref(iref->v);
    if (elref) {
        elref->lambda=t;
        elref->posn=x;
        elref->qinterpolated=quantity;
    }
    
    if (morpho_call(iref->v, iref->integrand, nquantity+1, args, &out)) {
        morpho_valuetofloat(out, fout);
        return true;
    }

    return false;
}

/** Integrate a callable over elements of the grade stored in the integral ref */
bool integral_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    integralref iref = *(integralref *) ref;
    grade g = iref.g;
    double *x[nv];
    bool success;
    value qgrad[iref.nfields+1], qhess[iref.nfields+1];
    for (int i=0; i<iref.nfields; i++) { qgrad[i] = MORPHO_NIL; qhess[i] = MORPHO_NIL; }
    
    objectintegralelementref elref = MORPHO_STATICINTEGRALELEMENTREF(mesh, g, id, nv, vid);
    elref.iref = &iref;
    elref.vertexposn = x;
    elref.qgrad=qgrad;
    elref.qhess=qhess;
    elref.invj=NULL;
    
    objectmesh *sizemesh = (iref.weightbyref ? iref.mref : mesh);
    if (!functional_elementsize(v, sizemesh, g, id, nv, vid, &elref.elementsize)) return false;

    iref.v=v;
    for (unsigned int i=0; i<nv; i++) {
        mesh_getvertexcoordinatesaslist(mesh, vid[i], &x[i]);
    }

    /* Set up quantities */
    integral_cleartlvars(v);
    vm_settlvar(v, elementhandle, MORPHO_OBJECT(&elref));

    if (MORPHO_ISDICTIONARY(iref.method)) {
        double err;
        quantity quantities[iref.nfields+1];
        if (!integral_preparequantities(&iref, nv, vid, quantities)) {
            integral_clearelref(&elref);
            return false;
        }
        elref.quantities=quantities;
        
        success=integrate(integral_integrandfn, MORPHO_GETDICTIONARY(iref.method), morpho_geterror(v), mesh->dim, g, x, iref.nfields, quantities, &iref, out, &err);
        
        integral_clearquantities(iref.nfields, quantities);
        integral_clearelref(&elref);
    } else { // Old integrator
        value qstore[nv][iref.nfields+1];
        value *q[nv];
        for (unsigned int i=0; i<nv; i++) q[i]=qstore[i];
        for (unsigned int k=0; k<iref.nfields; k++) {
            for (unsigned int i=0; i<nv; i++) {
                field_getelement(MORPHO_GETFIELD(iref.fields[k]), MESH_GRADE_VERTEX, vid[i], 0, &q[i][k]);
            }
        }
        
        success=integrate_integrate(integral_integrandfn, mesh->dim, g, x, iref.nfields, q, &iref, out);
    }
    
    if (success) *out *= elref.elementsize;

    integral_freetlvars(v);
    integral_freegradhess(iref.nfields, qgrad, qhess);
    
    return success;
}

/** Shared method path for Line/Area/Volume integrals */
static value integral_domap(vm *v, int nargs, value *args, bool (*mapfn)(vm *, functional_mapinfo *, value *)) {
    functional_mapinfo info;
    integralref ref;
    value out=MORPHO_NIL;
    
    if (functional_validateargs(v, nargs, args, &info)) {
        objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
        grade g=0;
        if (!functional_readgrade(self, &g) ||
            !integral_prepareref(self, info.mesh, g, info.sel, &ref)) {
            morpho_runtimeerror(v, INTEGRAL_ARGS);
        } else {
            info.integrand = integral_integrand;
            info.start = integral_startfn;
            info.dependencies = NULL;
            info.sym = SYMMETRY_NONE;
            info.g = g;
            info.ref = &ref;
            functional_runmap(v, &info, mapfn, &out);
        }
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

static value Integral_integrand(vm *v, int nargs, value *args) {
    return integral_domap(v, nargs, args, functional_mapintegrand);
}

static value Integral_total(vm *v, int nargs, value *args) {
    return integral_domap(v, nargs, args, functional_sumintegrand);
}

static value Integral_gradient(vm *v, int nargs, value *args) {
    return integral_domap(v, nargs, args, functional_mapnumericalgradient);
}

static value Integral_hessian(vm *v, int nargs, value *args) {
    return integral_domap(v, nargs, args, functional_mapnumericalhessian);
}

/** Field gradients for Line/Area/Volume integrals */
static value Integral_fieldgradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    integralref ref;
    value out=MORPHO_NIL;
    
    if (functional_validateargs(v, nargs, args, &info)) {
        objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
        grade g=0;
        if (!functional_readgrade(self, &g) ||
            !integral_prepareref(self, info.mesh, g, info.sel, &ref)) {
            morpho_runtimeerror(v, INTEGRAL_ARGS);
        } else {
            info.g=g;
            info.integrand=integral_integrand;
            info.start=integral_startfn;
            info.cloneref=integral_cloneref;
            info.freeref=integral_freeref;
            info.ref=&ref;
            functional_runmap(v, &info, functional_mapnumericalfieldgradient, &out);
        }
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

/** Initialize a Line/Area/Volume/Jump integral object */
static value integral_init(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    int nparams = -1;
    int nfixed;
    value method=MORPHO_NIL;
    value mref=MORPHO_NIL;
    value wtbyref=MORPHO_NIL;

    if (builtin_options(v, nargs, args, &nfixed, 3,
                        functional_methodproperty, &method,
                        linearelasticity_referenceproperty, &mref,
                        linearelasticity_weightbyreferenceproperty, &wtbyref)) {
        if (MORPHO_ISDICTIONARY(method)) {
            objectinstance_setproperty(self, functional_methodproperty, method);
        } else if (!MORPHO_ISNIL(method)) {
            morpho_runtimeerror(v, INTEGRAL_ARGS);
        }

        if (MORPHO_ISMESH(mref)) objectinstance_setproperty(self, linearelasticity_referenceproperty, mref);
        if (MORPHO_ISBOOL(wtbyref)) objectinstance_setproperty(self, linearelasticity_weightbyreferenceproperty, wtbyref);
    } else MORPHO_RAISE(v, INTEGRAL_ARGS);
    
    if (nfixed>0) {
        value f = MORPHO_GETARG(args, 0);

        if (morpho_countparameters(f, &nparams)) {
            objectinstance_setproperty(self, scalarpotential_functionproperty, MORPHO_GETARG(args, 0));
        } else MORPHO_RAISE(v, INTEGRAL_ARGS);
    }

    if (nparams!=nfixed) MORPHO_RAISE(v, INTEGRAL_ARGS);

    if (nfixed>1) {
        /* Remaining arguments should be fields */
        objectlist *list = object_newlist(nfixed-1, & MORPHO_GETARG(args, 1));
        if (!list) MORPHO_RAISE(v, ERROR_ALLOCATIONFAILED);

        for (unsigned int i=1; i<nfixed; i++) {
            if (!MORPHO_ISFIELD(MORPHO_GETARG(args, i))) {
                morpho_runtimeerror(v, INTEGRAL_ARGS);
                object_free((object *) list);
                return MORPHO_NIL;
            }
        }

        value field = MORPHO_OBJECT(list);
        objectinstance_setproperty(self, functional_fieldproperty, field);
        morpho_bindobjects(v, 1, &field);
    }

    return MORPHO_NIL;
}

static value integral_initwithgrade(vm *v, int nargs, value *args, grade g) {
    objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), functional_gradeproperty, MORPHO_INTEGER(g));
    return integral_init(v, nargs, args);
}

value LineIntegral_init(vm *v, int nargs, value *args) {
    return integral_initwithgrade(v, nargs, args, MESH_GRADE_LINE);
}

value AreaIntegral_init(vm *v, int nargs, value *args) {
    return integral_initwithgrade(v, nargs, args, MESH_GRADE_AREA);
}

value VolumeIntegral_init(vm *v, int nargs, value *args) {
    return integral_initwithgrade(v, nargs, args, MESH_GRADE_VOLUME);
}

#define INTEGRAL_METHODFLAGS (MORPHO_FN_REENTRANT|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED)
#define INTEGRAL_TOTALFLAGS  (MORPHO_FN_REENTRANT|MORPHO_FN_THROWS|MORPHO_FN_MULTITHREADED)

MORPHO_BEGINCLASS(LineIntegral)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, LineIntegral_init, MORPHO_FN_MUTATES|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, Integral_integrand, INTEGRAL_METHODFLAGS),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, Integral_total, INTEGRAL_TOTALFLAGS),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, Integral_gradient, INTEGRAL_METHODFLAGS),
MORPHO_METHOD(FUNCTIONAL_FIELDGRADIENT_METHOD, Integral_fieldgradient, INTEGRAL_METHODFLAGS),
MORPHO_METHOD(FUNCTIONAL_HESSIAN_METHOD, Integral_hessian, INTEGRAL_METHODFLAGS)
MORPHO_ENDCLASS

MORPHO_BEGINCLASS(AreaIntegral)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, AreaIntegral_init, MORPHO_FN_MUTATES|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, Integral_integrand, INTEGRAL_METHODFLAGS),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, Integral_total, INTEGRAL_TOTALFLAGS),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, Integral_gradient, INTEGRAL_METHODFLAGS),
MORPHO_METHOD(FUNCTIONAL_FIELDGRADIENT_METHOD, Integral_fieldgradient, INTEGRAL_METHODFLAGS),
MORPHO_METHOD(FUNCTIONAL_HESSIAN_METHOD, Integral_hessian, INTEGRAL_METHODFLAGS)
MORPHO_ENDCLASS

MORPHO_BEGINCLASS(VolumeIntegral)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, VolumeIntegral_init, MORPHO_FN_MUTATES|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, Integral_integrand, INTEGRAL_METHODFLAGS),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, Integral_total, INTEGRAL_TOTALFLAGS),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, Integral_gradient, INTEGRAL_METHODFLAGS),
MORPHO_METHOD(FUNCTIONAL_FIELDGRADIENT_METHOD, Integral_fieldgradient, INTEGRAL_METHODFLAGS),
MORPHO_METHOD(FUNCTIONAL_HESSIAN_METHOD, Integral_hessian, INTEGRAL_METHODFLAGS)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Jump
 * ---------------------------------------------- */

struct jumpref_s {
    integralref integral;
    grade parentgrade;
    grade interfacegrade;
    objectsparse *interfaceparents;
    objectsparse *parentinterfaces;
    objectsparse *parentvertices;
    jumpstrategy strategy;
};

static bool jump_preparetopology(vm *v, objectmesh *mesh, jumpref *ref) {
    ref->parentgrade=mesh_maxgrade(mesh);
    if (ref->parentgrade<1) MORPHO_FAILVARGS(v, FUNC_ELNTFND, (unsigned int) MESH_GRADE_LINE);

    ref->interfacegrade=ref->parentgrade-1;
    ref->interfaceparents=mesh_addconnectivityelement(mesh, ref->parentgrade, ref->interfacegrade);
    ref->parentinterfaces=mesh_addconnectivityelement(mesh, ref->interfacegrade, ref->parentgrade);
    ref->parentvertices=mesh_getconnectivityelement(mesh, 0, ref->parentgrade);

    if (!ref->parentvertices) MORPHO_FAILVARGS(v, FUNC_ELNTFND, (unsigned int) ref->parentgrade);
    if (!ref->interfaceparents || !ref->parentinterfaces) MORPHO_FAILVARGS(v, FUNC_ELNTFND, (unsigned int) ref->interfacegrade);

    return true;
}

static bool jump_preparestrategy(vm *v, jumpref *ref) {
    ref->strategy=JUMP_STRATEGY_CENTROID_MODE;

    if (!MORPHO_ISDICTIONARY(ref->integral.method)) return true;

    objectdictionary *dict=MORPHO_GETDICTIONARY(ref->integral.method);
    objectstring strategylabel = MORPHO_STATICSTRING(JUMP_STRATEGY_LABEL);
    value val=MORPHO_NIL;

    if (!dictionary_get(&dict->dict, MORPHO_OBJECT(&strategylabel), &val)) return true;
    if (!MORPHO_ISSTRING(val)) MORPHO_FAIL(v, FUNCTIONAL_ARGS);

    char *strategy=MORPHO_GETCSTRING(val);
    if (strcmp(strategy, JUMP_STRATEGY_CENTROID)==0) {
        ref->strategy=JUMP_STRATEGY_CENTROID_MODE;
        return true;
    }
    if (strcmp(strategy, JUMP_STRATEGY_QUADRATURE)==0) {
        ref->strategy=JUMP_STRATEGY_QUADRATURE_MODE;
        return true;
    }

    MORPHO_FAIL(v, FUNCTIONAL_ARGS);
}

/** Initialize a Jump object.
    For now this matches the existing integral optional-argument surface:
    'method', 'mref' and 'weightbyreference'. */
static value Jump_init(vm *v, int nargs, value *args) {
    value ret = integral_init(v, nargs, args);
    if (nargs>1 && MORPHO_ISFIELD(MORPHO_GETARG(args, 1))) {
        grade g = mesh_maxgrade(MORPHO_GETFIELD(MORPHO_GETARG(args, 1))->mesh);
        if (g>0) g--;
        objectinstance_setproperty(MORPHO_GETINSTANCE(MORPHO_SELF(args)), functional_gradeproperty, MORPHO_INTEGER(g));
    }
    return ret;
}

/** Prepare a jump reference.
    Shared functional metadata is handled by integral_prepareref; Jump only adds
    codimension-1 topology needed for interior-interface traversal. */
static bool jump_prepareref(vm *v, objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, jumpref *ref) {
    ref->parentgrade=0;
    ref->interfacegrade=0;
    ref->interfaceparents=NULL;
    ref->parentinterfaces=NULL;
    ref->parentvertices=NULL;
    ref->strategy=JUMP_STRATEGY_CENTROID_MODE;

    if (!integral_prepareref(self, mesh, g, sel, &ref->integral)) MORPHO_FAIL(v, INTEGRAL_ARGS);
    if (!jump_preparetopology(v, mesh, ref)) return false;
    if (!jump_preparestrategy(v, ref)) return false;
    objectinstance_setproperty(self, functional_gradeproperty, MORPHO_INTEGER(ref->interfacegrade));
    return true;
}

static bool jump_startfn(vm *v, functional_mapinfo *info) {
    jumpref *ref = (jumpref *) info->ref;
    return functional_preparefieldlist(v, ref->integral.fields, ref->integral.nfields, ref->parentgrade);
}

/** Clone a jump reference with a substituted field. */
static void *jump_cloneref(void *ref, objectfield *field, objectfield *sub) {
    jumpref *nref = (jumpref *) ref;
    jumpref *clone = MORPHO_MALLOC(sizeof(jumpref));

    if (clone) {
        *clone = *nref;
        clone->integral.originalfields=nref->integral.originalfields;
        clone->integral.fields=MORPHO_MALLOC(sizeof(value)*clone->integral.nfields);
        if (!clone->integral.fields) { MORPHO_FREE(clone); return NULL; }

        for (int i=0; i<clone->integral.nfields; i++) {
            clone->integral.fields[i]=nref->integral.fields[i];
            if (MORPHO_ISFIELD(nref->integral.fields[i]) &&
                MORPHO_GETFIELD(nref->integral.fields[i])==field) {
                clone->integral.fields[i]=MORPHO_OBJECT(sub);
            }
        }
    }

    return clone;
}

/** Free a cloned jump reference. */
static void jump_freeref(void *ref) {
    jumpref *nref = (jumpref *) ref;
    MORPHO_FREE(nref->integral.fields);
    MORPHO_FREE(ref);
}

/** Get the adjacent parent elements for an interface. */
static bool jump_getadjacentparents(jumpref *ref, elementid interfaceid, int *nparents, int **parents) {
    if (!ref->interfaceparents) return false;
    return mesh_getconnectivity(ref->interfaceparents, interfaceid, nparents, parents);
}

/** Return mesh vertices outside the interface that still influence the jump term
    through the two adjacent parent elements. */
static bool jump_dependencies(functional_mapinfo *info, elementid id, varray_elementid *out) {
    jumpref *ref = (jumpref *) info->ref;
    int nparents=0, *parents=NULL;

    if (!jump_getadjacentparents(ref, id, &nparents, &parents)) return false;
    if (nparents!=2) return true;

    int interface_nv=0, *interface_vid=NULL;
    objectsparse *ifaceverts=NULL;
    int n=0;

    if (!functional_countelements(NULL, info->mesh, ref->interfacegrade, &n, &ifaceverts)) return false;
    (void) n;
    if (!ifaceverts) return false;
    if (!sparseccs_getrowindices(&ifaceverts->ccs, id, &interface_nv, &interface_vid)) return false;

    for (int p=0; p<nparents; p++) {
        int parent_nv=0, *parent_vid=NULL;
        if (!mesh_getconnectivity(ref->parentvertices, parents[p], &parent_nv, &parent_vid)) return false;
        for (int j=0; j<parent_nv; j++) {
            if (!functional_containsvertex(interface_nv, interface_vid, parent_vid[j])) {
                varray_elementidwriteunique(out, parent_vid[j]);
            }
        }
    }

    return true;
}

/** Return interface elements that share one of the adjacent parent elements.
    This is appropriate for coordinate gradients, but not for FE field gradients:
    in the FE case the local parent-element DOF collection already captures the
    two-sided support of the interface term, and the outer interface traversal
    accounts for neighboring interfaces exactly once. */
static bool jump_fielddependencies(functional_mapinfo *info, elementid id, varray_elementid *out) {
    jumpref *ref = (jumpref *) info->ref;
    int nparents=0, *parents=NULL;

    if (!jump_getadjacentparents(ref, id, &nparents, &parents)) return false;
    if (nparents!=2) return true;

    for (int p=0; p<nparents; p++) {
        int nifaces=0, *ifaces=NULL;
        if (!mesh_getconnectivity(ref->parentinterfaces, parents[p], &nifaces, &ifaces)) return false;
        for (int j=0; j<nifaces; j++) {
            if (ifaces[j]!=id) varray_elementidwriteunique(out, ifaces[j]);
        }
    }

    return true;
}

static bool jump_preparejumpside(jumpref *ref, int nv, int *vid, jumpside *trace) {
    trace->nv=nv;
    trace->vid=vid;
    trace->quantities=NULL;
    if (ref->integral.nfields==0) return true;

    trace->quantities=MORPHO_MALLOC(sizeof(quantity)*ref->integral.nfields);
    if (!trace->quantities) return false;

    for (int i=0; i<ref->integral.nfields; i++) {
        trace->quantities[i].nnodes=0;
        trace->quantities[i].vals=NULL;
        trace->quantities[i].ifn=NULL;
        trace->quantities[i].ndof=0;
    }

    if (!integral_preparequantities(&ref->integral, nv, vid, trace->quantities)) {
        integral_clearquantities(ref->integral.nfields, trace->quantities);
        MORPHO_FREE(trace->quantities);
        trace->quantities=NULL;
        return false;
    }

    return true;
}

static void jump_clearjumpside(jumpref *ref, jumpside *trace) {
    if (trace->quantities) {
        integral_clearquantities(ref->integral.nfields, trace->quantities);
        MORPHO_FREE(trace->quantities);
        trace->quantities=NULL;
    }
}

static void jump_clearinterfaceref(objectjumpinterfaceref *iref) {
    jump_clearjumpside(iref->jref, &iref->qplus);
    jump_clearjumpside(iref->jref, &iref->qminus);
    if (iref->pluslambda) {
        MORPHO_FREE(iref->pluslambda);
        iref->pluslambda=NULL;
    }
    if (iref->minuslambda) {
        MORPHO_FREE(iref->minuslambda);
        iref->minuslambda=NULL;
    }
    if (iref->normal) {
        object_free((object *) iref->normal);
        iref->normal=NULL;
    }
}

static void jump_orderparents(int *parents, elementid *plusid, elementid *minusid) {
    if (parents[0]<parents[1]) {
        *plusid=parents[0]; *minusid=parents[1];
    } else {
        *plusid=parents[1]; *minusid=parents[0];
    }
}

static bool jump_getinterfacevertexpositions(objectmesh *mesh, int nv, int *vid, double **x) {
    for (int i=0; i<nv; i++) {
        if (!mesh_getvertexcoordinatesaslist(mesh, vid[i], &x[i])) return false;
    }
    return true;
}

static void jump_centroid(unsigned int dim, int nv, double **x, double *out) {
    for (unsigned int i=0; i<dim; i++) out[i]=0.0;
    for (int i=0; i<nv; i++) {
        for (unsigned int j=0; j<dim; j++) out[j]+=x[i][j];
    }
    functional_vecscale(dim, 1.0/nv, out, out);
}

static bool jump_getelementcentroid(objectmesh *mesh, int nv, int *vid, double *centroid) {
    int dim=mesh->dim;
    for (int i=0; i<dim; i++) centroid[i]=0.0;

    for (int i=0; i<nv; i++) {
        double *x=NULL;
        if (!mesh_getvertexcoordinatesaslist(mesh, vid[i], &x)) return false;
        for (int j=0; j<dim; j++) centroid[j]+=x[j];
    }

    functional_vecscale(dim, 1.0/nv, centroid, centroid);
    return true;
}

static bool jump_parentlambda(unsigned int dim, grade g, double **x, double *posn, double *lambda) {
    double invjdata[g*dim], sdata[dim];
    objectmatrix invj = MORPHO_STATICMATRIX(invjdata, g, dim);
    objectmatrix s = MORPHO_STATICMATRIX(sdata, dim, 1);
    objectmatrix l = MORPHO_STATICMATRIX(lambda+1, g, 1);

    functional_vecsub(dim, posn, x[0], sdata);

    if (!integral_prepareinvjacobian(dim, g, x, &invj)) return false;
    if (matrix_mul(&invj, &s, &l)!=LINALGERR_OK) return false;

    lambda[0]=1.0;
    for (int i=1; i<g+1; i++) lambda[0]-=lambda[i];

    return true;
}

static bool jump_interpolatequantity(quantity *q, grade g, double *lambda, value *out) {
    int nnodes=q->nnodes;
    double wts[nnodes];

    if (q->ifn) {
        (q->ifn) (lambda, wts);
    } else {
        if (nnodes!=1) return false;
        wts[0]=1.0;
    }

    return integrator_sumquantityweighted(nnodes, wts, q->vals, out);
}

static bool jump_preparepointdata(objectjumpinterfaceref *iref, double *posn, value *qinterp) {
    jumpref *ref=iref->jref;
    double *xplus[iref->plusnv], *xminus[iref->minusnv];

    if (!jump_getinterfacevertexpositions(iref->mesh, iref->plusnv, iref->plusvid, xplus) ||
        !jump_getinterfacevertexpositions(iref->mesh, iref->minusnv, iref->minusvid, xminus) ||
        !jump_parentlambda(iref->mesh->dim, ref->parentgrade, xplus, posn, iref->pluslambda) ||
        !jump_parentlambda(iref->mesh->dim, ref->parentgrade, xminus, posn, iref->minuslambda)) return false;

    iref->posn=posn;
    iref->qinterpolated=qinterp;

    for (int i=0; i<ref->integral.nfields; i++) {
        if (!jump_interpolatequantity(&iref->qplus.quantities[i], ref->parentgrade, iref->pluslambda, &qinterp[i])) return false;
    }

    return true;
}

static bool jump_callintegrand(objectjumpinterfaceref *iref, double *posn, double *out) {
    jumpref *ref=iref->jref;
    value qinterp[ref->integral.nfields+1], args[ref->integral.nfields+1], outval=MORPHO_NIL;
    objectmatrix mposn = MORPHO_STATICMATRIX(posn, iref->mesh->dim, 1);

    if (!jump_preparepointdata(iref, posn, qinterp)) return false;

    args[0]=MORPHO_OBJECT(&mposn);
    for (int i=0; i<ref->integral.nfields; i++) args[i+1]=qinterp[i];

    if (!morpho_call(iref->v, ref->integral.integrand, ref->integral.nfields+1, args, &outval)) return false;
    return morpho_valuetofloat(outval, out);
}

static bool jump_integrandfn(unsigned int dim, double *t, double *x, unsigned int nquantity, value *quantity, void *ref, double *fout) {
    objectjumpinterfaceref *iref = (objectjumpinterfaceref *) ref;
    return jump_callintegrand(iref, x, fout);
}

static bool jump_evaluatesidegradient(objectjumpinterfaceref *iref, int ifld, bool plus, double *grad) {
    objectfield *fld = MORPHO_GETFIELD(iref->jref->integral.fields[ifld]);
    jumpside *side = (plus ? &iref->qplus : &iref->qminus);
    int nv = (plus ? iref->plusnv : iref->minusnv);
    int *vid = (plus ? iref->plusvid : iref->minusvid);
    double *lambda = (plus ? iref->pluslambda : iref->minuslambda);
    int dim = iref->mesh->dim;
    grade g = iref->jref->parentgrade;

    if (!MORPHO_ISFESPACE(fld->fnspc) || !MORPHO_ISNIL(fld->prototype)) return false;

    fespace *disc = MORPHO_GETFESPACE(fld->fnspc)->fespace;
    if (!FESPACE_HASGRADIENT(disc)) return false;

    double *x[nv];
    if (!jump_getinterfacevertexpositions(iref->mesh, nv, vid, x)) return false;

    double invjdata[g*dim];
    objectmatrix invj = MORPHO_STATICMATRIX(invjdata, g, dim);
    if (!integral_prepareinvjacobian(dim, g, x, &invj)) return false;

    int nnodes = disc->nnodes;
    double gdata[nnodes*g];
    double fdata[nnodes*dim];
    objectmatrix gmat = MORPHO_STATICMATRIX(gdata, nnodes, g);
    objectmatrix fmat = MORPHO_STATICMATRIX(fdata, nnodes, dim);

    fespace_gradient(disc, lambda, &gmat);
    if (matrix_mul(&gmat, &invj, &fmat)!=LINALGERR_OK) return false;

    for (int i=0; i<dim; i++) {
        value sum=MORPHO_FLOAT(0.0);
        if (!integrator_sumquantityweighted(nnodes, fmat.elements+i*nnodes, side->quantities[ifld].vals, &sum)) return false;
        if (!morpho_valuetofloat(sum, &grad[i])) return false;
    }

    return true;
}

static bool jump_preparenormal(vm *v, objectjumpinterfaceref *iref) {
    int dim=iref->mesh->dim;
    double pluscentroid[dim], minuscentroid[dim], d[dim];

    if (!jump_getelementcentroid(iref->mesh, iref->plusnv, iref->plusvid, pluscentroid)) return false;
    if (!jump_getelementcentroid(iref->mesh, iref->minusnv, iref->minusvid, minuscentroid)) return false;

    functional_vecsub(dim, minuscentroid, pluscentroid, d);

    objectmatrix *mnormal = matrix_new(dim, 1, false);
    if (!mnormal) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);

    for (int i=0; i<dim; i++) mnormal->elements[i]=0.0;

    if (iref->g==0) {
        for (int i=0; i<dim; i++) mnormal->elements[i]=d[i];
    } else if (iref->g==1) {
        double t[dim], n[dim];

        functional_vecsub(dim, iref->vertexposn[1], iref->vertexposn[0], t);
        double tnorm=functional_vecnorm(dim, t);
        if (tnorm<MORPHO_EPS) { object_free((object *) mnormal); return false; }
        functional_vecscale(dim, 1.0/tnorm, t, t);

        double dott=functional_vecdot(dim, d, t);
        functional_vecaddscale(dim, d, -dott, t, n);

        double nnorm=functional_vecnorm(dim, n);
        if (nnorm<MORPHO_EPS) {
            if (dim==2) {
                n[0]=-t[1];
                n[1]= t[0];
            } else {
                for (int i=0; i<dim; i++) n[i]=d[i];
            }
            nnorm=functional_vecnorm(dim, n);
        }

        if (nnorm<MORPHO_EPS) { object_free((object *) mnormal); return false; }
        functional_vecscale(dim, 1.0/nnorm, n, mnormal->elements);
    } else if (iref->g==2) {
        if (dim!=3) { object_free((object *) mnormal); return false; }

        double s0[3], s1[3];
        functional_vecsub(3, iref->vertexposn[1], iref->vertexposn[0], s0);
        functional_vecsub(3, iref->vertexposn[2], iref->vertexposn[1], s1);
        functional_veccross(s0, s1, mnormal->elements);
    } else {
        object_free((object *) mnormal);
        return false;
    }

    double nnorm=functional_vecnorm(dim, mnormal->elements);
    if (nnorm<MORPHO_EPS) { object_free((object *) mnormal); return false; }

    if (functional_vecdot(dim, mnormal->elements, d)<0.0) {
        functional_vecscale(dim, -1.0, mnormal->elements, mnormal->elements);
    }

    nnorm=functional_vecnorm(dim, mnormal->elements);
    if (nnorm<MORPHO_EPS) { object_free((object *) mnormal); return false; }
    functional_vecscale(dim, 1.0/nnorm, mnormal->elements, mnormal->elements);

    iref->normal=mnormal;
    return true;
}

static bool jump_preparegeometry(vm *v, objectjumpinterfaceref *iref, double **vertexposn) {
    iref->vertexposn=vertexposn;
    if (iref->g==0) iref->interfacesize=1.0;
    else if (!functional_elementsize(v, iref->mesh, iref->g, iref->id, iref->nv, iref->vid, &iref->interfacesize)) return false;

    if (iref->g>0 && iref->interfacesize<MORPHO_EPS) return true;

    return jump_preparenormal(v, iref);
}

static bool jump_prepareinterfaceref(vm *v, objectmesh *mesh, jumpref *ref, elementid id, int nv, int *vid, double **vertexposn, int *parents, objectjumpinterfaceref *iref) {
    int plusnv=0, minusnv=0;
    int *plusvid=NULL, *minusvid=NULL;

    *iref = (objectjumpinterfaceref) MORPHO_STATICJUMPINTERFACEREF(mesh, ref->interfacegrade, id, nv, vid);
    iref->v=v;
    iref->jref=ref;

    jump_orderparents(parents, &iref->plusid, &iref->minusid);

    if (!mesh_getconnectivity(ref->parentvertices, iref->plusid, &plusnv, &plusvid)) return false;
    if (!mesh_getconnectivity(ref->parentvertices, iref->minusid, &minusnv, &minusvid)) return false;

    iref->plusnv=plusnv;
    iref->plusvid=plusvid;
    iref->minusnv=minusnv;
    iref->minusvid=minusvid;
    iref->pluslambda=MORPHO_MALLOC(sizeof(double)*iref->plusnv);
    iref->minuslambda=MORPHO_MALLOC(sizeof(double)*iref->minusnv);
    if (!iref->pluslambda || !iref->minuslambda) {
        jump_clearinterfaceref(iref);
        MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    }

    if (!jump_preparejumpside(ref, iref->plusnv, iref->plusvid, &iref->qplus)) {
        jump_clearinterfaceref(iref);
        return false;
    }
    if (!jump_preparejumpside(ref, iref->minusnv, iref->minusvid, &iref->qminus)) {
        jump_clearinterfaceref(iref);
        return false;
    }

    if (!jump_preparegeometry(v, iref, vertexposn)) {
        jump_clearinterfaceref(iref);
        return false;
    }

    return true;
}

/** Basic Jump scan over codimension-1 entities.
    This currently only identifies interior interfaces by checking that they
    have exactly two adjacent parent elements. */
static bool jump_scan_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *r, double *out) {
    jumpref *ref = (jumpref *) r;
    int nparents=0, *parents=NULL;
    double *x[nv];

    if (!jump_getadjacentparents(ref, id, &nparents, &parents)) return false;

    /* Boundary interfaces or malformed topology are ignored for now. */
    if (nparents!=2) { *out=0.0; return true; }

    if (!jump_getinterfacevertexpositions(mesh, nv, vid, x)) return false;

    objectjumpinterfaceref iref;
    if (!jump_prepareinterfaceref(v, mesh, ref, id, nv, vid, x, parents, &iref)) return false;

    if (iref.g>0 && iref.interfacesize<MORPHO_EPS) {
        *out=0.0;
        jump_clearinterfaceref(&iref);
        return true;
    }

    vm_settlvar(v, jumpinterfacehandle, MORPHO_OBJECT(&iref));

    if (ref->strategy==JUMP_STRATEGY_CENTROID_MODE || ref->interfacegrade==0) {
        double posn[mesh->dim];
        jump_centroid(mesh->dim, nv, x, posn);
        if (!jump_callintegrand(&iref, posn, out)) {
            vm_settlvar(v, jumpinterfacehandle, MORPHO_NIL);
            jump_clearinterfaceref(&iref);
            return false;
        }
        *out *= iref.interfacesize;
    } else if (ref->strategy==JUMP_STRATEGY_QUADRATURE_MODE) {
        double err=0.0;
        if (!integrate(jump_integrandfn, MORPHO_GETDICTIONARY(ref->integral.method), morpho_geterror(v), mesh->dim, ref->interfacegrade, x, 0, NULL, &iref, out, &err)) {
            vm_settlvar(v, jumpinterfacehandle, MORPHO_NIL);
            jump_clearinterfaceref(&iref);
            return false;
        }
        *out *= iref.interfacesize;
    } else {
        morpho_runtimeerror(v, JUMP_UNIMPL);
        vm_settlvar(v, jumpinterfacehandle, MORPHO_NIL);
        jump_clearinterfaceref(&iref);
        return false;
    }

    vm_settlvar(v, jumpinterfacehandle, MORPHO_NIL);
    jump_clearinterfaceref(&iref);
    return true;
}

static bool jump_mapfieldgradient(vm *v, functional_mapinfo *info, value *out) {
    jumpref *ref = (jumpref *) info->ref;
    return functional_mapjumpnumericalfieldgradient(v, info, ref->parentvertices, ref, out);
}

static value Jump_integrand(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    jumpref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (jump_prepareref(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, 0, info.sel, &ref)) {
            info.g=ref.interfacegrade;
            info.integrand=jump_scan_integrand;
            info.start=jump_startfn;
            info.ref=&ref;
            functional_runmap(v, &info, functional_mapintegrand, &out);
        }
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

static value Jump_total(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    jumpref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (jump_prepareref(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, 0, info.sel, &ref)) {
            info.g=ref.interfacegrade;
            info.integrand=jump_scan_integrand;
            info.start=jump_startfn;
            info.ref=&ref;
            functional_runmap(v, &info, functional_sumintegrand, &out);
        }
    }

    return out;
}

static value Jump_gradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    jumpref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (jump_prepareref(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, 0, info.sel, &ref)) {
            info.g=ref.interfacegrade;
            info.integrand=jump_scan_integrand;
            info.start=jump_startfn;
            info.dependencies=jump_dependencies;
            info.ref=&ref;
            functional_runmap(v, &info, functional_mapnumericalgradient, &out);
        }
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

static value Jump_fieldgradient(vm *v, int nargs, value *args) {
    functional_mapinfo info;
    jumpref ref;
    value out=MORPHO_NIL;

    if (functional_validateargs(v, nargs, args, &info)) {
        if (jump_prepareref(v, MORPHO_GETINSTANCE(MORPHO_SELF(args)), info.mesh, 0, info.sel, &ref)) {
            info.g=ref.interfacegrade;
            info.integrand=jump_scan_integrand;
            info.start=jump_startfn;
            info.dependencies=NULL;
            info.cloneref=jump_cloneref;
            info.freeref=jump_freeref;
            info.ref=&ref;
            functional_runmap(v, &info, jump_mapfieldgradient, &out);
        }
    }
    if (!MORPHO_ISNIL(out)) morpho_bindobjects(v, 1, &out);
    return out;
}

static value integral_jumpdnfn(vm *v, int nargs, value *args) {
    objectjumpinterfaceref *iref = jump_getinterfaceref(v);
    if (!iref) MORPHO_RAISEVARGS(v, INTEGRAL_SPCLFN, JUMPDN_FUNCTION);

    value q = MORPHO_GETARG(args, 0);
    int ifld, xfld=-1;

    for (ifld=0; ifld<iref->jref->integral.nfields; ifld++) {
        if (MORPHO_ISFIELD(q) && MORPHO_ISSAME(iref->jref->integral.originalfields[ifld], q)) break;
        else if (iref->qinterpolated && MORPHO_ISSAME(iref->qinterpolated[ifld], q)) {
            if (xfld>=0) MORPHO_RAISE(v, INTEGRAL_FLD);
            xfld=ifld;
        }
    }
    if (xfld>=0) ifld=xfld;

    if (ifld>=iref->jref->integral.nfields) MORPHO_RAISE(v, INTEGRAL_FLD);

    double gradplus[iref->mesh->dim], gradminus[iref->mesh->dim];
    if (!jump_evaluatesidegradient(iref, ifld, true, gradplus) ||
        !jump_evaluatesidegradient(iref, ifld, false, gradminus)) MORPHO_RAISE(v, JUMP_UNIMPL);

    double jp = functional_vecdot(iref->mesh->dim, gradplus, iref->normal->elements);
    double jm = functional_vecdot(iref->mesh->dim, gradminus, iref->normal->elements);
    return MORPHO_FLOAT(jp-jm);
}

MORPHO_BEGINCLASS(Jump)
MORPHO_METHOD(MORPHO_INITIALIZER_METHOD, Jump_init, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_INTEGRAND_METHOD, Jump_integrand, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_TOTAL_METHOD, Jump_total, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_GRADIENT_METHOD, Jump_gradient, BUILTIN_FLAGSEMPTY),
MORPHO_METHOD(FUNCTIONAL_FIELDGRADIENT_METHOD, Jump_fieldgradient, BUILTIN_FLAGSEMPTY)
MORPHO_ENDCLASS

/* **********************************************************************
 * Initialization
 * ********************************************************************** */

void functional_initialize(void) {
    fddelta1 = pow(MORPHO_EPS, 1.0/3.0);
    fddelta2 = pow(MORPHO_EPS, 1.0/4.0);
    
    functional_gradeproperty=builtin_internsymbolascstring(FUNCTIONAL_GRADE_PROPERTY);
    functional_fieldproperty=builtin_internsymbolascstring(FUNCTIONAL_FIELD_PROPERTY);
    scalarpotential_functionproperty=builtin_internsymbolascstring(SCALARPOTENTIAL_FUNCTION_PROPERTY);
    scalarpotential_gradfunctionproperty=builtin_internsymbolascstring(SCALARPOTENTIAL_GRADFUNCTION_PROPERTY);
    linearelasticity_referenceproperty=builtin_internsymbolascstring(LINEARELASTICITY_REFERENCE_PROPERTY);
    linearelasticity_weightbyreferenceproperty=builtin_internsymbolascstring(LINEARELASTICITY_WTBYREF_PROPERTY);
    linearelasticity_poissonproperty=builtin_internsymbolascstring(LINEARELASTICITY_POISSON_PROPERTY);
    hydrogel_aproperty=builtin_internsymbolascstring(HYDROGEL_A_PROPERTY);
    hydrogel_bproperty=builtin_internsymbolascstring(HYDROGEL_B_PROPERTY);
    hydrogel_cproperty=builtin_internsymbolascstring(HYDROGEL_C_PROPERTY);
    hydrogel_dproperty=builtin_internsymbolascstring(HYDROGEL_D_PROPERTY);
    hydrogel_phirefproperty=builtin_internsymbolascstring(HYDROGEL_PHIREF_PROPERTY);
    hydrogel_phi0property=builtin_internsymbolascstring(HYDROGEL_PHI0_PROPERTY);
    equielement_weightproperty=builtin_internsymbolascstring(EQUIELEMENT_WEIGHT_PROPERTY);
    nematic_ksplayproperty=builtin_internsymbolascstring(NEMATIC_KSPLAY_PROPERTY);
    nematic_ktwistproperty=builtin_internsymbolascstring(NEMATIC_KTWIST_PROPERTY);
    nematic_kbendproperty=builtin_internsymbolascstring(NEMATIC_KBEND_PROPERTY);
    nematic_pitchproperty=builtin_internsymbolascstring(NEMATIC_PITCH_PROPERTY);
    
    functional_methodproperty=builtin_internsymbolascstring(INTEGRAL_METHOD_PROPERTY);

    curvature_integrandonlyproperty=builtin_internsymbolascstring(CURVATURE_INTEGRANDONLY_PROPERTY);
    curvature_geodesicproperty=builtin_internsymbolascstring(CURVATURE_GEODESIC_PROPERTY);

    objectstring objclassname = MORPHO_STATICSTRING(OBJECT_CLASSNAME);
    value objclass = builtin_findclass(MORPHO_OBJECT(&objclassname));

    builtin_addclass(LENGTH_CLASSNAME, MORPHO_GETCLASSDEFINITION(Length), objclass);
    builtin_addclass(AREA_CLASSNAME, MORPHO_GETCLASSDEFINITION(Area), objclass);
    builtin_addclass(AREAENCLOSED_CLASSNAME, MORPHO_GETCLASSDEFINITION(AreaEnclosed), objclass);
    builtin_addclass(VOLUMEENCLOSED_CLASSNAME, MORPHO_GETCLASSDEFINITION(VolumeEnclosed), objclass);
    builtin_addclass(VOLUME_CLASSNAME, MORPHO_GETCLASSDEFINITION(Volume), objclass);
    builtin_addclass(SCALARPOTENTIAL_CLASSNAME, MORPHO_GETCLASSDEFINITION(ScalarPotential), objclass);
    builtin_addclass(LINEARELASTICITY_CLASSNAME, MORPHO_GETCLASSDEFINITION(LinearElasticity), objclass);
    builtin_addclass(HYDROGEL_CLASSNAME, MORPHO_GETCLASSDEFINITION(Hydrogel), objclass);
    builtin_addclass(EQUIELEMENT_CLASSNAME, MORPHO_GETCLASSDEFINITION(EquiElement), objclass);
    builtin_addclass(LINECURVATURESQ_CLASSNAME, MORPHO_GETCLASSDEFINITION(LineCurvatureSq), objclass);
    builtin_addclass(LINETORSIONSQ_CLASSNAME, MORPHO_GETCLASSDEFINITION(LineTorsionSq), objclass);
    builtin_addclass(MEANCURVATURESQ_CLASSNAME, MORPHO_GETCLASSDEFINITION(MeanCurvatureSq), objclass);
    builtin_addclass(GAUSSCURVATURE_CLASSNAME, MORPHO_GETCLASSDEFINITION(GaussCurvature), objclass);
    builtin_addclass(GRADSQ_CLASSNAME, MORPHO_GETCLASSDEFINITION(GradSq), objclass);
    builtin_addclass(NORMSQ_CLASSNAME, MORPHO_GETCLASSDEFINITION(NormSq), objclass);
    builtin_addclass(LINEINTEGRAL_CLASSNAME, MORPHO_GETCLASSDEFINITION(LineIntegral), objclass);
    builtin_addclass(AREAINTEGRAL_CLASSNAME, MORPHO_GETCLASSDEFINITION(AreaIntegral), objclass);
    builtin_addclass(VOLUMEINTEGRAL_CLASSNAME, MORPHO_GETCLASSDEFINITION(VolumeIntegral), objclass);
    builtin_addclass(JUMP_CLASSNAME, MORPHO_GETCLASSDEFINITION(Jump), objclass);
    builtin_addclass(NEMATIC_CLASSNAME, MORPHO_GETCLASSDEFINITION(Nematic), objclass);
    builtin_addclass(NEMATICELECTRIC_CLASSNAME, MORPHO_GETCLASSDEFINITION(NematicElectric), objclass);

    builtin_addfunction(ELEMENTID_FUNCTION, integral_elementid, MORPHO_FN_THREADLOCAL | MORPHO_FN_THROWS);
    builtin_addfunction(TANGENT_FUNCTION, integral_tangent, MORPHO_FN_THREADLOCAL | MORPHO_FN_ALLOCATES | MORPHO_FN_THROWS);
    builtin_addfunction(NORMAL_FUNCTION, integral_normal, MORPHO_FN_THREADLOCAL | MORPHO_FN_ALLOCATES | MORPHO_FN_THROWS);
    builtin_addfunction(GRAD_FUNCTION, integral_gradfn, MORPHO_FN_THREADLOCAL | MORPHO_FN_ALLOCATES | MORPHO_FN_THROWS);
    builtin_addfunction(HESS_FUNCTION, integral_hessfn, MORPHO_FN_THREADLOCAL | MORPHO_FN_ALLOCATES | MORPHO_FN_THROWS);
    builtin_addfunction(CGTENSOR_FUNCTION, integral_cgfn, MORPHO_FN_THREADLOCAL | MORPHO_FN_ALLOCATES | MORPHO_FN_THROWS);
    builtin_addfunction(JUMPDN_FUNCTION, integral_jumpdnfn, MORPHO_FN_THREADLOCAL | MORPHO_FN_ALLOCATES | MORPHO_FN_THROWS);
    builtin_addfunction(JACOBIAN_FUNCTION, integral_jacobian, MORPHO_FN_THREADLOCAL | MORPHO_FN_ALLOCATES | MORPHO_FN_THROWS);
    builtin_addfunction(INVJACOBIAN_FUNCTION, integral_invjacobian, MORPHO_FN_THREADLOCAL | MORPHO_FN_ALLOCATES | MORPHO_FN_THROWS);

    morpho_defineerror(VOLUMEENCLOSED_ZERO, ERROR_HALT, VOLUMEENCLOSED_ZERO_MSG);
    morpho_defineerror(FUNC_ELNTFND, ERROR_HALT, FUNC_ELNTFND_MSG);
    morpho_defineerror(FUNC_FESPACE, ERROR_HALT, FUNC_FESPACE_MSG);

    morpho_defineerror(SCALARPOTENTIAL_FNCLLBL, ERROR_HALT, SCALARPOTENTIAL_FNCLLBL_MSG);

    morpho_defineerror(HYDROGEL_FLDGRD, ERROR_HALT, HYDROGEL_FLDGRD_MSG);
    morpho_defineerror(HYDROGEL_ZEEROREFELEMENT, ERROR_WARNING, HYDROGEL_ZEEROREFELEMENT_MSG);
    morpho_defineerror(HYDROGEL_BNDS, ERROR_WARNING, HYDROGEL_BNDS_MSG);

    morpho_defineerror(FUNCTIONAL_ARGS, ERROR_HALT, FUNCTIONAL_ARGS_MSG);
    
    morpho_defineerror(INTEGRAL_ARGS, ERROR_HALT, INTEGRAL_ARGS_MSG);
    morpho_defineerror(INTEGRAL_FLD, ERROR_HALT, INTEGRAL_FLD_MSG);
    morpho_defineerror(INTEGRAL_SPCLFN, ERROR_HALT, INTEGRAL_SPCLFN_MSG);
    morpho_defineerror(INTEGRAL_DFFEVL, ERROR_HALT, INTEGRAL_DFFEVL_MSG);
    morpho_defineerror(JUMP_UNIMPL, ERROR_HALT, JUMP_UNIMPL_MSG);
    
    functional_poolinitialized = false;
    
    objectintegralelementreftype=object_addtype(&objectintegralelementrefdefn);
    objectjumpinterfacereftype=object_addtype(&objectjumpinterfacerefdefn);
    elementhandle=vm_addtlvar();
    jumpinterfacehandle=vm_addtlvar();
    tangenthandle=vm_addtlvar();
    normlhandle=vm_addtlvar();
    cauchygreenhandle=vm_addtlvar();
    jacobianhandle=vm_addtlvar();
    invjacobianhandle=vm_addtlvar();
    
    morpho_addfinalizefn(functional_finalize);
}

void functional_finalize(void) {
    if (functional_poolinitialized) threadpool_clear(&functional_pool);
}

#endif
