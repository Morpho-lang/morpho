/** @file functional.c
 *  @author T J Atherton
 *
 *  @brief Functionals
 */

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "functional.h"
#include "morpho.h"
#include "classes.h"
#include "common.h"
#include "optimize.h"

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
    info->fieldgrad=NULL;
    info->start=NULL;
    info->end=NULL;
    info->taskstart=NULL;
    info->taskend=NULL;
    info->dependencies=NULL;
    info->cloneref=NULL;
    info->freeref=NULL;
    info->ref=NULL;
    info->sym=SYMMETRY_NONE;
    info->cost=FUNCTIONAL_COST_REGULAR;
}

/** Fill mapinfo from typed pointers. Unused slots are NULL.
 * If field is set and mesh is NULL, the field's mesh is used. */
void _functional_mapinfo(functional_mapinfo *info,
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
value _functional_run(vm *v, functional_mapinfo *info, grade g, functional_mapcallback *mapfn, bool bind) {
    if (info->g < 0) info->g = g;
    value out=MORPHO_NIL;
    functional_runmap(v, info, mapfn, &out);
    if (!bind || !MORPHO_ISOBJECT(out) || !MORPHO_GETOBJECT(out)) return out;
    return morpho_wrapandbind(v, MORPHO_GETOBJECT(out));
}

value _functional_integrand(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn) {
    info->integrand = fn;
    return _functional_run(v, info, g, functional_mapintegrand, true);
}

value _functional_integrand_elem(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn) {
    info->integrand = fn;
    return _functional_run(v, info, g, functional_mapintegrandforelement, false);
}

value _functional_total(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn) {
    info->integrand = fn;
    return _functional_run(v, info, g, functional_sumintegrand, false);
}

value _functional_gradient(vm *v, functional_mapinfo *info, grade g, functional_gradient *fn, symmetrybhvr sym) {
    info->grad = fn;
    info->sym = sym;
    return _functional_run(v, info, g, functional_mapgradient, true);
}

value _functional_fieldgradient(vm *v, functional_mapinfo *info, grade g, functional_fieldgradient *fn) {
    info->fieldgrad = fn;
    return _functional_run(v, info, g, functional_mapfieldgradient, true);
}

value _functional_numericalgradient(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn, symmetrybhvr sym) {
    info->integrand = fn;
    info->sym = sym;
    return _functional_run(v, info, g, functional_mapnumericalgradient, true);
}

value _functional_hessian(vm *v, functional_mapinfo *info, grade g, functional_integrand *fn) {
    info->integrand = fn;
    return _functional_run(v, info, g, functional_mapnumericalhessian, true);
}

/** Validates the arguments provided to a functional.
 * Used by the FUNCTIONAL_* compatibility shim. */
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

/** Count elements of grade g. Returns false if the mesh has no such grade. */
static bool functional_countelements(objectmesh *mesh, grade g, int *n, objectsparse **s) {
    if (s) *s=NULL;
    *n=0;
    if (g==MESH_GRADE_VERTEX) {
        *n=mesh->vert->ncols;
        return true;
    }
    objectsparse *conn=mesh_getconnectivityelement(mesh, 0, g);
    if (!conn) return false;
    if (s) *s=conn;
    *n=conn->ccs.ncols;
    return true;
}

/** Count elements, raising FUNC_ELNTFND if the grade is missing. */
static bool functional_requireelements(vm *v, objectmesh *mesh, grade g, int *n, objectsparse **s) {
    if (functional_countelements(mesh, g, n, s)) return true;
    MORPHO_FAILVARGS(v, FUNC_ELNTFND, g);
}

/** Call the optional start hook once per user-facing map. Also verifies the mesh
    provides the requested grade. */
bool functional_startmap(vm *v, functional_mapinfo *info) {
    int n=0;
    objectsparse *s=NULL;
    if (info->mesh && !functional_requireelements(v, info->mesh, info->g, &n, &s)) return false;
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
    functional_mapinfo *mapinfo; /* Parent mapinfo for taskstart/taskend */
    functional_taskstart *taskstart; /* Optional per-task setup */
    functional_taskend *taskend; /* Optional per-task teardown */
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
    task->mapinfo=info;
    task->taskstart=(info ? info->taskstart : NULL);
    task->taskend=(info ? info->taskend : NULL);
    
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
    bool success=true;
    
    if (task->selection) {
        selected=&task->selection->selected[task->g];
        if (selected->count==0) return true;
    }
    
    if (task->taskstart &&
        !(*task->taskstart) (task->v, task->mapinfo)) return false;
    
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
            if (!sparseccs_getrowindices(&task->conn->ccs, task->id, &nv, &vid)) { success=false; break; }
        }
        
        // Perform the map function
        if (!(*task->mapfn) (task->v, task->mesh, task->id, nv, vid, task->ref, task->result)) { success=false; break; }
        
        // Perform post-processing if needed
        if (task->processfn) if (!(*task->processfn) (task)) { success=false; break; }
        
        // Temporary objects on worker VMs must not accumulate across elements
        if (task->usesubkernel) vm_cleansubkernel(task->v);
    }
    
    if (task->taskend) (*task->taskend) (task->v, task->mapinfo);
    return success;
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
    
    void *args[ntasks];
    for (int i=0; i<ntasks; i++) args[i]=(void *) &tasks[i];
    threadpool_add_tasks(&functional_pool, ntasks, functional_mapfn_elements, args);
    return threadpool_fence(&functional_pool);
}

/** Map over prepared tasks, using a threadpool only when worker threads are available */
bool functional_map(int ntasks, functional_task *tasks) {
    if (ntasks<1) return true;
    if (ntasks==1 || morpho_threadnumber()<1) return functional_serialmap(ntasks, tasks);
    return functional_parallelmap(ntasks, tasks);
}

/** Determine the number of tasks to use based on the number of elements and the cost class. */
static int functional_ntasks(functional_mapinfo *info) {
    int n=morpho_threadnumber();
    if (n<1) return 1;
    
    int nwork=0;
    if (!info || !info->mesh || info->g<0 ||
        !functional_countelements(info->mesh, info->g, &nwork, NULL)) return 1;
    if (info->sel) nwork=selection_count(info->sel, info->g);
    if (nwork<=1) return 1; /* Mapper does not split a single element */
    if (info->cost>=FUNCTIONAL_COST_REGULAR) return n;
    
    if ((int64_t) nwork * info->cost <
        (int64_t) FUNCTIONAL_FORKWEIGHT * (n-1)) return 1;
    return n;
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
    if (!functional_requireelements(v, info->mesh, info->g, &nel, &conn)) {
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
    int ntask=functional_ntasks(info);
    
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
    if (!functional_requireelements(v, mesh, g, &n, &s)) return false;
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
    int ntask=functional_ntasks(info);
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
 * Shared gradient combiner
 * ---------------------------- */

/** Add inc into *p. Safe when several workers share an output. */
static void functional_accum(double *p, double inc) {
    if (inc==0.0) return;
    MorphoAtomic_adddouble(p, inc);
}

/** Add alpha*b into column col. */
static bool functional_addtocolumn(objectmatrix *a, MatrixIdx_t col, double alpha, double *b) {
    if (!a || col<0 || col>=a->ncols) return false;
    double *dest=a->elements+a->nvals*col*a->nrows;
    int n=a->nrows*a->nvals;
    for (int i=0; i<n; i++) MorphoAtomic_madddouble(&dest[i], alpha, b[i]);
    return true;
}

/** Add inc into element (row, col). */
static bool functional_addtoelement(objectmatrix *a, MatrixIdx_t row, MatrixIdx_t col, double inc) {
    double *p=NULL;
    if (matrix_getelementptr(a, row, col, &p)!=LINALGERR_OK) return false;
    functional_accum(p, inc);
    return true;
}

/* ----------------------------
 * Map gradients
 * ---------------------------- */

/** Compute the gradient */
bool functional_mapgradient(vm *v, functional_mapinfo *info, value *out) {
    int success=false;
    int ntask=functional_ntasks(info);
    functional_task task[ntask];
    
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    objectmatrix *new=NULL;
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    
    new=matrix_new(info->mesh->vert->nrows, info->mesh->vert->ncols, true);
    if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_mapgradient_cleanup; }
    
    for (int i=0; i<ntask; i++) {
        task[i].mapfn=(functional_mapfn *) info->grad;
        task[i].result=(void *) new;
    }
    
    if (!functional_map(ntask, task)) goto functional_mapgradient_cleanup;
    
    if (info->sym==SYMMETRY_ADD) functional_symmetrysumforces(info->mesh, new);
    
    success=true;
    *out = MORPHO_OBJECT(new);
    
functional_mapgradient_cleanup:
    if (!success && new) object_free((object *) new);
    
    functional_cleanuptasks(v, ntask, task, &imageids);
    return success;
}

/* ----------------------------
 * Map analytic field gradients
 * ---------------------------- */

/** Compute an analytic field gradient. A NULL fieldgrad yields a zero Field. */
bool functional_mapfieldgradient(vm *v, functional_mapinfo *info, value *out) {
    int success=false;
    objectfield *new=NULL;
    
    if (!info->field) MORPHO_FAIL(v, FUNCTIONAL_ARGS);
    
    new=object_newfield(info->mesh, info->field->prototype, info->field->fnspc, info->field->dof);
    if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return false; }
    field_zero(new);
    
    if (!info->fieldgrad) {
        *out = MORPHO_OBJECT(new);
        return true;
    }
    
    int ntask=functional_ntasks(info);
    functional_task task[ntask];
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) {
        object_free((object *) new);
        return false;
    }
    
    for (int i=0; i<ntask; i++) {
        task[i].mapfn=(functional_mapfn *) info->fieldgrad;
        task[i].result=(void *) new;
    }
    
    if (!functional_map(ntask, task)) goto functional_mapfieldgradient_analytic_cleanup;
    
    success=true;
    *out = MORPHO_OBJECT(new);
    
functional_mapfieldgradient_analytic_cleanup:
    if (!success && new) object_free((object *) new);
    
    functional_cleanuptasks(v, ntask, task, &imageids);
    return success;
}

/* ----------------------------
 * Map numerical gradients
 * ---------------------------- */

/** Computes the gradient of element eid with respect to vertex i */
bool functional_numericalgrad(vm *v, objectmesh *mesh, elementid eid, elementid i, int nv, int *vid, functional_integrand *integrand, void *ref, objectmatrix *frc) {
    double fp,fm,x0,eps=1e-6;
    
    // Loop over coordinates
    for (unsigned int k=0; k<mesh->dim; k++) {
        matrix_getelement(mesh->vert, k, i, &x0);
        
        eps=functional_fdstepsize(x0, 1);
        matrix_setelement(mesh->vert, k, i, x0+eps);
        if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fp)) return false;
        matrix_setelement(mesh->vert, k, i, x0-eps);
        if (!(*integrand) (v, mesh, eid, nv, vid, ref, &fm)) return false;
        matrix_setelement(mesh->vert, k, i, x0);

        if (!functional_addtoelement(frc, k, i, (fp-fm)/(2*eps))) return false;
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
    int ntask=functional_ntasks(info);
    functional_task task[ntask];
    
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    objectmatrix *new=NULL;
    objectmesh meshclones[ntask]; // Shallow clones with private vertex matrices (parallel only)
    for (int i=0; i<ntask; i++) meshclones[i].vert=NULL;
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    
    new=matrix_new(info->mesh->vert->nrows, info->mesh->vert->ncols, true);
    if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_mapnumericalgradient_cleanup; }
    
    for (int i=0; i<ntask; i++) {
        // Serial maps perturb the original vertices in place; clone only for workers
        if (ntask>1) {
            objectmatrix *vert=matrix_clone(info->mesh->vert);
            if (!vert) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_mapnumericalgradient_cleanup; }
            meshclones[i]=*info->mesh;
            meshclones[i].vert=vert;
            task[i].mesh=&meshclones[i];
        }
        
        task[i].ref=(void *) info;
        task[i].mapfn=functional_numericalgradientmapfn;
        task[i].result=(void *) new;
    }
    
    if (!functional_map(ntask, task)) goto functional_mapnumericalgradient_cleanup;
    
    if (info->sym==SYMMETRY_ADD) functional_symmetrysumforces(info->mesh, new);
    
    success=true;
    *out = MORPHO_OBJECT(new);
    
functional_mapnumericalgradient_cleanup:
    for (int i=0; i<ntask; i++) if (meshclones[i].vert) object_free((object *) meshclones[i].vert);
    if (!success && new) object_free((object *) new);
    
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

        functional_accum(&grad->data.elements[k], (fr-fl)/(2*eps));
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
        functional_accum(&gentry[j], (fr-fl)/(2*eps));
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

/** Fill a numerical fieldgradient task ref. Clones the Field when clone is true. */
static bool functional_preparenumericalfieldgradientref(vm *v, functional_mapinfo *info, bool clone, functional_numericalfieldgradientref *tref, objectfield **fieldclone) {
    tref->info=info;
    tref->integrand=info->integrand;
    tref->conn=mesh_getconnectivityelement(info->mesh, 0, info->g);
    tref->disc=NULL;
    tref->ref=info->ref;
    *fieldclone=NULL;

    if (clone) {
        *fieldclone=field_clone(info->field);
        if (!*fieldclone) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); return false; }
        tref->field=*fieldclone;
        if (!info->cloneref) UNREACHABLE("Functional calls numericalfieldgradient but doesn't provide cloneref");
        tref->ref=(info->cloneref)(info->ref, info->field, *fieldclone);
        if (!tref->ref) {
            object_free((object *) *fieldclone);
            *fieldclone=NULL;
            morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
            return false;
        }
    } else tref->field=info->field;

    if (MORPHO_ISFESPACE(tref->field->fnspc)) {
        tref->disc=MORPHO_GETFESPACE(tref->field->fnspc)->fespace;
        if (info->g==0 && tref->disc->shape[0]>0) tref->disc=NULL;
        else if (info->g<tref->disc->grade) {
            if (!fespace_lower(tref->disc, info->g, &tref->disc)) {
                functional_fespaceerror(v, tref->field, info->g);
                return false;
            }
        }
    }
    return true;
}

static void functional_clearnumericalfieldgradientref(functional_mapinfo *info, functional_numericalfieldgradientref *tref, objectfield *fieldclone) {
    if (!fieldclone) return;
    if (tref->ref) {
        if (info->freeref) (info->freeref)(tref->ref);
        else if (info->cloneref) MORPHO_FREE(tref->ref);
    }
    object_free((object *) fieldclone);
}

/** Compute the field gradient numerically */
bool functional_mapnumericalfieldgradient(vm *v, functional_mapinfo *info, value *out) {
    int success=false;
    int ntask=functional_ntasks(info);
    functional_task task[ntask];
    
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    objectfield *new=NULL;
    objectfield *fieldclones[ntask];
    functional_numericalfieldgradientref tref[ntask];
    for (int i=0; i<ntask; i++) {
        fieldclones[i]=NULL;
        tref[i].ref=NULL;
    }
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    
    new=object_newfield(info->mesh, info->field->prototype, info->field->fnspc, info->field->dof);
    if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_mapfieldgradient_cleanup; }
    field_zero(new);
    
    for (int i=0; i<ntask; i++) {
        if (!functional_preparenumericalfieldgradientref(v, info, ntask>1, &tref[i], &fieldclones[i])) goto functional_mapfieldgradient_cleanup;
        task[i].ref=(void *) &tref[i];
        task[i].mapfn=functional_numericalfieldgradientmapfn;
        task[i].result=(void *) new;
    }
    
    if (!functional_map(ntask, task)) goto functional_mapfieldgradient_cleanup;
    
    success=true;
    *out = MORPHO_OBJECT(new);
    
functional_mapfieldgradient_cleanup:
    for (int i=0; i<ntask; i++) {
        functional_clearnumericalfieldgradientref(info, &tref[i], fieldclones[i]);
    }
    
    if (!success && new) object_free((object *) new);
    
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
    int ntask=functional_ntasks(info);
    functional_task task[ntask];

    varray_elementid imageids;
    varray_elementidinit(&imageids);

    objectfield *new=NULL;
    objectfield *fieldclones[ntask];
    jump_numericalfieldgradientref tref[ntask];
    for (int i=0; i<ntask; i++) {
        fieldclones[i]=NULL;
        tref[i].ref=NULL;
    }

    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;

    new=object_newfield(info->mesh, info->field->prototype, info->field->fnspc, info->field->dof);
    if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto functional_mapjumpfieldgradient_cleanup; }
    field_zero(new);

    for (int i=0; i<ntask; i++) {
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
        task[i].result=(void *) new;
    }

    if (!functional_map(ntask, task)) goto functional_mapjumpfieldgradient_cleanup;

    success=true;
    *out=MORPHO_OBJECT(new);

functional_mapjumpfieldgradient_cleanup:
    for (int i=0; i<ntask; i++) {
        if (fieldclones[i]) {
            if (info->freeref && tref[i].ref) (info->freeref)(tref[i].ref);
            else if (info->cloneref && tref[i].ref) MORPHO_FREE(tref[i].ref);
            object_free((object *) fieldclones[i]);
        }
    }
    if (!success && new) object_free((object *) new);
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
    int ntask=functional_ntasks(info);
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
    
    if (!functional_addtocolumn(frc, vid[0], -1.0/norm*scale, s0)) return false;
    if (!functional_addtocolumn(frc, vid[1], 1./norm*scale, s0)) return false;

    return true;
}

/** Calculate gradient */
bool length_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    return length_gradient_scale(v, mesh, id, nv, vid, NULL, frc, 1.0);
}

FUNCTIONAL_INIT(Length, MESH_GRADE_LINE)
FUNCTIONAL_MD_INTEGRAND_COST(Length, MESH_GRADE_LINE, length_integrand, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_TOTAL_COST(Length, MESH_GRADE_LINE, length_integrand, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_GRADIENT_COST(Length, MESH_GRADE_LINE, length_gradient, SYMMETRY_ADD, FUNCTIONAL_COST_CHEAPEST)
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

/** Analytic gradient of 1/2 |x0 × x1|. */
bool areaenclosed_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    if (nv!=2) return true;
    double *x[2], x0[3]={0,0,0}, x1[3]={0,0,0}, c[3], f0[3], f1[3], nrm;
    for (int j=0; j<nv; j++) if (matrix_getcolumnptr(mesh->vert, vid[j], &x[j])!=LINALGERR_OK) return false;
    for (int k=0; k<mesh->dim; k++) { x0[k]=x[0][k]; x1[k]=x[1][k]; }

    functional_veccross(x0, x1, c);
    nrm=functional_vecnorm(3, c);
    if (nrm<MORPHO_EPS) return true;

    functional_veccross(x1, c, f0); /* ∂/∂x0 ~ x1 × c */
    functional_veccross(c, x0, f1); /* ∂/∂x1 ~ c × x0 */

    if (!functional_addtocolumn(frc, vid[0], 0.5/nrm, f0)) return false;
    if (!functional_addtocolumn(frc, vid[1], 0.5/nrm, f1)) return false;

    return true;
}

FUNCTIONAL_INIT(AreaEnclosed, MESH_GRADE_LINE)
FUNCTIONAL_MD_INTEGRAND_COST(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_integrand, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_TOTAL_COST(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_integrand, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_GRADIENT_COST(AreaEnclosed, MESH_GRADE_LINE, areaenclosed_gradient, SYMMETRY_ADD, FUNCTIONAL_COST_CHEAPEST)
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

    if (!functional_addtocolumn(frc, vid[0], 0.5/norm*scale, s011)) return false;
    if (!functional_addtocolumn(frc, vid[2], 0.5/norm*scale, s010)) return false;

    functional_vecadd(mesh->dim, s010, s011, s0);

    if (!functional_addtocolumn(frc, vid[1], -0.5/norm*scale, s0)) return false;

    return true;
}

/** Calculate gradient */
bool area_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    return area_gradient_scale(v, mesh, id, nv, vid, NULL, frc, 1.0);
}

FUNCTIONAL_INIT(Area, MESH_GRADE_AREA)
FUNCTIONAL_MD_INTEGRAND_COST(Area, MESH_GRADE_AREA, area_integrand, FUNCTIONAL_COST_CHEAP)
FUNCTIONAL_MD_TOTAL_COST(Area, MESH_GRADE_AREA, area_integrand, FUNCTIONAL_COST_CHEAP)
FUNCTIONAL_MD_GRADIENT_COST(Area, MESH_GRADE_AREA, area_gradient, SYMMETRY_ADD, FUNCTIONAL_COST_CHEAP)
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

    if (!functional_addtocolumn(frc, vid[2], dot/6.0, cx)) return false;

    functional_veccross(x[1], x[2], cx);
    if (!functional_addtocolumn(frc, vid[0], dot/6.0, cx)) return false;

    functional_veccross(x[2], x[0], cx);
    if (!functional_addtocolumn(frc, vid[1], dot/6.0, cx)) return false;

    return true;
}

FUNCTIONAL_INIT(VolumeEnclosed, MESH_GRADE_AREA)
FUNCTIONAL_MD_INTEGRAND_COST(VolumeEnclosed, MESH_GRADE_AREA, volumeenclosed_integrand, FUNCTIONAL_COST_CHEAP)
FUNCTIONAL_MD_TOTAL_COST(VolumeEnclosed, MESH_GRADE_AREA, volumeenclosed_integrand, FUNCTIONAL_COST_CHEAP)
FUNCTIONAL_MD_GRADIENT_COST(VolumeEnclosed, MESH_GRADE_AREA, volumeenclosed_gradient, SYMMETRY_ADD, FUNCTIONAL_COST_CHEAP)
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

    if (!functional_addtocolumn(frc, vid[1], uu/6.0*scale, cx)) return false;

    functional_veccross(s31, s21, cx);
    if (!functional_addtocolumn(frc, vid[0], uu/6.0*scale, cx)) return false;

    functional_veccross(s30, s10, cx);
    if (!functional_addtocolumn(frc, vid[2], uu/6.0*scale, cx)) return false;

    functional_veccross(s10, s20, cx);
    if (!functional_addtocolumn(frc, vid[3], uu/6.0*scale, cx)) return false;

    return true;
}

/** Calculate gradient */
bool volume_gradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    return volume_gradient_scale(v, mesh, id, nv, vid, NULL, frc, 1.0);
}

FUNCTIONAL_INIT(Volume, MESH_GRADE_VOLUME)
FUNCTIONAL_MD_INTEGRAND_COST(Volume, MESH_GRADE_VOLUME, volume_integrand, FUNCTIONAL_COST_CHEAP)
FUNCTIONAL_MD_TOTAL_COST(Volume, MESH_GRADE_VOLUME, volume_integrand, FUNCTIONAL_COST_CHEAP)
FUNCTIONAL_MD_GRADIENT_COST(Volume, MESH_GRADE_VOLUME, volume_gradient, SYMMETRY_ADD, FUNCTIONAL_COST_CHEAP)
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
                return functional_addtocolumn(frc, id, 1.0, vf->elements);
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

FUNCTIONAL_MD_REF_BIND(ScalarPotential, scalarpotentialref, scalarpotential_prepareref, scalarpotential_integrand, SCALARPOTENTIAL_FNCLLBL)
FUNCTIONAL_MD_REF_INTEGRAND(ScalarPotential, scalarpotentialref, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_TOTAL(ScalarPotential, scalarpotentialref, MESH_GRADE_VERTEX)

static value _ScalarPotential_gradient(vm *v, objectinstance *self, functional_mapinfo *info) {
    scalarpotentialref ref;
    value fn;

    if (objectinstance_getpropertyinterned(self, scalarpotential_gradfunctionproperty, &fn)) {
        if (!MORPHO_ISCALLABLE(fn)) MORPHO_RAISE(v, SCALARPOTENTIAL_FNCLLBL);
        ref.fn = fn;
        info->ref = &ref;
        info->grad = scalarpotential_gradient;
        return _functional_run(v, info, MESH_GRADE_VERTEX, functional_mapgradient, true);
    }

    if (!_ScalarPotential_bindref(v, self, info, &ref)) return MORPHO_NIL;
    return _functional_run(v, info, MESH_GRADE_VERTEX, functional_mapnumericalgradient, true);
}

FUNCTIONAL_MD_REF_OVERLOADS(ScalarPotential, gradient, _ScalarPotential_gradient)
FUNCTIONAL_MD_REF_HESSIAN(ScalarPotential, scalarpotentialref, MESH_GRADE_VERTEX, NULL, SYMMETRY_NONE)

#define SP_MAPFLAGS (MORPHO_FN_REENTRANT|FUNCTIONAL_MD_MAPFLAGS)
#define SP_TOTALFLAGS (MORPHO_FN_REENTRANT|FUNCTIONAL_MD_TOTALFLAGS)
#define SP_ELEMFLAGS (MORPHO_FN_REENTRANT|FUNCTIONAL_MD_ELEMFLAGS)

MORPHO_BEGINCLASS(ScalarPotential)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", ScalarPotential_init, MORPHO_FN_MUTATES),
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Callable)", ScalarPotential_init__fn, MORPHO_FN_MUTATES),
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Callable, Callable)", ScalarPotential_init__fn_fn, MORPHO_FN_MUTATES),

FUNCTIONAL_MD_INTEGRAND_METHODS_FLAGS(ScalarPotential, SP_MAPFLAGS, SP_ELEMFLAGS),
FUNCTIONAL_MD_TOTAL_METHODS_FLAGS(ScalarPotential, SP_TOTALFLAGS),
FUNCTIONAL_MD_GRADIENT_METHODS_FLAGS(ScalarPotential, SP_MAPFLAGS),
FUNCTIONAL_MD_HESSIAN_METHODS_FLAGS(ScalarPotential, SP_MAPFLAGS)
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

static void _linearelasticity_initmesh(objectinstance *self, value meshval) {
    objectinstance_setproperty(self, linearelasticity_referenceproperty, meshval);
    functional_setgrade(self, mesh_maxgrade(MORPHO_GETMESH(meshval)));
    objectinstance_setproperty(self, linearelasticity_poissonproperty, MORPHO_FLOAT(0.3));
}

value LinearElasticity_init__mesh(vm *v, int nargs, value *args) {
    _linearelasticity_initmesh(MORPHO_GETINSTANCE(MORPHO_SELF(args)), MORPHO_GETARG(args, 0));
    return MORPHO_NIL;
}

value LinearElasticity_init__mesh_int(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    _linearelasticity_initmesh(self, MORPHO_GETARG(args, 0));
    functional_setgrade(self, MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1)));
    return MORPHO_NIL;
}

static bool _LinearElasticity_bindref(vm *v, objectinstance *self, functional_mapinfo *info, linearelasticityref *ref) {
    if (!linearelasticity_prepareref(self, ref)) MORPHO_FAIL(v, FUNCTIONAL_ARGS);
    if (info->g < 0) info->g = ref->grade;
    info->ref = ref;
    info->integrand = linearelasticity_integrand;
    return true;
}

FUNCTIONAL_MD_REF_INTEGRAND(LinearElasticity, linearelasticityref, ref.grade)
FUNCTIONAL_MD_REF_TOTAL(LinearElasticity, linearelasticityref, ref.grade)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(LinearElasticity, linearelasticityref, ref.grade, NULL, SYMMETRY_ADD)

MORPHO_BEGINCLASS(LinearElasticity)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Mesh)", LinearElasticity_init__mesh, MORPHO_FN_MUTATES),
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Mesh, Int)", LinearElasticity_init__mesh_int, MORPHO_FN_MUTATES),

FUNCTIONAL_MD_INTEGRAND_METHODS(LinearElasticity),
FUNCTIONAL_MD_TOTAL_METHODS(LinearElasticity),
FUNCTIONAL_MD_GRADIENT_METHODS(LinearElasticity)
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

value Hydrogel_init__mesh(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    objectmesh *mesh = MORPHO_GETMESH(MORPHO_GETARG(args, 0));
    value grade=MORPHO_INTEGER(-1);
    value a=MORPHO_NIL, b=MORPHO_NIL, c=MORPHO_NIL, d=MORPHO_NIL, phiref=MORPHO_NIL, phi0=MORPHO_NIL;

    builtin_options(v, nargs, args, NULL, 7,
                    hydrogel_aproperty, &a,
                    hydrogel_bproperty, &b,
                    hydrogel_cproperty, &c,
                    hydrogel_dproperty, &d,
                    hydrogel_phirefproperty, &phiref,
                    hydrogel_phi0property, &phi0,
                    functional_gradeproperty, &grade);

    objectinstance_setproperty(self, hydrogel_aproperty, a);
    objectinstance_setproperty(self, hydrogel_bproperty, b);
    objectinstance_setproperty(self, hydrogel_cproperty, c);
    objectinstance_setproperty(self, hydrogel_dproperty, d);
    objectinstance_setproperty(self, hydrogel_phirefproperty, phiref);
    objectinstance_setproperty(self, hydrogel_phi0property, phi0);
    objectinstance_setproperty(self, linearelasticity_referenceproperty, MORPHO_GETARG(args, 0));

    if (MORPHO_ISINTEGER(grade) && MORPHO_GETINTEGERVALUE(grade)>=0) {
        functional_setgrade(self, MORPHO_GETINTEGERVALUE(grade));
    } else {
        functional_setgrade(self, mesh_maxgrade(mesh));
    }
    return MORPHO_NIL;
}

FUNCTIONAL_MD_REF_BIND(Hydrogel, hydrogelref, hydrogel_prepareref, hydrogel_integrand, FUNCTIONAL_ARGS)
FUNCTIONAL_MD_REF_INTEGRAND(Hydrogel, hydrogelref, ref.grade)
FUNCTIONAL_MD_REF_TOTAL(Hydrogel, hydrogelref, ref.grade)
FUNCTIONAL_MD_REF_GRADIENT(Hydrogel, hydrogelref, ref.grade, hydrogel_gradient, SYMMETRY_ADD)

MORPHO_BEGINCLASS(Hydrogel)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Mesh)", Hydrogel_init__mesh, MORPHO_FN_MUTATES|MORPHO_FN_OPTARGS),

FUNCTIONAL_MD_INTEGRAND_METHODS(Hydrogel),
FUNCTIONAL_MD_TOTAL_METHODS(Hydrogel),
FUNCTIONAL_MD_GRADIENT_METHODS(Hydrogel)
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
    value grade=MORPHO_INTEGER(-1);
    value weight=MORPHO_NIL;

    builtin_options(v, nargs, args, NULL, 2,
                    equielement_weightproperty, &weight,
                    functional_gradeproperty, &grade);

    objectinstance_setproperty(self, equielement_weightproperty, weight);
    objectinstance_setproperty(self, functional_gradeproperty, grade);
    return MORPHO_NIL;
}

FUNCTIONAL_MD_REF_BIND_FORCEGRADE(EquiElement, equielementref, equielement_prepareref, equielement_integrand, FUNCTIONAL_ARGS, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_INTEGRAND(EquiElement, equielementref, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_TOTAL(EquiElement, equielementref, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(EquiElement, equielementref, MESH_GRADE_VERTEX, equielement_dependencies, SYMMETRY_ADD)
FUNCTIONAL_MD_REF_HESSIAN(EquiElement, equielementref, MESH_GRADE_VERTEX, equielement_dependencies, SYMMETRY_ADD)

MORPHO_BEGINCLASS(EquiElement)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", EquiElement_init, MORPHO_FN_MUTATES|MORPHO_FN_OPTARGS),

FUNCTIONAL_MD_INTEGRAND_METHODS(EquiElement),
FUNCTIONAL_MD_TOTAL_METHODS(EquiElement),
FUNCTIONAL_MD_GRADIENT_METHODS(EquiElement),
FUNCTIONAL_MD_HESSIAN_METHODS(EquiElement)
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
FUNCTIONAL_MD_REF_BIND_FORCEGRADE(LineCurvatureSq, curvatureref, curvature_prepareref, linecurvsq_integrand, FUNCTIONAL_ARGS, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_INTEGRAND(LineCurvatureSq, curvatureref, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_TOTAL(LineCurvatureSq, curvatureref, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(LineCurvatureSq, curvatureref, MESH_GRADE_VERTEX, linecurvsq_dependencies, SYMMETRY_ADD)
FUNCTIONAL_MD_REF_HESSIAN(LineCurvatureSq, curvatureref, MESH_GRADE_VERTEX, linecurvsq_dependencies, SYMMETRY_ADD)

MORPHO_BEGINCLASS(LineCurvatureSq)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", LineCurvatureSq_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(LineCurvatureSq),
FUNCTIONAL_MD_TOTAL_METHODS(LineCurvatureSq),
FUNCTIONAL_MD_GRADIENT_METHODS(LineCurvatureSq),
FUNCTIONAL_MD_HESSIAN_METHODS(LineCurvatureSq)
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
FUNCTIONAL_MD_REF_BIND_FORCEGRADE(LineTorsionSq, curvatureref, curvature_prepareref, linetorsionsq_integrand, FUNCTIONAL_ARGS, MESH_GRADE_LINE)
FUNCTIONAL_MD_REF_INTEGRAND(LineTorsionSq, curvatureref, MESH_GRADE_LINE)
FUNCTIONAL_MD_REF_TOTAL(LineTorsionSq, curvatureref, MESH_GRADE_LINE)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(LineTorsionSq, curvatureref, MESH_GRADE_LINE, linetorsionsq_dependencies, SYMMETRY_ADD)
FUNCTIONAL_MD_REF_HESSIAN(LineTorsionSq, curvatureref, MESH_GRADE_LINE, linetorsionsq_dependencies, SYMMETRY_ADD)

MORPHO_BEGINCLASS(LineTorsionSq)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", LineTorsionSq_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(LineTorsionSq),
FUNCTIONAL_MD_TOTAL_METHODS(LineTorsionSq),
FUNCTIONAL_MD_GRADIENT_METHODS(LineTorsionSq),
FUNCTIONAL_MD_HESSIAN_METHODS(LineTorsionSq)
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
FUNCTIONAL_MD_REF_BIND_FORCEGRADE(MeanCurvatureSq, areacurvatureref, areacurvature_prepareref, meancurvaturesq_integrand, FUNCTIONAL_ARGS, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_INTEGRAND(MeanCurvatureSq, areacurvatureref, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_TOTAL(MeanCurvatureSq, areacurvatureref, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(MeanCurvatureSq, areacurvatureref, MESH_GRADE_VERTEX, meancurvaturesq_dependencies, SYMMETRY_ADD)

MORPHO_BEGINCLASS(MeanCurvatureSq)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", MeanCurvatureSq_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(MeanCurvatureSq),
FUNCTIONAL_MD_TOTAL_METHODS(MeanCurvatureSq),
FUNCTIONAL_MD_GRADIENT_METHODS(MeanCurvatureSq)
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
FUNCTIONAL_MD_REF_BIND_FORCEGRADE(GaussCurvature, areacurvatureref, areacurvature_prepareref, gausscurvature_integrand, FUNCTIONAL_ARGS, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_INTEGRAND(GaussCurvature, areacurvatureref, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_TOTAL(GaussCurvature, areacurvatureref, MESH_GRADE_VERTEX)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(GaussCurvature, areacurvatureref, MESH_GRADE_VERTEX, meancurvaturesq_dependencies, SYMMETRY_ADD)

MORPHO_BEGINCLASS(GaussCurvature)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "()", GaussCurvature_init, MORPHO_FN_MUTATES),
FUNCTIONAL_MD_INTEGRAND_METHODS(GaussCurvature),
FUNCTIONAL_MD_TOTAL_METHODS(GaussCurvature),
FUNCTIONAL_MD_GRADIENT_METHODS(GaussCurvature)
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

#define FUNCTIONAL_FESPACE_FAIL(v, field, g) \
    { functional_fespaceerror(v, field, g); return false; }

static bool functional_preparefespacefield(vm *v, objectfield *field, grade g) {
    if (!field || !MORPHO_ISFESPACE(field->fnspc)) return true;

    fespace *disc = MORPHO_GETFESPACE(field->fnspc)->fespace;
    /* Refuse a silent raise (AreaIntegral of a line-grade space). */
    if (g>disc->grade) FUNCTIONAL_FESPACE_FAIL(v, field, g);
    /* Restrict to a lower grade, unless the field already has vertex dofs (CG1).
       CG0 has no trace, so Jump/NormSq on a lower grade is FnctlFESpc. */
    if (g<disc->grade && !(g==0 && disc->shape[0]>0)) {
        if (!fespace_lower(disc, g, &disc)) FUNCTIONAL_FESPACE_FAIL(v, field, g);
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
        if (!MORPHO_ISFIELD(fields[i])) continue;

        objectfield *field = MORPHO_GETFIELD(fields[i]);
        if (!MORPHO_ISFESPACE(field->fnspc)) MORPHO_FAIL(v, FUNC_NOFESPACE);
        if (!functional_preparefespacefield(v, field, g)) return false;
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

/* TODO: Support other FEspaces. Today this is vertex P1 (CG1), same kernel as
 * Nematic / NematicElectric; NormSq is vertex |q|^2. High-order |grad q|^2 is
 * Integral + grad(). */

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

static void _gradsq_initfield(objectinstance *self, value fieldval) {
    objectinstance_setproperty(self, functional_fieldproperty, fieldval);
    functional_setgrade(self, mesh_maxgrade(MORPHO_GETFIELD(fieldval)->mesh));
}

value GradSq_init__field(vm *v, int nargs, value *args) {
    _gradsq_initfield(MORPHO_GETINSTANCE(MORPHO_SELF(args)), MORPHO_GETARG(args, 0));
    return MORPHO_NIL;
}

value GradSq_init__field_int(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    _gradsq_initfield(self, MORPHO_GETARG(args, 0));
    functional_setgrade(self, MORPHO_GETINTEGERVALUE(MORPHO_GETARG(args, 1)));
    return MORPHO_NIL;
}

FUNCTIONAL_MD_REF_BIND_START(GradSq, fieldref, gradsq_prepareref, gradsq_integrand, FUNCTIONAL_ARGS, fieldref_startfn)
FUNCTIONAL_MD_REF_INTEGRAND(GradSq, fieldref, ref.grade)
FUNCTIONAL_MD_REF_TOTAL(GradSq, fieldref, ref.grade)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(GradSq, fieldref, ref.grade, NULL, SYMMETRY_ADD)
FUNCTIONAL_MD_REF_FIELDGRADIENT(GradSq, fieldref, ref.grade, gradsq_cloneref, NULL)

MORPHO_BEGINCLASS(GradSq)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Field)", GradSq_init__field, MORPHO_FN_MUTATES),
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Field, Int)", GradSq_init__field_int, MORPHO_FN_MUTATES),

FUNCTIONAL_MD_INTEGRAND_METHODS(GradSq),
FUNCTIONAL_MD_TOTAL_METHODS(GradSq),
FUNCTIONAL_MD_GRADIENT_METHODS(GradSq),
FUNCTIONAL_MD_FIELDGRADIENT_METHODS(GradSq)
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

value Nematic_init__field(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    value ksplay=MORPHO_FLOAT(1.0),
          ktwist=MORPHO_FLOAT(1.0),
          kbend=MORPHO_FLOAT(1.0);
    value pitch=MORPHO_NIL;

    builtin_options(v, nargs, args, NULL, 4,
                    nematic_ksplayproperty, &ksplay,
                    nematic_ktwistproperty, &ktwist,
                    nematic_kbendproperty, &kbend,
                    nematic_pitchproperty, &pitch);

    objectinstance_setproperty(self, nematic_ksplayproperty, ksplay);
    objectinstance_setproperty(self, nematic_ktwistproperty, ktwist);
    objectinstance_setproperty(self, nematic_kbendproperty, kbend);
    objectinstance_setproperty(self, nematic_pitchproperty, pitch);
    _gradsq_initfield(self, MORPHO_GETARG(args, 0));
    return MORPHO_NIL;
}

FUNCTIONAL_MD_REF_BIND_START(Nematic, nematicref, nematic_prepareref, nematic_integrand, FUNCTIONAL_ARGS, nematic_startfn)
FUNCTIONAL_MD_REF_INTEGRAND(Nematic, nematicref, ref.grade)
FUNCTIONAL_MD_REF_TOTAL(Nematic, nematicref, ref.grade)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(Nematic, nematicref, ref.grade, NULL, SYMMETRY_NONE)
FUNCTIONAL_MD_REF_FIELDGRADIENT(Nematic, nematicref, ref.grade, nematic_cloneref, NULL)

MORPHO_BEGINCLASS(Nematic)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Field)", Nematic_init__field, MORPHO_FN_MUTATES|MORPHO_FN_OPTARGS),

FUNCTIONAL_MD_INTEGRAND_METHODS(Nematic),
FUNCTIONAL_MD_TOTAL_METHODS(Nematic),
FUNCTIONAL_MD_GRADIENT_METHODS(Nematic),
FUNCTIONAL_MD_FIELDGRADIENT_METHODS(Nematic)
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

value NematicElectric_init__field_field(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    objectlist *new = object_newlist(2, &MORPHO_GETARG(args, 0));

    if (new) {
        objectinstance_setproperty(self, functional_fieldproperty, MORPHO_OBJECT(new));
        functional_setgrade(self, mesh_maxgrade(MORPHO_GETFIELD(MORPHO_GETARG(args, 0))->mesh));
    }

    return morpho_wrapandbind(v, (object *) new);
}

FUNCTIONAL_MD_REF_BIND_START(NematicElectric, nematicelectricref, nematicelectric_prepareref, nematicelectric_integrand, FUNCTIONAL_ARGS, nematicelectric_startfn)
FUNCTIONAL_MD_REF_INTEGRAND(NematicElectric, nematicelectricref, ref.grade)
FUNCTIONAL_MD_REF_TOTAL(NematicElectric, nematicelectricref, ref.grade)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(NematicElectric, nematicelectricref, ref.grade, NULL, SYMMETRY_NONE)
FUNCTIONAL_MD_REF_FIELDGRADIENT(NematicElectric, nematicelectricref, ref.grade, nematicelectric_cloneref, NULL)

MORPHO_BEGINCLASS(NematicElectric)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Field, Field)", NematicElectric_init__field_field, MORPHO_FN_MUTATES|MORPHO_FN_ALLOCATES),

FUNCTIONAL_MD_INTEGRAND_METHODS(NematicElectric),
FUNCTIONAL_MD_TOTAL_METHODS(NematicElectric),
FUNCTIONAL_MD_GRADIENT_METHODS(NematicElectric),
FUNCTIONAL_MD_FIELDGRADIENT_METHODS(NematicElectric)
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

/** Analytic fieldgradient: d/dq |q|^2 = 2q. */
bool normsq_fieldgradient(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectfield *grad) {
    fieldref *eref = ref;
    unsigned int nentries;
    double *entries, *gentries;

    if (!field_getelementaslist(eref->field, MESH_GRADE_VERTEX, id, 0, &nentries, &entries)) return false;
    if (!field_getelementaslist(grad, MESH_GRADE_VERTEX, id, 0, &nentries, &gentries)) return false;

    for (unsigned int i=0; i<nentries; i++) functional_accum(&gentries[i], 2.0*entries[i]);
    return true;
}

value NormSq_init__field(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    _gradsq_initfield(self, MORPHO_GETARG(args, 0));
    functional_setgrade(self, MESH_GRADE_VERTEX);
    return MORPHO_NIL;
}

FUNCTIONAL_MD_REF_BIND_FORCEGRADE_START(NormSq, fieldref, gradsq_prepareref, normsq_integrand, FUNCTIONAL_ARGS, MESH_GRADE_VERTEX, fieldref_startfn)
FUNCTIONAL_MD_REF_INTEGRAND_COST(NormSq, fieldref, MESH_GRADE_VERTEX, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_REF_TOTAL_COST(NormSq, fieldref, MESH_GRADE_VERTEX, FUNCTIONAL_COST_CHEAPEST)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(NormSq, fieldref, MESH_GRADE_VERTEX, NULL, SYMMETRY_NONE)
FUNCTIONAL_MD_REF_ANALYTICALFIELDGRADIENT_COST(NormSq, fieldref, MESH_GRADE_VERTEX, normsq_fieldgradient, FUNCTIONAL_COST_CHEAPEST)

MORPHO_BEGINCLASS(NormSq)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(Field)", NormSq_init__field, MORPHO_FN_MUTATES),

FUNCTIONAL_MD_INTEGRAND_METHODS(NormSq),
FUNCTIONAL_MD_TOTAL_METHODS(NormSq),
FUNCTIONAL_MD_GRADIENT_METHODS(NormSq),
FUNCTIONAL_MD_FIELDGRADIENT_METHODS(NormSq)
MORPHO_ENDCLASS

/* **********************************************************************
 * Integrals
 * ********************************************************************** */

/** Integral references
 @brief Immutable Integral definition passed through the functional map. */

typedef struct {
    value integrand;
    
    int nfields;
    value *fields;
    value *originalfields; // Original fields
    value method; // Method dictionary
    objectmesh *mref; // Reference mesh
    grade g; // Grade to integrate over
    bool weightbyref; // Use reference mesh for the element
    bool optimize; // Hint: may use chain-rule derivative shortcuts
} integralref;

#define INTEGRAL_ALLOW_ALL UINT_MAX

/** ----------------------------------------------
 * Detect use of special functions
 * ---------------------------------------------- */

enum { // constants that indicate which special functions are used in the integrand
    INTEGRAL_USES_NONE     = 0,
    INTEGRAL_USES_X        = 1u << 0,
    INTEGRAL_USES_GRAD     = 1u << 1,
    INTEGRAL_USES_HESS     = 1u << 2,
    INTEGRAL_USES_TANGENT  = 1u << 3,
    INTEGRAL_USES_NORMAL   = 1u << 4,
    INTEGRAL_USES_JACOBIAN = 1u << 5,
    INTEGRAL_USES_INVJ     = 1u << 6,
    INTEGRAL_USES_CG       = 1u << 7,
    INTEGRAL_USES_JUMPDN   = 1u << 8
};

/* Local fieldgradient may evaluate these; hess and jumpdn stay out. */
#define INTEGRAL_FIELDGRAD_ALLOWED (INTEGRAL_USES_GRAD | INTEGRAL_USES_TANGENT | \
    INTEGRAL_USES_NORMAL | INTEGRAL_USES_JACOBIAN | INTEGRAL_USES_INVJ | INTEGRAL_USES_CG)

#define INTEGRAL_MAXSPECIALS 8
static value _specialfns[INTEGRAL_MAXSPECIALS];
static unsigned _specialbits[INTEGRAL_MAXSPECIALS];
static unsigned int nspecials=0;

/** Add a special function to the list */
static void _addspecial(char *name, builtinfunction fn, unsigned bit) {
    if (nspecials<INTEGRAL_MAXSPECIALS) {
        _specialfns[nspecials]=builtin_addfunction(name, fn, MORPHO_FN_THREADLOCAL | MORPHO_FN_ALLOCATES | MORPHO_FN_THROWS);
        _specialbits[nspecials]=bit;
        nspecials++;
    } else UNREACHABLE("Too many special functions in functional.c");
}

/** Detect which special functions are used in the integrand */
static unsigned integral_fnuses(vm *v, value integrand) {
    unsigned uses=0;
    bool hit[INTEGRAL_MAXSPECIALS];

    if (optimize_fnaccessesarg(v, integrand, 0)) uses|=INTEGRAL_USES_X;
    optimize_fnloadsconstants(v, integrand, (int) nspecials, _specialfns, hit);
    for (unsigned int i=0; i<nspecials; i++) if (hit[i]) uses|=_specialbits[i];
    return uses;
}

/* ----------------------------------------------
 * Integrand functions
 * ---------------------------------------------- */

/** Integral element references
 @brief Thread-local integral context. */
#define ELREF_PERSISTENT   (1u<<0) /* Heap elref outlives a single integrand call */
#define ELREF_CONFIGURED   (1u<<1) /* method={} integrator is ready */
#define ELREF_HASTANGENT   (1u<<2)
#define ELREF_HASNORMAL    (1u<<3)
#define ELREF_HASJACOBIAN  (1u<<4)
#define ELREF_HASCG        (1u<<5)
#define ELREF_HASINVJ      (1u<<6)
#define ELREF_HASINTEG     (1u<<7) /* integrator_init has been called */
#define ELREF_GEOM         (ELREF_HASTANGENT|ELREF_HASNORMAL|ELREF_HASJACOBIAN|ELREF_HASCG|ELREF_HASINVJ)

typedef struct {
    object obj;
    objectmesh *mesh;    // The current mesh object
    
    integralref *iref;   // The current integral ref structure
    vm *v;               // Task VM (Integrate callbacks have no vm *)
    
// Information about the element reference:
    grade g;             // Current grade
    elementid id;        // Current element
    int nv;              // Number of vertices
    unsigned flags;
    unsigned allowed;    // Specials permitted; INTEGRAL_ALLOW_ALL = unrestricted
    int target_field;    // ifld being differentiated; -1 none
    bool target_grad_used;
    int freeze_grad;     // ifld whose qgrad is frozen for local FD; -1 none
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
    
// Per-task workspace (heap elref from taskstart)
    integrator integ;
    int nfields;
    
    objectmatrix *tangent;
    objectmatrix *normal;
    objectmatrix *jacobian;
    objectmatrix *invjacobian;
    objectmatrix *cgtensor;
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

int elementhandle;

/** Get the current element ref from thread-local storage in the VM */
objectintegralelementref *integral_getelementref(vm *v) {
    value elref=MORPHO_NIL;
    vm_gettlvar(v, elementhandle, &elref);
    if (MORPHO_ISINTEGRALELEMENTREF(elref)) return MORPHO_GETINTEGRALELEMENTREF(elref);
    
    return NULL;
}

/** Checks whether an existing matrix is the correct size and allocates if not. */
static objectmatrix *integral_ensurematrix(objectmatrix **slot, int nrows, int ncols) {
    if (!*slot || (*slot)->nrows!=nrows || (*slot)->ncols!=ncols) {
        if (*slot) object_free((object *) *slot);
        *slot=matrix_new(nrows, ncols, false);
    }
    return *slot;
}

/** Resets the geometry flags and other geometry-related data. */
static void integral_resetgeometryflags(objectintegralelementref *elref) {
    elref->flags &= ~ELREF_GEOM;
    elref->lambda=NULL;
    elref->posn=NULL;
    elref->qinterpolated=NULL;
}

/** Initialize an elref to a cleared state. */
static void _integral_initelref(objectintegralelementref *elref) {
    memset(elref, 0, sizeof(objectintegralelementref));
    object_init((object *) elref, OBJECT_INTEGRALELEMENTREF);
    elref->allowed=INTEGRAL_ALLOW_ALL;
    elref->target_field=-1;
    elref->freeze_grad=-1;
}

/** Bind an elref to a particular element and reset per-element geometry cache.
 * @warning Does not clear map policy (allowed, target_field). */
static void _integral_bindelref(objectintegralelementref *elref, objectmesh *mesh, grade g, elementid id, int nv, int *vid, double **vertexposn, integralref *iref) {
    elref->mesh=mesh;
    elref->g=g;
    elref->id=id;
    elref->nv=nv;
    elref->vid=vid;
    elref->vertexposn=vertexposn;
    elref->iref=iref;
    elref->freeze_grad=-1;
    integral_resetgeometryflags(elref);
}

/** Fill n values with nil. */
static void _integral_nilvalues(value *p, int n) {
    for (int i=0; i<n; i++) p[i]=MORPHO_NIL;
}

static void integral_releasegeometry(objectintegralelementref *elref) {
    if (elref->invj) { object_free((object *) elref->invj); elref->invj=NULL; }
    if (elref->tangent) { object_free((object *) elref->tangent); elref->tangent=NULL; }
    if (elref->normal) { object_free((object *) elref->normal); elref->normal=NULL; }
    if (elref->jacobian) { object_free((object *) elref->jacobian); elref->jacobian=NULL; }
    if (elref->invjacobian) { object_free((object *) elref->invjacobian); elref->invjacobian=NULL; }
    if (elref->cgtensor) { object_free((object *) elref->cgtensor); elref->cgtensor=NULL; }
    integral_resetgeometryflags(elref);
}

static void integral_freegradhess(int nfields, value *qgrad, value *qhess);

/** Release buffers and geometry; does not free the elref object itself. */
static void integral_clearelref(objectintegralelementref *elref) {
    if (!elref) return;
    
    if (elref->flags & ELREF_HASINTEG) integrator_clear(&elref->integ);
    
    if (elref->quantities) {
        for (int i=0; i<elref->nfields; i++) {
            if (elref->quantities[i].vals) MORPHO_FREE(elref->quantities[i].vals);
            if (elref->quantities[i].findx) MORPHO_FREE(elref->quantities[i].findx);
        }
        MORPHO_FREE(elref->quantities);
        elref->quantities=NULL;
    }
    
    if (elref->qgrad && elref->qhess) integral_freegradhess(elref->nfields, elref->qgrad, elref->qhess);
    if (elref->qgrad) { MORPHO_FREE(elref->qgrad); elref->qgrad=NULL; }
    if (elref->qhess) { MORPHO_FREE(elref->qhess); elref->qhess=NULL; }
    
    integral_releasegeometry(elref);
}

/** Frees a heap elref and attached data. */
static void integral_freeelref(objectintegralelementref *elref) {
    if (!elref) return;
    integral_clearelref(elref);
    object_free((object *) elref);
}

/* ----------------------------------------------
 * Jump interface references
 * ---------------------------------------------- */

/** Thread-local interface context for Jump functionals.
    iface is first so the JUMP object is layout-compatible with an elref. */
typedef struct {
    objectintegralelementref iface; /* Interface: integrator, oriented normal, bind */
    jumpref *jref;
    vm *v;
    objectintegralelementref plus;  /* Parent + : quantities, invj, lambda */
    objectintegralelementref minus; /* Parent - */
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

int jumpinterfacehandle;

static objectjumpinterfaceref *jump_getinterfaceref(vm *v) {
    value iref=MORPHO_NIL;
    vm_gettlvar(v, jumpinterfacehandle, &iref);
    if (MORPHO_ISJUMPINTERFACEREF(iref)) return MORPHO_GETJUMPINTERFACEREF(iref);
    
    return NULL;
}

static bool integral_contextactive(vm *v) {
    return integral_getelementref(v) || jump_getinterfaceref(v);
}

/** Reject specials not in elref->allowed. */
static bool integral_checkfastpath(vm *v, unsigned bit, const char *name) {
    objectintegralelementref *elref=integral_getelementref(v);
    if (!elref) return true;
    if (!(elref->allowed & bit)) MORPHO_FAILVARGS(v, INTEGRAL_FASTPATH, name);
    return true;
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

/** Evaluate the tangent vector */
static bool integral_evaluatetangent(vm *v) {
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref || elref->g!=1) MORPHO_FAILVARGS(v, INTEGRAL_SPCLFN, TANGENT_FUNCTION);
    
    int dim = elref->mesh->dim;
    objectmatrix *mtangent=integral_ensurematrix(&elref->tangent, dim, 1);
    if (!mtangent) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    
    functional_vecsub(dim, elref->vertexposn[1], elref->vertexposn[0], mtangent->elements);

    double tnorm=functional_vecnorm(dim, mtangent->elements);
    if (fabs(tnorm)>MORPHO_EPS) functional_vecscale(dim, 1.0/tnorm, mtangent->elements, mtangent->elements);
    
    elref->flags |= ELREF_HASTANGENT;
    return true;
}

static value integral_tangent(vm *v, int nargs, value *args) {
    if (!integral_checkfastpath(v, INTEGRAL_USES_TANGENT, TANGENT_FUNCTION)) return MORPHO_NIL;
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref || elref->g!=1) MORPHO_RAISEVARGS(v, INTEGRAL_SPCLFN, TANGENT_FUNCTION);
    
    if (!(elref->flags & ELREF_HASTANGENT) && !integral_evaluatetangent(v)) return MORPHO_NIL;
    return MORPHO_OBJECT(elref->tangent);
}

/* --------
 * Normal
 * -------- */

/** Evaluates the normal vector */
static bool integral_evaluatenormal(vm *v) {
    objectintegralelementref *elref = integral_getelementref(v);
    
    if (!elref || elref->g!=2) MORPHO_FAILVARGS(v, INTEGRAL_SPCLFN, NORMAL_FUNCTION);
    
    int dim = elref->mesh->dim;
    double s0[dim], s1[dim];
    objectmatrix *mnormal=integral_ensurematrix(&elref->normal, dim, 1);
    if (!mnormal) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    
    functional_vecsub(dim, elref->vertexposn[1], elref->vertexposn[0], s0);
    functional_vecsub(dim, elref->vertexposn[2], elref->vertexposn[1], s1);
    functional_veccross(s0, s1, mnormal->elements);
    
    double nnorm=functional_vecnorm(dim, mnormal->elements);
    if (fabs(nnorm)>MORPHO_EPS) functional_vecscale(dim, 1.0/nnorm, mnormal->elements, mnormal->elements);
    
    elref->flags |= ELREF_HASNORMAL;
    return true;
}

static value integral_normal(vm *v, int nargs, value *args) {
    if (!integral_checkfastpath(v, INTEGRAL_USES_NORMAL, NORMAL_FUNCTION)) return MORPHO_NIL;
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref || elref->g!=2) MORPHO_RAISEVARGS(v, INTEGRAL_SPCLFN, NORMAL_FUNCTION);
    
    if (!(elref->flags & ELREF_HASNORMAL) && !integral_evaluatenormal(v)) return MORPHO_NIL;
    return MORPHO_OBJECT(elref->normal);
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

static bool integral_ensureinvj(vm *v, objectintegralelementref *elref) {
    if (elref->flags & ELREF_HASINVJ) return true;
    if (!integral_ensurematrix(&elref->invj, elref->g, elref->mesh->dim) ||
        !integral_prepareinvjacobian(elref->mesh->dim, elref->g, elref->vertexposn, elref->invj)) {
        MORPHO_FAIL(v, INTEGRAL_DFFEVL);
    }
    elref->flags |= ELREF_HASINVJ;
    return true;
}

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

    if (ifld==elref->target_field) elref->target_grad_used=true;

    if (elref->freeze_grad==ifld && MORPHO_ISOBJECT(elref->qgrad[ifld])) {
        *out=elref->qgrad[ifld];
        return true;
    }
    
    // Extract information from the field
    objectfield *fld = MORPHO_GETFIELD(elref->iref->fields[ifld]);
    int dim = elref->mesh->dim;
    
    // Allocate objects if need be. Don't bind these; these will be freed when the elref is cleared.
    if (!MORPHO_ISOBJECT(elref->qgrad[ifld])) {
        if (!integral_gradalloc(dim, fld->prototype, &elref->qgrad[ifld])) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    }
    
    bool success=false;
    
    // Evaluate gradient. TODO: remove quantities check as we deprecate old integrator.
    if (MORPHO_ISFESPACE(fld->fnspc) && elref->quantities) {
        if (!integral_ensureinvj(v, elref)) return false;
        
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
    if (!integral_checkfastpath(v, INTEGRAL_USES_GRAD, GRAD_FUNCTION)) return MORPHO_NIL;
    value out=MORPHO_NIL;
    if (nargs!=1) MORPHO_RAISE(v, INTEGRAL_FLD);
    integral_evaluategradient(v, MORPHO_GETARG(args, 0), &out);
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
    
    if (MORPHO_ISFESPACE(fld->fnspc) && elref->quantities) {
        if (!integral_ensureinvj(v, elref)) return false;
        
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
    if (!integral_checkfastpath(v, INTEGRAL_USES_HESS, HESS_FUNCTION)) return MORPHO_NIL;
    value out=MORPHO_NIL;
    if (nargs!=1) MORPHO_RAISE(v, INTEGRAL_FLD);
    integral_evaluatehessian(v, MORPHO_GETARG(args, 0), &out);
    return out;
}

/* -------------------
 * Cauchy green strain
 * ------------------- */

/** Evaluates the cg strain tensor */
static bool integral_evaluatecg(vm *v) {
    objectintegralelementref *elref = integral_getelementref(v);
    
    if (!elref || !elref->iref->mref) MORPHO_FAILVARGS(v, INTEGRAL_SPCLFN, CGTENSOR_FUNCTION);
    
    int gdim=elref->nv-1; // Dimension of Gram matrix
    objectmatrix *cg=integral_ensurematrix(&elref->cgtensor, gdim, gdim);
    if (!cg) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    
    double gramrefel[gdim*gdim], gramdefel[gdim*gdim], qel[gdim*gdim], rel[gdim*gdim];
    objectmatrix gramref = MORPHO_STATICMATRIX(gramrefel, gdim, gdim); // Gram matrices
    objectmatrix gramdef = MORPHO_STATICMATRIX(gramdefel, gdim, gdim); //
    objectmatrix q = MORPHO_STATICMATRIX(qel, gdim, gdim); // Inverse of Gram in source domain
    objectmatrix r = MORPHO_STATICMATRIX(rel, gdim, gdim); // Intermediate calculations
    
    linearelasticity_calculategram(elref->iref->mref->vert, elref->mesh->dim, elref->nv, elref->vid, &gramref);
    linearelasticity_calculategram(elref->mesh->vert, elref->mesh->dim, elref->nv, elref->vid, &gramdef);
    
    if (matrix_copy(&gramref, &q)!=LINALGERR_OK) return false;
    if (matrix_inverse(&q)!=LINALGERR_OK) return false;
    if (matrix_mul(&gramdef, &q, &r)!=LINALGERR_OK) return false;

    if (matrix_identity(cg)!=LINALGERR_OK) return false;
    matrix_scale(cg, -0.5);
    matrix_axpy(0.5, &r, cg);
    
    elref->flags |= ELREF_HASCG;
    return true;
}

static value integral_cgfn(vm *v, int nargs, value *args) {
    if (!integral_checkfastpath(v, INTEGRAL_USES_CG, CGTENSOR_FUNCTION)) return MORPHO_NIL;
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref || !elref->iref->mref) MORPHO_RAISEVARGS(v, INTEGRAL_SPCLFN, CGTENSOR_FUNCTION);
    
    if (!(elref->flags & ELREF_HASCG) && !integral_evaluatecg(v)) return MORPHO_NIL;
    return MORPHO_OBJECT(elref->cgtensor);
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

void _fetchvertices(objectintegralelementref *elref, objectmesh *mesh, int nv, elementid *vid, double **x) {
    // Fetch reference vertices
    for (int j=0; j<nv; j++) matrix_getcolumnptr(elref->iref->mref->vert, vid[j], &x[j]);
}

void _edgevectors(grade g, int dim, double **x, double *out) {
    for (int i=0; i<g; i++) functional_vecsub(dim, x[i+1], x[0], out + i*dim);
}

/** Evaluates the jacobian and inverse jacobian */
static bool integral_evaluatejacobian(vm *v) {
    objectintegralelementref *elref = integral_getelementref(v);
    
    if (!elref) MORPHO_FAILVARGS(v, INTEGRAL_SPCLFN, JACOBIAN_FUNCTION);
    
    int dim = elref->mesh->dim;     // Dimension of the mesh
    objectmatrix *J=integral_ensurematrix(&elref->jacobian, dim, dim);
    objectmatrix *Jinv=integral_ensurematrix(&elref->invjacobian, dim, dim);
    if (!J || !Jinv) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    
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
    
    elref->flags |= ELREF_HASJACOBIAN;
    return true;
}

static value integral_jacobian(vm *v, int nargs, value *args) {
    if (!integral_checkfastpath(v, INTEGRAL_USES_JACOBIAN, JACOBIAN_FUNCTION)) return MORPHO_NIL;
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref) MORPHO_RAISEVARGS(v, INTEGRAL_SPCLFN, JACOBIAN_FUNCTION);
    
    if (!(elref->flags & ELREF_HASJACOBIAN) && !integral_evaluatejacobian(v)) return MORPHO_NIL;
    return MORPHO_OBJECT(elref->jacobian);
}

static value integral_invjacobian(vm *v, int nargs, value *args) {
    if (!integral_checkfastpath(v, INTEGRAL_USES_INVJ, INVJACOBIAN_FUNCTION)) return MORPHO_NIL;
    objectintegralelementref *elref = integral_getelementref(v);
    if (!elref) MORPHO_RAISEVARGS(v, INTEGRAL_SPCLFN, INVJACOBIAN_FUNCTION);
    
    if (!(elref->flags & ELREF_HASJACOBIAN) && !integral_evaluatejacobian(v)) return MORPHO_NIL;
    return MORPHO_OBJECT(elref->invjacobian);
}

/* ----------------------------------------------
 * Generic integral support functions
 * ---------------------------------------------- */

value functional_methodproperty;
value functional_optimizeproperty;

/** Prepares an integral reference */
bool integral_prepareref(objectinstance *self, objectmesh *mesh, grade g, objectselection *sel, integralref *ref) {
    bool success=false;
    value func=MORPHO_NIL;
    value mref=MORPHO_NIL;
    value wtbyref=MORPHO_NIL;
    value field=MORPHO_NIL;
    value method=MORPHO_NIL;
    value optimize=MORPHO_NIL;
    ref->nfields=0;
    ref->method=MORPHO_NIL;
    ref->mref=NULL;
    ref->g=g;
    ref->weightbyref=false;
    ref->optimize=true;

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
    if (objectinstance_getpropertyinterned(self, functional_optimizeproperty, &optimize) &&
        MORPHO_ISBOOL(optimize)) {
        ref->optimize=MORPHO_GETBOOLVALUE(optimize);
    }
    if (objectinstance_getpropertyinterned(self, functional_fieldproperty, &field) &&
        MORPHO_ISLIST(field)) {
        objectlist *list = MORPHO_GETLIST(field);
        ref->nfields=list->val.count;
        ref->fields=list->val.data;
        ref->originalfields=list->val.data;
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

/** Ensure quantity.vals and findx can hold at least n entries. */
static bool _integral_ensurequantityvals(quantity *q, int n) {
    if (q->capacity>=n) return true;
    value *vals=MORPHO_REALLOC(q->vals, sizeof(value)*n);
    if (!vals) return false;
    fieldindx *findx=MORPHO_REALLOC(q->findx, sizeof(fieldindx)*n);
    if (!findx) { q->vals=vals; return false; } /* vals kept; capacity unchanged */
    q->vals=vals;
    q->findx=findx;
    q->capacity=n;
    return true;
}

/** Prepares quantity list. Prefers to reuse existing buffers if possible. */
bool integral_preparequantities(integralref *iref, int nv, int *vid, quantity *quantities) {
    for (int k=0; k<iref->nfields; k++) {
        objectfield *f=MORPHO_GETFIELD(iref->fields[k]);
        
        if (MORPHO_ISFESPACE(f->fnspc)) {
            fespace *disc=MORPHO_GETFESPACE(f->fnspc)->fespace;
            if (nv-1<disc->grade) {
                if (!fespace_lower(disc, nv-1, &disc)) return false;
            }
            
            quantities[k].nnodes=disc->nnodes;
            quantities[k].ifn=disc->ifn;
            if (!_integral_ensurequantityvals(&quantities[k], disc->nnodes)) return false;
            if (!fespace_doftofieldindx(f, disc, nv, vid, quantities[k].findx)) return false;
            
            for (int i=0; i<disc->nnodes; i++) {
                int dof;
                fieldindx *fx=&quantities[k].findx[i];
                if (!field_getindex(f, fx->g, fx->id, fx->indx, &dof)) return false;
                if (!field_getelementwithindex(f, dof, &quantities[k].vals[i])) return false;
            }
        } else {
            quantities[k].nnodes=nv;
            quantities[k].ifn=NULL;
            if (!_integral_ensurequantityvals(&quantities[k], nv)) return false;
            for (unsigned int i=0; i<nv; i++) {
                quantities[k].findx[i]=(fieldindx){ .g=MESH_GRADE_VERTEX, .id=vid[i], .indx=0 };
                if (!field_getelement(f, MESH_GRADE_VERTEX, vid[i], 0, &quantities[k].vals[i])) return false;
            }
        }
    }
    return true;
}

/** Clears a list of quantities */
void integral_clearquantities(int nq, quantity *quantities) {
    for (int k=0; k<nq; k++) {
        if (quantities[k].vals) MORPHO_FREE(quantities[k].vals);
        if (quantities[k].findx) MORPHO_FREE(quantities[k].findx);
    }
    memset(quantities, 0, sizeof(quantity)*nq);
}

bool integral_integrandfn(unsigned int dim, double *t, double *x, unsigned int nquantity, value *quantity, void *ref, unsigned int nout, double *fout) {
    objectintegralelementref *elref = ref;
    integralref *iref = elref->iref;
    vm *v = elref->v;
    objectmatrix posn = MORPHO_STATICMATRIX(x, dim, 1);
    value args[nquantity+1], out;

    if (nout!=1) return false;

    // The integrand function is called with the position and then interpolated quantities.
    args[0]=MORPHO_OBJECT(&posn);
    for (unsigned int i=0; i<nquantity; i++) args[i+1]=quantity[i];
    
    elref->lambda=t;
    elref->posn=x;
    elref->qinterpolated=quantity;
    
    if (morpho_call(v, iref->integrand, nquantity+1, args, &out)) {
        morpho_valuetofloat(out, fout);
        return true;
    }

    return false;
}

/** Bind elref to an element, compute its measure, and fill vertex coordinate pointers. */
static bool integral_bindelement(vm *v, objectmesh *mesh, elementid id, int nv, int *vid,
                                 integralref *iref, objectintegralelementref *elref, double **x) {
    _integral_bindelref(elref, mesh, iref->g, id, nv, vid, x, iref);
    if (!functional_elementsize(v, iref->weightbyref ? iref->mref : mesh, iref->g, id, nv, vid, &elref->elementsize))
        return false;
    for (int i=0; i<nv; i++) mesh_getvertexcoordinatesaslist(mesh, vid[i], &x[i]);
    return true;
}

/** Claim elref->quantities or qlocal and prepare DOF values/findx. If *localq, caller must detach. */
static bool integral_prepareelementquantities(integralref *iref, objectintegralelementref *elref,
                                              int nv, int *vid, quantity *qlocal,
                                              quantity **quantities, bool *localq) {
    *quantities=elref->quantities ? elref->quantities : qlocal;
    *localq=(*quantities==qlocal);
    if (*localq) {
        memset(qlocal, 0, sizeof(quantity)*(size_t)(iref->nfields+1));
        elref->quantities=*quantities;
    }
    return integral_preparequantities(iref, nv, vid, *quantities);
}

/** Release a stack-local quantities buffer claimed by prepareelementquantities. */
static void integral_detachquantities(objectintegralelementref *elref, int nfields, quantity *quantities, bool localq) {
    if (!localq) return;
    elref->quantities=NULL;
    integral_clearquantities(nfields, quantities);
}

/** Integrate a callable over elements of the grade stored in the integral ref */
bool integral_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, double *out) {
    integralref *iref = (integralref *) ref;
    grade g = iref->g;
    double *x[nv];
    bool success=false;
    objectintegralelementref stackelref;
    objectintegralelementref *elref = integral_getelementref(v);
    bool persistent = (elref && (elref->flags & ELREF_PERSISTENT));
    value qgrad_local[iref->nfields+1], qhess_local[iref->nfields+1];
    quantity quantities_local[iref->nfields+1];
    quantity *quantities=NULL;
    bool localquantities=false;
    
    if (!persistent) {
        _integral_initelref(&stackelref);
        elref = &stackelref;
        _integral_nilvalues(qgrad_local, iref->nfields);
        _integral_nilvalues(qhess_local, iref->nfields);
        elref->qgrad=qgrad_local;
        elref->qhess=qhess_local;
        vm_settlvar(v, elementhandle, MORPHO_OBJECT(elref));
    }
    
    elref->v=v;
    if (!integral_bindelement(v, mesh, id, nv, vid, iref, elref, x)) goto integral_integrand_cleanup;

    if (MORPHO_ISDICTIONARY(iref->method)) {
        if (!integral_prepareelementquantities(iref, elref, nv, vid, quantities_local, &quantities, &localquantities))
            goto integral_integrand_cleanup;
        
        if (elref->flags & ELREF_CONFIGURED) {
            success=integrator_integrate(&elref->integ, integral_integrandfn, mesh->dim, x, iref->nfields, quantities, elref, 1, out);
        } else {
            double err;
            success=integrate(integral_integrandfn, MORPHO_GETDICTIONARY(iref->method), morpho_geterror(v), mesh->dim, g, x, iref->nfields, quantities, elref, out, &err);
        }
    } else { // Old integrator
        value qstore[nv][iref->nfields+1];
        value *q[nv];
        for (unsigned int i=0; i<nv; i++) q[i]=qstore[i];
        for (unsigned int k=0; k<iref->nfields; k++) {
            for (unsigned int i=0; i<nv; i++) {
                field_getelement(MORPHO_GETFIELD(iref->fields[k]), MESH_GRADE_VERTEX, vid[i], 0, &q[i][k]);
            }
        }
        
        success=integrate_integrate(integral_integrandfn, mesh->dim, g, x, iref->nfields, q, elref, out);
    }
    
    if (success) *out *= elref->elementsize;

integral_integrand_cleanup:
    if (!persistent) vm_settlvar(v, elementhandle, MORPHO_NIL);
    integral_detachquantities(elref, iref->nfields, quantities, localquantities);
    if (!persistent) {
        integral_releasegeometry(elref);
        integral_freegradhess(iref->nfields, elref->qgrad, elref->qhess);
    }
    
    return success;
}

static void *_integral_zalloc(int n, size_t size) {
    size_t bytes=(size_t) n*size;
    void *p=MORPHO_MALLOC(bytes);
    if (p) memset(p, 0, bytes);
    return p;
}

static bool integral_taskstart(vm *v, functional_mapinfo *info) {
    integralref *iref=(integralref *) info->ref;
    objectintegralelementref *elref=NULL;
    
    if (integral_contextactive(v)) MORPHO_FAIL(v, INTEGRAL_NESTED);
    
    elref=(objectintegralelementref *) object_new(sizeof(objectintegralelementref), OBJECT_INTEGRALELEMENTREF);
    if (!elref) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    
    _integral_initelref(elref);
    elref->v=v;
    elref->iref=iref;
    integrator_init(&elref->integ);
    elref->flags |= ELREF_PERSISTENT | ELREF_HASINTEG;
    
    if (MORPHO_ISDICTIONARY(iref->method)) {
        if (!integrator_configurewithdictionary(&elref->integ, morpho_geterror(v), info->g, MORPHO_GETDICTIONARY(iref->method))) goto integral_taskstart_cleanup;
        elref->flags |= ELREF_CONFIGURED;
    }
    
    elref->nfields=iref->nfields;
    if (iref->nfields>0) {
        elref->qgrad=_integral_zalloc(iref->nfields, sizeof(value));
        elref->qhess=_integral_zalloc(iref->nfields, sizeof(value));
        if (!elref->qgrad || !elref->qhess) goto integral_taskstart_cleanup;
        
        if (elref->flags & ELREF_CONFIGURED) {
            elref->quantities=_integral_zalloc(iref->nfields, sizeof(quantity));
            if (!elref->quantities) goto integral_taskstart_cleanup;
        }
    }
    
    vm_settlvar(v, elementhandle, MORPHO_OBJECT(elref));
    return true;

integral_taskstart_cleanup:
    if (!morpho_checkerror(morpho_geterror(v))) morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    integral_freeelref(elref);
    return false;
}

static void integral_taskend(vm *v, functional_mapinfo *info) {
    objectintegralelementref *elref=integral_getelementref(v);
    if (!elref || !(elref->flags & ELREF_PERSISTENT)) return;
    vm_settlvar(v, elementhandle, MORPHO_NIL);
    integral_freeelref(elref);
}

/** One-pass shape gradient: I_ref * d(measure)/dx. */
static bool integral_gradient_fq(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, objectmatrix *frc) {
    integralref *iref=(integralref *) ref;
    objectintegralelementref *elref=integral_getelementref(v);
    double E=0.0;

    if (iref->weightbyref) return true;
    if (elref) elref->allowed=INTEGRAL_USES_NONE;
    if (!integral_integrand(v, mesh, id, nv, vid, ref, &E)) return false;

    elref=integral_getelementref(v);
    if (!elref) return false;
    if (elref->elementsize<=MORPHO_EPS) return true;

    return functional_elementgradient_scale(v, mesh, iref->g, id, nv, vid, frc, E/elref->elementsize);
}

/** True if integrand is f(q)-only (no x, no Integral specials). */
static bool integral_checkfieldonly(vm *v, value integrand) {
    return integral_fnuses(v, integrand)==INTEGRAL_USES_NONE;
}

static bool integral_field_hasgradient(objectfield *field) {
    if (!field || !MORPHO_ISFESPACE(field->fnspc)) return false;
    return FESPACE_HASGRADIENT(MORPHO_GETFESPACE(field->fnspc)->fespace);
}

/** Local fieldgradient: hess/jumpdn force global FD. Static grad() needs a gradient on the target. */
static bool integral_checklocalfieldgrad(vm *v, value integrand, objectfield *field) {
    unsigned uses=integral_fnuses(v, integrand);
    if (uses & ~(INTEGRAL_USES_X | INTEGRAL_FIELDGRAD_ALLOWED)) return false;
    if ((uses & INTEGRAL_USES_GRAD) && !integral_field_hasgradient(field)) return false;
    return true;
}

typedef struct {
    integralref *iref;
    objectfield *field;
    int ifield, ncomp;
    unsigned allowed;
} integral_fq_mapref;

static void _integral_fq_mapref(integral_fq_mapref *mref, integralref *iref, objectfield *field, int ifield, unsigned allowed) {
    mref->iref=iref;
    mref->field=field;
    mref->ifield=ifield;
    mref->ncomp=(int) field->psize;
    mref->allowed=allowed;
}

typedef struct {
    integral_fq_mapref *mref;
    objectintegralelementref *elref;
    int nnodes;
} integral_fq_local;

typedef struct {
    integral_fq_mapref local;
    functional_numericalfieldgradientref numerical;
    objectfield *fieldclone;
    functional_mapinfo *info;
    bool clone;
} integral_fieldgradient_taskref;

/* ----------------------------------------------
 * Local derivatives at a quadrature point
 * ---------------------------------------------- */

/** Central difference of the integrand wrt *p. Restores *p.
 * @details If qfloat is non-NULL, q is a Morpho Float and *qfloat is rewritten
 * after each step so the integrand sees the new value. */
static bool _integral_centdiff(unsigned int dim, double *lambda, double *x, unsigned int nq, value *quantity,
                              objectintegralelementref *elref, double *p, value *qfloat, double *df) {
    double f0=*p, eps=functional_fdstepsize(f0, 1), f[2];
    bool ok=true;
    for (int s=0; s<2 && ok; s++) {
        *p=f0+(s ? -eps : eps);
        if (qfloat) *qfloat=MORPHO_FLOAT(*p);
        ok=integral_integrandfn(dim, lambda, x, nq, quantity, elref, 1, &f[s]);
    }
    *p=f0;
    if (qfloat) *qfloat=MORPHO_FLOAT(f0);
    if (ok) *df=(f[0]-f[1])/(2.0*eps);
    return ok;
}

/** Estimate dfdq by central difference of the interpolated field value. */
static bool integral_fq_dfdq(unsigned int dim, double *lambda, double *x, unsigned int nq, value *quantity, objectintegralelementref *elref, value *qval, int ncomp, double *dfdq) {
    double *p, tmp; objectmatrix *m=NULL;

    if (MORPHO_ISFLOAT(*qval)) {
        if (ncomp!=1) return false;
        tmp=MORPHO_GETFLOATVALUE(*qval); p=&tmp;
    } else if (MORPHO_ISMATRIX(*qval)) {
        m=MORPHO_GETMATRIX(*qval);
        if ((int) m->nels!=ncomp) return false;
        p=m->elements;
    } else return false;

    for (int c=0; c<ncomp; c++) {
        if (!_integral_centdiff(dim, lambda, x, nq, quantity, elref, &p[c], m ? NULL : qval, &dfdq[c])) return false;
    }
    return true;
}

/** Flatten a grad() Matrix (or List of Matrices) to double pointers. */
static int integral_grad_ptrs(value gval, double **ptrs, int maxn) {
    int n=0;
    if (MORPHO_ISMATRIX(gval)) {
        objectmatrix *m=MORPHO_GETMATRIX(gval);
        for (unsigned int i=0; i<m->nels && n<maxn; i++) ptrs[n++]=&m->elements[i];
        return n;
    }
    if (MORPHO_ISLIST(gval)) {
        objectlist *lst=MORPHO_GETLIST(gval);
        for (unsigned int i=0; i<list_length(lst); i++) {
            value el;
            if (!list_getelement(lst, (int) i, &el) || !MORPHO_ISMATRIX(el)) return -1;
            objectmatrix *m=MORPHO_GETMATRIX(el);
            for (unsigned int k=0; k<m->nels && n<maxn; k++) ptrs[n++]=&m->elements[k];
        }
        return n;
    }
    return -1;
}

/** Estimate dfdgrad by central difference of the cached grad() result.
 * @details freeze_grad stops EvaluateGradient from overwriting the perturbation. */
static bool integral_fq_dfdgrad(unsigned int dim, double *lambda, double *x, unsigned int nq, value *quantity, objectintegralelementref *elref, int ifield, int ngrad, double *dfdg) {
    double *ptrs[ngrad];

    if (!elref || integral_grad_ptrs(elref->qgrad[ifield], ptrs, ngrad)!=ngrad) return false;
    for (int i=0; i<ngrad; i++) {
        if (!_integral_centdiff(dim, lambda, x, nq, quantity, elref, ptrs[i], NULL, &dfdg[i])) return false;
    }
    return true;
}

/** Shape-function gradients in physical coordinates.
 * @details out[k*nnodes+a] is dNa/dx_k. */
static bool integral_physical_gradNa(vm *v, objectintegralelementref *elref, objectfield *fld, int nnodes, double *out) {
    if (!elref || !fld || !MORPHO_ISFESPACE(fld->fnspc)) return false;
    fespace *disc=MORPHO_GETFESPACE(fld->fnspc)->fespace;
    if (!FESPACE_HASGRADIENT(disc) || disc->nnodes!=nnodes) return false;
    if (!integral_ensureinvj(v, elref)) return false;

    double gdata[nnodes * elref->g];
    objectmatrix gmat=MORPHO_STATICMATRIX(gdata, nnodes, elref->g);
    objectmatrix fmat=MORPHO_STATICMATRIX(out, nnodes, elref->mesh->dim);
    fespace_gradient(disc, elref->lambda, &gmat);
    return matrix_mul(&gmat, elref->invj, &fmat)==LINALGERR_OK;
}

/** Integrand for local fieldgradient assembly: (dfdq) Na [+ (dfdgrad) · ∇Na]. */
static bool integral_fq_vector_integrand(unsigned int dim, double *lambda, double *x, unsigned int nquantity, value *quantity, void *ref, unsigned int nout, double *fout) {
    integral_fq_local *s=(integral_fq_local *) ref;
    integral_fq_mapref *mref=s->mref;
    objectintegralelementref *elref=s->elref;
    vm *v=elref->v;
    int nnodes=s->nnodes, ncomp=mref->ncomp, ifield=mref->ifield, ngrad=(int) dim*ncomp;
    double Na[nnodes], dfdq[ncomp];

    if (!elref || (unsigned)(nnodes*ncomp)!=nout) return false;
    if (elref->quantities && elref->quantities[ifield].ifn) (elref->quantities[ifield].ifn)(lambda, Na);
    else for (int a=0; a<nnodes; a++) Na[a]=lambda[a];
    elref->lambda=lambda; elref->posn=x; elref->qinterpolated=quantity;

    elref->target_grad_used=false;
    if (!integral_fq_dfdq(dim, lambda, x, nquantity, quantity, elref, &quantity[ifield], ncomp, dfdq)) return false;

    if (!elref->target_grad_used) {
        for (int a=0; a<nnodes; a++) for (int c=0; c<ncomp; c++) fout[a*ncomp+c]=dfdq[c]*Na[a];
        return true;
    }

    elref->freeze_grad=ifield;
    double dfdg[ngrad], gNa[nnodes*(int) dim];
    bool ok=integral_fq_dfdgrad(dim, lambda, x, nquantity, quantity, elref, ifield, ngrad, dfdg);
    elref->freeze_grad=-1;
    if (!ok) return false;
    if (!integral_physical_gradNa(v, elref, mref->field, nnodes, gNa)) return false;

    for (int a=0; a<nnodes; a++) {
        for (int c=0; c<ncomp; c++) {
            double g=dfdq[c]*Na[a];
            for (unsigned int k=0; k<dim; k++) g+=dfdg[k*ncomp+c]*gNa[k*nnodes+a];
            fout[a*ncomp+c]=g;
        }
    }
    return true;
}

/** Determine whether the local integrand is no more expensive than numerical FD. */
static bool integral_fieldgradient_preferlocal(integrator *integ, bool usesgrad, unsigned int dim, int nnodes, int ncomp) {
    if (!integ->acceptedrule) return false;
    unsigned qapply=(unsigned) integ->acceptedrule->nnodes;
    unsigned clocal=qapply*2u*(unsigned) ncomp*(1u+(usesgrad?dim:0u));
    unsigned cfd=2u*(unsigned) nnodes*(unsigned) ncomp*integ->nevals;
    return clocal<=cfd;
}

/** Add this element's fieldgradient into out via the local integrand, or numerical FD of the scalar integral if try refines or FD is cheaper. */
static bool integral_fieldgradient_fq_element(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *ref, void *out) {
    integral_fieldgradient_taskref *tref=ref;
    integral_fq_mapref *mref=&tref->local;
    integralref *iref=mref->iref;
    objectfield *grad=(objectfield *) out;
    objectintegralelementref *elref=integral_getelementref(v);
    double *x[nv];
    quantity qlocal[iref->nfields+1], *quantities=NULL;
    bool localq=false, ok=false;
    int ifield=mref->ifield, ncomp=mref->ncomp;

    if (!elref || !(elref->flags & ELREF_CONFIGURED)) return false;

    elref->v=v;
    elref->allowed=mref->allowed;
    elref->target_field=mref->ifield;
    if (!integral_bindelement(v, mesh, id, nv, vid, iref, elref, x)) return false;
    if (elref->elementsize<=MORPHO_EPS) return true;
    if (!integral_prepareelementquantities(iref, elref, nv, vid, qlocal, &quantities, &localq)) {
        integral_detachquantities(elref, iref->nfields, quantities, localq);
        return false;
    }

    int nnodes=quantities[ifield].nnodes;
    unsigned int nout=(unsigned)(nnodes*ncomp);
    double local[nout];
    integral_fq_local s={ .mref=mref, .elref=elref, .nnodes=nnodes };
    bool numerical=false;

    if (!elref->integ.adapt) {
        ok=integrator_integrate(&elref->integ, integral_fq_vector_integrand,
                          mesh->dim, x, iref->nfields, quantities, &s, nout, local);
    } else {
        double fval=0.0;
        elref->target_grad_used=false;
        integratortrystatus st=integrator_try(&elref->integ, integral_integrandfn,
                          mesh->dim, x, iref->nfields, quantities, elref, 1, &fval);
        if (st==INTEGRATOR_TRY_FAILED) {
            integral_detachquantities(elref, iref->nfields, quantities, localq);
            return false;
        }
        if (st==INTEGRATOR_TRY_ACCEPTED &&
            integral_fieldgradient_preferlocal(&elref->integ, elref->target_grad_used, mesh->dim, nnodes, ncomp)) {
            ok=integrator_apply(&elref->integ, integral_fq_vector_integrand, &s, nout, local);
        } else numerical=true;
    }

    if (numerical) {
        integral_detachquantities(elref, iref->nfields, quantities, localq);
        if (!tref->numerical.field &&
            !functional_preparenumericalfieldgradientref(v, tref->info, tref->clone, &tref->numerical, &tref->fieldclone)) return false;
        return functional_numericalfieldgradientmapfn(v, mesh, id, nv, vid, &tref->numerical, out);
    }

    for (int a=0; ok && a<nnodes; a++) {
        fieldindx *fx=&quantities[ifield].findx[a];
        unsigned int nentries=0; double *gentry=NULL;
        ok=field_getelementaslist(grad, fx->g, fx->id, fx->indx, &nentries, &gentry) &&
           ncomp<=(int) nentries;
        if (ok) for (int c=0; c<ncomp; c++) functional_accum(&gentry[c], elref->elementsize*local[a*ncomp+c]);
    }

    integral_detachquantities(elref, iref->nfields, quantities, localq);
    return ok;
}

/** Map fieldgradient with the local integrand on an accepted formula, or numerical FD if that formula is insufficient. */
static bool integral_mapfieldgradient(vm *v, functional_mapinfo *info, value *out) {
    int success=false, ntask=functional_ntasks(info), ifield=-1;
    functional_task task[ntask];
    integral_fieldgradient_taskref tref[ntask];
    integralref *iref=(integralref *) info->ref;
    varray_elementid imageids;
    objectfield *new=NULL;
    unsigned allowed=INTEGRAL_FIELDGRAD_ALLOWED;

    memset(tref, 0, sizeof(tref));

    for (int i=0; i<iref->nfields; i++)
        if (MORPHO_ISFIELD(iref->fields[i]) && MORPHO_GETFIELD(iref->fields[i])==info->field) { ifield=i; break; }
    if (ifield<0) return false;

    varray_elementidinit(&imageids);
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    new=object_newfield(info->mesh, info->field->prototype, info->field->fnspc, info->field->dof);
    if (!new) { morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED); goto cleanup; }
    field_zero(new);

    for (int i=0; i<ntask; i++) {
        _integral_fq_mapref(&tref[i].local, iref, info->field, ifield, allowed);
        tref[i].info=info;
        tref[i].clone=(ntask>1);
        task[i].ref=&tref[i];
        task[i].mapfn=integral_fieldgradient_fq_element;
        task[i].result=new;
    }
    if (!functional_map(ntask, task)) goto cleanup;
    success=true; *out=MORPHO_OBJECT(new);
cleanup:
    for (int i=0; i<ntask; i++) {
        functional_clearnumericalfieldgradientref(info, &tref[i].numerical, tref[i].fieldclone);
    }
    if (!success && new) object_free((object *) new);
    functional_cleanuptasks(v, ntask, task, &imageids);
    return success;
}

/** Prepare stored Fields' FE spaces once per map. */
static bool integral_startfn(vm *v, functional_mapinfo *info) {
    integralref *ref = (integralref *) info->ref;
    return functional_preparefieldlist(v, ref->fields, ref->nfields, info->g);
}

/** Shared bindref for Line/Area/Volume integrals. Grade comes from the
 * instance; startfn prepares any stored Fields' FE spaces. */
static bool _Integral_bindref(vm *v, objectinstance *self, functional_mapinfo *info, integralref *ref) {
    grade g=0;

    if (integral_contextactive(v)) MORPHO_FAIL(v, INTEGRAL_NESTED);

    if (!functional_readgrade(self, &g) ||
        !integral_prepareref(self, info->mesh, (info->g < 0 ? g : info->g), info->sel, ref)) {
        MORPHO_FAIL(v, INTEGRAL_ARGS);
    }
    if (info->g < 0) info->g = g;
    info->ref = ref;
    info->integrand = integral_integrand;
    info->start = integral_startfn;
    info->taskstart = integral_taskstart;
    info->taskend = integral_taskend;
    return true;
}

static value _Integral_gradient(vm *v, objectinstance *self, functional_mapinfo *info) {
    integralref ref;
    functional_mapcallback *mapfn=functional_mapnumericalgradient;

    if (!_Integral_bindref(v, self, info, &ref)) return MORPHO_NIL;
    if (ref.optimize && integral_checkfieldonly(v, ref.integrand)) {
        info->grad=integral_gradient_fq;
        mapfn=functional_mapgradient;
    }
    return _functional_run(v, info, ref.g, mapfn, true);
}

static value _Integral_fieldgradient(vm *v, objectinstance *self, functional_mapinfo *info) {
    integralref ref;
    functional_mapcallback *mapfn=functional_mapnumericalfieldgradient;

    if (!_Integral_bindref(v, self, info, &ref)) return MORPHO_NIL;
    info->cloneref=integral_cloneref;
    info->freeref=integral_freeref;
    if (ref.optimize && MORPHO_ISDICTIONARY(ref.method) &&
        integral_checklocalfieldgrad(v, ref.integrand, info->field)) {
        mapfn=integral_mapfieldgradient;
    }
    return _functional_run(v, info, ref.g, mapfn, true);
}

FUNCTIONAL_MD_REF_INTEGRAND(Integral, integralref, ref.g)
FUNCTIONAL_MD_REF_TOTAL(Integral, integralref, ref.g)
FUNCTIONAL_MD_REF_OVERLOADS(Integral, gradient, _Integral_gradient)
FUNCTIONAL_MD_REF_HESSIAN(Integral, integralref, ref.g, NULL, SYMMETRY_NONE)
FUNCTIONAL_MD_REF_FIELD_OVERLOADS(Integral, fieldgradient, _Integral_fieldgradient)

/** Initialize a Line/Area/Volume/Jump integral object */
static value integral_init(vm *v, int nargs, value *args) {
    objectinstance *self = MORPHO_GETINSTANCE(MORPHO_SELF(args));
    int nparams = -1;
    int nfixed;
    value method=MORPHO_NIL;
    value mref=MORPHO_NIL;
    value wtbyref=MORPHO_NIL;
    value optimize=MORPHO_NIL;

    if (builtin_options(v, nargs, args, &nfixed, 4,
                        functional_methodproperty, &method,
                        linearelasticity_referenceproperty, &mref,
                        linearelasticity_weightbyreferenceproperty, &wtbyref,
                        functional_optimizeproperty, &optimize)) {
        if (MORPHO_ISDICTIONARY(method)) {
            objectinstance_setproperty(self, functional_methodproperty, method);
        } else if (!MORPHO_ISNIL(method)) MORPHO_RAISE(v, INTEGRAL_ARGS);

        if (MORPHO_ISMESH(mref)) objectinstance_setproperty(self, linearelasticity_referenceproperty, mref);
        if (MORPHO_ISBOOL(wtbyref)) objectinstance_setproperty(self, linearelasticity_weightbyreferenceproperty, wtbyref);
        if (MORPHO_ISBOOL(optimize)) {
            objectinstance_setproperty(self, functional_optimizeproperty, optimize);
        } else if (!MORPHO_ISNIL(optimize)) MORPHO_RAISE(v, INTEGRAL_ARGS);
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
        for (unsigned int i=1; i<nfixed; i++) {
            if (!MORPHO_ISFIELD(MORPHO_GETARG(args, i))) MORPHO_RAISE(v, INTEGRAL_ARGS);
        }

        objectlist *list = object_newlist(nfixed-1, & MORPHO_GETARG(args, 1));
        if (list) objectinstance_setproperty(self, functional_fieldproperty, MORPHO_OBJECT(list));
        return morpho_wrapandbind(v, (object *) list);
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

#define INTEGRAL_MAPFLAGS  (MORPHO_FN_REENTRANT|FUNCTIONAL_MD_MAPFLAGS)
#define INTEGRAL_TOTALFLAGS (MORPHO_FN_REENTRANT|FUNCTIONAL_MD_TOTALFLAGS)
#define INTEGRAL_ELEMFLAGS (MORPHO_FN_REENTRANT|FUNCTIONAL_MD_ELEMFLAGS)
#define INTEGRAL_INITFLAGS (MORPHO_FN_MUTATES|MORPHO_FN_ALLOCATES|MORPHO_FN_THROWS|MORPHO_FN_OPTARGS)

MORPHO_BEGINCLASS(LineIntegral)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(...)", LineIntegral_init, INTEGRAL_INITFLAGS),
FUNCTIONAL_MD_INTEGRAND_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS, INTEGRAL_ELEMFLAGS),
FUNCTIONAL_MD_TOTAL_METHODS_FLAGS(Integral, INTEGRAL_TOTALFLAGS),
FUNCTIONAL_MD_GRADIENT_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS),
FUNCTIONAL_MD_FIELDGRADIENT_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS),
FUNCTIONAL_MD_HESSIAN_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS)
MORPHO_ENDCLASS

MORPHO_BEGINCLASS(AreaIntegral)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(...)", AreaIntegral_init, INTEGRAL_INITFLAGS),
FUNCTIONAL_MD_INTEGRAND_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS, INTEGRAL_ELEMFLAGS),
FUNCTIONAL_MD_TOTAL_METHODS_FLAGS(Integral, INTEGRAL_TOTALFLAGS),
FUNCTIONAL_MD_GRADIENT_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS),
FUNCTIONAL_MD_FIELDGRADIENT_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS),
FUNCTIONAL_MD_HESSIAN_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS)
MORPHO_ENDCLASS

MORPHO_BEGINCLASS(VolumeIntegral)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(...)", VolumeIntegral_init, INTEGRAL_INITFLAGS),
FUNCTIONAL_MD_INTEGRAND_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS, INTEGRAL_ELEMFLAGS),
FUNCTIONAL_MD_TOTAL_METHODS_FLAGS(Integral, INTEGRAL_TOTALFLAGS),
FUNCTIONAL_MD_GRADIENT_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS),
FUNCTIONAL_MD_FIELDGRADIENT_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS),
FUNCTIONAL_MD_HESSIAN_METHODS_FLAGS(Integral, INTEGRAL_MAPFLAGS)
MORPHO_ENDCLASS

/* ----------------------------------------------
 * Jump
 * ---------------------------------------------- */

typedef enum {
    JUMP_STRATEGY_CENTROID_MODE,
    JUMP_STRATEGY_QUADRATURE_MODE
} jumpstrategy;

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
    Optional arguments match Integral: 'method', 'mref', 'weightbyreference' and 'optimize'. */
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
    /* Check the interface grade: Jump needs a trace, which CG0 does not provide. */
    return functional_preparefieldlist(v, ref->integral.fields, ref->integral.nfields, ref->interfacegrade);
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

    if (!functional_countelements(info->mesh, ref->interfacegrade, &n, &ifaceverts)) return false;
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

static bool jump_ensuresidequantities(objectintegralelementref *side, int nfields) {
    if (nfields<=0) return true;
    if (side->quantities) return true;
    side->quantities=_integral_zalloc(nfields, sizeof(quantity));
    if (!side->quantities) return false;
    side->nfields=nfields;
    return true;
}

static void jump_bindside(objectintegralelementref *side, objectmesh *mesh, grade g, elementid id, int nv, int *vid, integralref *iref) {
    double *lam=side->lambda;
    quantity *q=side->quantities;
    int nfields=side->nfields;
    _integral_bindelref(side, mesh, g, id, nv, vid, NULL, iref);
    side->lambda=lam;
    side->quantities=q;
    side->nfields=nfields;
}

static bool jump_ensuresidelambda(objectintegralelementref *side, int nv) {
    if (side->lambda) return true;
    side->lambda=MORPHO_MALLOC(sizeof(double)*nv);
    return side->lambda!=NULL;
}

static void jump_clearinterfaceref(objectjumpinterfaceref *iref, bool persistent) {
    if (persistent) return;
    if (iref->plus.lambda) { MORPHO_FREE(iref->plus.lambda); iref->plus.lambda=NULL; }
    if (iref->minus.lambda) { MORPHO_FREE(iref->minus.lambda); iref->minus.lambda=NULL; }
    integral_clearelref(&iref->iface);
    integral_clearelref(&iref->plus);
    integral_clearelref(&iref->minus);
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
    double *xplus[iref->plus.nv], *xminus[iref->minus.nv];

    if (!jump_getinterfacevertexpositions(iref->iface.mesh, iref->plus.nv, iref->plus.vid, xplus) ||
        !jump_getinterfacevertexpositions(iref->iface.mesh, iref->minus.nv, iref->minus.vid, xminus) ||
        !jump_parentlambda(iref->iface.mesh->dim, ref->parentgrade, xplus, posn, iref->plus.lambda) ||
        !jump_parentlambda(iref->iface.mesh->dim, ref->parentgrade, xminus, posn, iref->minus.lambda)) return false;

    iref->iface.posn=posn;
    iref->iface.qinterpolated=qinterp;

    for (int i=0; i<ref->integral.nfields; i++) {
        if (!jump_interpolatequantity(&iref->plus.quantities[i], ref->parentgrade, iref->plus.lambda, &qinterp[i])) return false;
    }

    return true;
}

static bool jump_callintegrand(objectjumpinterfaceref *iref, double *posn, double *out) {
    jumpref *ref=iref->jref;
    value qinterp[ref->integral.nfields+1], args[ref->integral.nfields+1], outval=MORPHO_NIL;
    objectmatrix mposn = MORPHO_STATICMATRIX(posn, iref->iface.mesh->dim, 1);

    if (!jump_preparepointdata(iref, posn, qinterp)) return false;

    args[0]=MORPHO_OBJECT(&mposn);
    for (int i=0; i<ref->integral.nfields; i++) args[i+1]=qinterp[i];

    if (!morpho_call(iref->v, ref->integral.integrand, ref->integral.nfields+1, args, &outval)) return false;
    return morpho_valuetofloat(outval, out);
}

static bool jump_integrandfn(unsigned int dim, double *t, double *x, unsigned int nquantity, value *quantity, void *ref, unsigned int nout, double *fout) {
    objectjumpinterfaceref *iref = (objectjumpinterfaceref *) ref;
    if (nout!=1) return false;
    iref->iface.lambda=t;
    iref->iface.posn=x;
    return jump_callintegrand(iref, x, fout);
}

static bool jump_evaluatesidegradient(objectjumpinterfaceref *iref, int ifld, bool plus, double *grad) {
    objectfield *fld = MORPHO_GETFIELD(iref->jref->integral.fields[ifld]);
    objectintegralelementref *side = (plus ? &iref->plus : &iref->minus);
    int nv = side->nv;
    int *vid = side->vid;
    double *lambda = side->lambda;
    int dim = iref->iface.mesh->dim;
    grade g = iref->jref->parentgrade;

    if (!MORPHO_ISFESPACE(fld->fnspc) || !MORPHO_ISNIL(fld->prototype)) return false;

    fespace *disc = MORPHO_GETFESPACE(fld->fnspc)->fespace;
    if (!FESPACE_HASGRADIENT(disc)) return false;

    double *x[nv];
    if (!jump_getinterfacevertexpositions(iref->iface.mesh, nv, vid, x)) return false;

    side->vertexposn=x;
    if (!(side->flags & ELREF_HASINVJ)) {
        if (!integral_ensurematrix(&side->invj, g, dim) ||
            !integral_prepareinvjacobian(dim, g, x, side->invj)) return false;
        side->flags |= ELREF_HASINVJ;
    }

    int nnodes = disc->nnodes;
    double gdata[nnodes*g];
    double fdata[nnodes*dim];
    objectmatrix gmat = MORPHO_STATICMATRIX(gdata, nnodes, g);
    objectmatrix fmat = MORPHO_STATICMATRIX(fdata, nnodes, dim);

    fespace_gradient(disc, lambda, &gmat);
    if (matrix_mul(&gmat, side->invj, &fmat)!=LINALGERR_OK) return false;

    for (int i=0; i<dim; i++) {
        value sum=MORPHO_FLOAT(0.0);
        if (!integrator_sumquantityweighted(nnodes, fmat.elements+i*nnodes, side->quantities[ifld].vals, &sum)) return false;
        if (!morpho_valuetofloat(sum, &grad[i])) return false;
    }

    return true;
}

static bool jump_preparenormal(vm *v, objectjumpinterfaceref *iref) {
    int dim=iref->iface.mesh->dim;
    double pluscentroid[dim], minuscentroid[dim], d[dim];

    if (!jump_getelementcentroid(iref->iface.mesh, iref->plus.nv, iref->plus.vid, pluscentroid)) return false;
    if (!jump_getelementcentroid(iref->iface.mesh, iref->minus.nv, iref->minus.vid, minuscentroid)) return false;

    functional_vecsub(dim, minuscentroid, pluscentroid, d);

    objectmatrix *mnormal=integral_ensurematrix(&iref->iface.normal, dim, 1);
    if (!mnormal) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);

    for (int i=0; i<dim; i++) mnormal->elements[i]=0.0;

    if (iref->iface.g==0) {
        for (int i=0; i<dim; i++) mnormal->elements[i]=d[i];
    } else if (iref->iface.g==1) {
        double t[dim], n[dim];

        functional_vecsub(dim, iref->iface.vertexposn[1], iref->iface.vertexposn[0], t);
        double tnorm=functional_vecnorm(dim, t);
        if (tnorm<MORPHO_EPS) return false;
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

        if (nnorm<MORPHO_EPS) return false;
        functional_vecscale(dim, 1.0/nnorm, n, mnormal->elements);
    } else if (iref->iface.g==2) {
        if (dim!=3) return false;

        double s0[3], s1[3];
        functional_vecsub(3, iref->iface.vertexposn[1], iref->iface.vertexposn[0], s0);
        functional_vecsub(3, iref->iface.vertexposn[2], iref->iface.vertexposn[1], s1);
        functional_veccross(s0, s1, mnormal->elements);
    } else {
        return false;
    }

    double nnorm=functional_vecnorm(dim, mnormal->elements);
    if (nnorm<MORPHO_EPS) return false;

    if (functional_vecdot(dim, mnormal->elements, d)<0.0) {
        functional_vecscale(dim, -1.0, mnormal->elements, mnormal->elements);
    }

    nnorm=functional_vecnorm(dim, mnormal->elements);
    if (nnorm<MORPHO_EPS) return false;
    functional_vecscale(dim, 1.0/nnorm, mnormal->elements, mnormal->elements);

    iref->iface.flags |= ELREF_HASNORMAL;
    return true;
}

static bool jump_preparegeometry(vm *v, objectjumpinterfaceref *iref, double **vertexposn) {
    iref->iface.vertexposn=vertexposn;
    if (iref->iface.g==0) iref->iface.elementsize=1.0;
    else if (!functional_elementsize(v, iref->iface.mesh, iref->iface.g, iref->iface.id, iref->iface.nv, iref->iface.vid, &iref->iface.elementsize)) return false;

    if (iref->iface.g>0 && iref->iface.elementsize<MORPHO_EPS) return true;

    return jump_preparenormal(v, iref);
}

static bool jump_prepareinterfaceref(vm *v, objectmesh *mesh, jumpref *ref, elementid id, int nv, int *vid, double **vertexposn, int *parents, objectjumpinterfaceref *iref) {
    int plusnv=0, minusnv=0;
    int *plusvid=NULL, *minusvid=NULL;
    bool persistent=(iref->iface.flags & ELREF_PERSISTENT);

    iref->v=v;
    iref->jref=ref;

    jump_orderparents(parents, &iref->plus.id, &iref->minus.id);

    if (!mesh_getconnectivity(ref->parentvertices, iref->plus.id, &plusnv, &plusvid)) return false;
    if (!mesh_getconnectivity(ref->parentvertices, iref->minus.id, &minusnv, &minusvid)) return false;

    _integral_bindelref(&iref->iface, mesh, ref->interfacegrade, id, nv, vid, vertexposn, &ref->integral);
    jump_bindside(&iref->plus, mesh, ref->parentgrade, iref->plus.id, plusnv, plusvid, &ref->integral);
    jump_bindside(&iref->minus, mesh, ref->parentgrade, iref->minus.id, minusnv, minusvid, &ref->integral);

    if (!jump_ensuresidelambda(&iref->plus, plusnv) || !jump_ensuresidelambda(&iref->minus, minusnv)) {
        jump_clearinterfaceref(iref, persistent);
        MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    }

    if (!jump_ensuresidequantities(&iref->plus, ref->integral.nfields) ||
        !jump_ensuresidequantities(&iref->minus, ref->integral.nfields)) {
        jump_clearinterfaceref(iref, persistent);
        return false;
    }

    if (ref->integral.nfields>0) {
        if (!integral_preparequantities(&ref->integral, plusnv, plusvid, iref->plus.quantities) ||
            !integral_preparequantities(&ref->integral, minusnv, minusvid, iref->minus.quantities)) {
            jump_clearinterfaceref(iref, persistent);
            return false;
        }
    }

    if (!jump_preparegeometry(v, iref, vertexposn)) {
        jump_clearinterfaceref(iref, persistent);
        return false;
    }

    return true;
}

static void jump_initstackref(objectjumpinterfaceref *iref) {
    memset(iref, 0, sizeof(*iref));
    object_init((object *) iref, OBJECT_JUMPINTERFACEREF);
    _integral_initelref(&iref->plus);
    _integral_initelref(&iref->minus);
}

/** Basic Jump scan over codimension-1 entities.
    This currently only identifies interior interfaces by checking that they
    have exactly two adjacent parent elements. */
static bool jump_scan_integrand(vm *v, objectmesh *mesh, elementid id, int nv, int *vid, void *r, double *out) {
    jumpref *ref = (jumpref *) r;
    int nparents=0, *parents=NULL;
    double *x[nv];
    objectjumpinterfaceref stackiref;
    objectjumpinterfaceref *iref;
    bool persistent=false;

    if (!jump_getadjacentparents(ref, id, &nparents, &parents)) return false;

    /* Boundary interfaces or malformed topology are ignored for now. */
    if (nparents!=2) { *out=0.0; return true; }

    if (!jump_getinterfacevertexpositions(mesh, nv, vid, x)) return false;

    iref=jump_getinterfaceref(v);
    if (iref && (iref->iface.flags & ELREF_PERSISTENT)) {
        persistent=true;
    } else {
        jump_initstackref(&stackiref);
        iref=&stackiref;
    }

    if (!jump_prepareinterfaceref(v, mesh, ref, id, nv, vid, x, parents, iref)) return false;

    if (iref->iface.g>0 && iref->iface.elementsize<MORPHO_EPS) {
        *out=0.0;
        jump_clearinterfaceref(iref, persistent);
        return true;
    }

    vm_settlvar(v, jumpinterfacehandle, MORPHO_OBJECT(iref));

    if (ref->strategy==JUMP_STRATEGY_CENTROID_MODE || ref->interfacegrade==0) {
        double posn[mesh->dim];
        jump_centroid(mesh->dim, nv, x, posn);
        if (!jump_callintegrand(iref, posn, out)) {
            if (!persistent) vm_settlvar(v, jumpinterfacehandle, MORPHO_NIL);
            jump_clearinterfaceref(iref, persistent);
            return false;
        }
        *out *= iref->iface.elementsize;
    } else if (ref->strategy==JUMP_STRATEGY_QUADRATURE_MODE) {
        bool success;
        if (iref->iface.flags & ELREF_CONFIGURED) {
            success=integrator_integrate(&iref->iface.integ, jump_integrandfn, mesh->dim, x, 0, NULL, iref, 1, out);
        } else {
            double err=0.0;
            success=integrate(jump_integrandfn, MORPHO_GETDICTIONARY(ref->integral.method), morpho_geterror(v), mesh->dim, ref->interfacegrade, x, 0, NULL, iref, out, &err);
        }
        if (!success) {
            if (!persistent) vm_settlvar(v, jumpinterfacehandle, MORPHO_NIL);
            jump_clearinterfaceref(iref, persistent);
            return false;
        }
        *out *= iref->iface.elementsize;
    } else {
        if (!persistent) vm_settlvar(v, jumpinterfacehandle, MORPHO_NIL);
        jump_clearinterfaceref(iref, persistent);
        MORPHO_FAIL(v, JUMP_UNIMPL);
    }

    if (!persistent) vm_settlvar(v, jumpinterfacehandle, MORPHO_NIL);
    jump_clearinterfaceref(iref, persistent);
    return true;
}


static bool jump_mapfieldgradient(vm *v, functional_mapinfo *info, value *out) {
    jumpref *ref = (jumpref *) info->ref;
    return functional_mapjumpnumericalfieldgradient(v, info, ref->parentvertices, ref, out);
}

static void jump_freeheapref(objectjumpinterfaceref *iref) {
    if (!iref) return;
    jump_clearinterfaceref(iref, false);
    object_free((object *) iref);
}

static bool jump_taskstart(vm *v, functional_mapinfo *info) {
    jumpref *ref=(jumpref *) info->ref;
    objectjumpinterfaceref *jiref=NULL;
    
    if (integral_contextactive(v)) MORPHO_FAIL(v, INTEGRAL_NESTED);
    
    jiref=(objectjumpinterfaceref *) object_new(sizeof(objectjumpinterfaceref), OBJECT_JUMPINTERFACEREF);
    if (!jiref) MORPHO_FAIL(v, ERROR_ALLOCATIONFAILED);
    
    memset(jiref, 0, sizeof(*jiref));
    object_init((object *) jiref, OBJECT_JUMPINTERFACEREF);
    _integral_initelref(&jiref->plus);
    _integral_initelref(&jiref->minus);
    
    jiref->jref=ref;
    jiref->v=v;
    jiref->iface.flags |= ELREF_PERSISTENT | ELREF_HASINTEG;
    integrator_init(&jiref->iface.integ);
    
    if (ref->strategy==JUMP_STRATEGY_QUADRATURE_MODE && ref->interfacegrade>0 &&
        MORPHO_ISDICTIONARY(ref->integral.method)) {
        if (!integrator_configurewithdictionary(&jiref->iface.integ, morpho_geterror(v), ref->interfacegrade, MORPHO_GETDICTIONARY(ref->integral.method))) goto jump_taskstart_cleanup;
        jiref->iface.flags |= ELREF_CONFIGURED;
    }
    
    if (!jump_ensuresidequantities(&jiref->plus, ref->integral.nfields) ||
        !jump_ensuresidequantities(&jiref->minus, ref->integral.nfields)) goto jump_taskstart_cleanup;
    
    int lamn=ref->parentgrade+1;
    jiref->plus.lambda=MORPHO_MALLOC(sizeof(double)*lamn);
    jiref->minus.lambda=MORPHO_MALLOC(sizeof(double)*lamn);
    if (!jiref->plus.lambda || !jiref->minus.lambda) goto jump_taskstart_cleanup;
    
    vm_settlvar(v, jumpinterfacehandle, MORPHO_OBJECT(jiref));
    return true;
    
jump_taskstart_cleanup:
    if (!morpho_checkerror(morpho_geterror(v))) morpho_runtimeerror(v, ERROR_ALLOCATIONFAILED);
    jump_freeheapref(jiref);
    return false;
}

static void jump_taskend(vm *v, functional_mapinfo *info) {
    objectjumpinterfaceref *iref=jump_getinterfaceref(v);
    if (!iref || !(iref->iface.flags & ELREF_PERSISTENT)) return;
    vm_settlvar(v, jumpinterfacehandle, MORPHO_NIL);
    jump_freeheapref(iref);
}

/** Jump bindref: prepare already raises IntgrlArgs / FnctlELNtFnd / FnctlArgs.
 * Map grade is the interface grade from topology, not the caller's Int. */
static bool _Jump_bindref(vm *v, objectinstance *self, functional_mapinfo *info, jumpref *ref) {
    if (integral_contextactive(v)) MORPHO_FAIL(v, INTEGRAL_NESTED);
    if (!jump_prepareref(v, self, info->mesh, 0, info->sel, ref)) return false;
    info->g = ref->interfacegrade;
    info->ref = ref;
    info->integrand = jump_scan_integrand;
    info->start = jump_startfn;
    info->taskstart = jump_taskstart;
    info->taskend = jump_taskend;
    return true;
}

FUNCTIONAL_MD_REF_INTEGRAND(Jump, jumpref, ref.interfacegrade)
FUNCTIONAL_MD_REF_TOTAL(Jump, jumpref, ref.interfacegrade)
FUNCTIONAL_MD_REF_NUMERICALGRADIENT(Jump, jumpref, ref.interfacegrade, jump_dependencies, SYMMETRY_NONE)
FUNCTIONAL_MD_REF_FIELDGRADIENT_MAP(Jump, jumpref, ref.interfacegrade, jump_mapfieldgradient, jump_cloneref, jump_freeref)

static value integral_jumpdnfn(vm *v, int nargs, value *args) {
    if (!integral_checkfastpath(v, INTEGRAL_USES_JUMPDN, JUMPDN_FUNCTION)) return MORPHO_NIL;
    objectjumpinterfaceref *iref = jump_getinterfaceref(v);
    if (!iref) MORPHO_RAISEVARGS(v, INTEGRAL_SPCLFN, JUMPDN_FUNCTION);

    value q = MORPHO_GETARG(args, 0);
    int ifld, xfld=-1;

    for (ifld=0; ifld<iref->jref->integral.nfields; ifld++) {
        if (MORPHO_ISFIELD(q) && MORPHO_ISSAME(iref->jref->integral.originalfields[ifld], q)) break;
        else if (iref->iface.qinterpolated && MORPHO_ISSAME(iref->iface.qinterpolated[ifld], q)) {
            if (xfld>=0) MORPHO_RAISE(v, INTEGRAL_FLD);
            xfld=ifld;
        }
    }
    if (xfld>=0) ifld=xfld;

    if (ifld>=iref->jref->integral.nfields) MORPHO_RAISE(v, INTEGRAL_FLD);

    double gradplus[iref->iface.mesh->dim], gradminus[iref->iface.mesh->dim];
    if (!jump_evaluatesidegradient(iref, ifld, true, gradplus) ||
        !jump_evaluatesidegradient(iref, ifld, false, gradminus)) MORPHO_RAISE(v, JUMP_UNIMPL);

    double jp = functional_vecdot(iref->iface.mesh->dim, gradplus, iref->iface.normal->elements);
    double jm = functional_vecdot(iref->iface.mesh->dim, gradminus, iref->iface.normal->elements);
    return MORPHO_FLOAT(jp-jm);
}

MORPHO_BEGINCLASS(Jump)
MORPHO_METHOD_SIGNATURE(MORPHO_INITIALIZER_METHOD, "(...)", Jump_init, INTEGRAL_INITFLAGS),
FUNCTIONAL_MD_INTEGRAND_METHODS_FLAGS(Jump, INTEGRAL_MAPFLAGS, INTEGRAL_ELEMFLAGS),
FUNCTIONAL_MD_TOTAL_METHODS_FLAGS(Jump, INTEGRAL_TOTALFLAGS),
FUNCTIONAL_MD_GRADIENT_METHODS_FLAGS(Jump, INTEGRAL_MAPFLAGS),
FUNCTIONAL_MD_FIELDGRADIENT_METHODS_FLAGS(Jump, INTEGRAL_MAPFLAGS)
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
    functional_optimizeproperty=builtin_internsymbolascstring(INTEGRAL_OPTIMIZE_PROPERTY);

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
    _addspecial(GRAD_FUNCTION, integral_gradfn, INTEGRAL_USES_GRAD);
    _addspecial(HESS_FUNCTION, integral_hessfn, INTEGRAL_USES_HESS);
    _addspecial(TANGENT_FUNCTION, integral_tangent, INTEGRAL_USES_TANGENT);
    _addspecial(NORMAL_FUNCTION, integral_normal, INTEGRAL_USES_NORMAL);
    _addspecial(JACOBIAN_FUNCTION, integral_jacobian, INTEGRAL_USES_JACOBIAN);
    _addspecial(INVJACOBIAN_FUNCTION, integral_invjacobian, INTEGRAL_USES_INVJ);
    _addspecial(CGTENSOR_FUNCTION, integral_cgfn, INTEGRAL_USES_CG);
    _addspecial(JUMPDN_FUNCTION, integral_jumpdnfn, INTEGRAL_USES_JUMPDN);

    morpho_defineerror(VOLUMEENCLOSED_ZERO, ERROR_HALT, VOLUMEENCLOSED_ZERO_MSG);
    morpho_defineerror(FUNC_ELNTFND, ERROR_HALT, FUNC_ELNTFND_MSG);
    morpho_defineerror(FUNC_FESPACE, ERROR_HALT, FUNC_FESPACE_MSG);
    morpho_defineerror(FUNC_NOFESPACE, ERROR_HALT, FUNC_NOFESPACE_MSG);

    morpho_defineerror(SCALARPOTENTIAL_FNCLLBL, ERROR_HALT, SCALARPOTENTIAL_FNCLLBL_MSG);

    morpho_defineerror(HYDROGEL_FLDGRD, ERROR_HALT, HYDROGEL_FLDGRD_MSG);
    morpho_defineerror(HYDROGEL_ZEEROREFELEMENT, ERROR_WARNING, HYDROGEL_ZEEROREFELEMENT_MSG);
    morpho_defineerror(HYDROGEL_BNDS, ERROR_WARNING, HYDROGEL_BNDS_MSG);

    morpho_defineerror(FUNCTIONAL_ARGS, ERROR_HALT, FUNCTIONAL_ARGS_MSG);
    
    morpho_defineerror(INTEGRAL_ARGS, ERROR_HALT, INTEGRAL_ARGS_MSG);
    morpho_defineerror(INTEGRAL_FLD, ERROR_HALT, INTEGRAL_FLD_MSG);
    morpho_defineerror(INTEGRAL_SPCLFN, ERROR_HALT, INTEGRAL_SPCLFN_MSG);
    morpho_defineerror(INTEGRAL_DFFEVL, ERROR_HALT, INTEGRAL_DFFEVL_MSG);
    morpho_defineerror(INTEGRAL_NESTED, ERROR_HALT, INTEGRAL_NESTED_MSG);
    morpho_defineerror(INTEGRAL_FASTPATH, ERROR_HALT, INTEGRAL_FASTPATH_MSG);
    morpho_defineerror(JUMP_UNIMPL, ERROR_HALT, JUMP_UNIMPL_MSG);
    
    functional_poolinitialized = false;
    
    objectintegralelementreftype=object_addtype(&objectintegralelementrefdefn);
    objectjumpinterfacereftype=object_addtype(&objectjumpinterfacerefdefn);
    elementhandle=vm_addtlvar();
    jumpinterfacehandle=vm_addtlvar();
    
    morpho_addfinalizefn(functional_finalize);
}

void functional_finalize(void) {
    if (functional_poolinitialized) threadpool_clear(&functional_pool);
}

#endif
