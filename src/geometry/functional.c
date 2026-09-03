/** @file functional.c
 *  @author T J Atherton
 *
 *  @brief Functional map engine
 */

#include "build.h"
#ifdef MORPHO_INCLUDE_GEOMETRY

#include <math.h>
#include <string.h>

#include "functional.h"
#include "morpho.h"
#include "classes.h"
#include "common.h"

#include "threadpool.h"

#include "linalg.h"
#include "sparse.h"
#include "geometry.h"

#include "size.h"
#include "scalarpotential.h"
#include "elasticity.h"
#include "hydrogel.h"
#include "equielement.h"
#include "curvature.h"
#include "gradsq.h"
#include "normsq.h"
#include "nematic.h"
#include "integral.h"
#include "jump.h"

value functional_gradeproperty;
value functional_fieldproperty;

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
    return morpho_wrapandbindrecursive(v, MORPHO_GETOBJECT(out));
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
bool functional_countelements(objectmesh *mesh, grade g, int *n, objectsparse **s) {
    if (s) *s=NULL;
    *n=0;
    if (g==MESH_GRADE_VERTEX) {
        *n=mesh_nvertices(mesh);
        return true;
    }
    objectsparse *conn=mesh_getconnectivityelement(mesh, 0, g);
    if (!conn) return false;
    if (s) *s=conn;
    *n=mesh_nelements(conn);
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
int functional_ntasks(functional_mapinfo *info) {
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
    mesh_freezeconnectivity(info->mesh);
    
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
 * @param[out] out - the integrand value as a Float
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

/** Scalar CG0 Field on grade `g` (one value per element of that grade). */
static objectfield *functional_newintegrandfield(objectmesh *mesh, grade g) {
    if (!mesh) return NULL;
    int ngrades=mesh_maxgrade(mesh)+1;
    if (g<0 || g>=ngrades) return NULL;

    value fnspc=MORPHO_NIL;
    unsigned int dof[ngrades];
    unsigned int *shape=NULL;
    for (int i=0; i<ngrades; i++) dof[i]=(i==g ? 1 : 0);

    objectfespace *obj=fespace_newfromname(FESPACE_CG0, g);
    if (obj) fnspc=MORPHO_OBJECT(obj);
    else shape=dof;

    objectfield *new=object_newfield(mesh, MORPHO_NIL, fnspc, shape);
    if (!new) morpho_freeobject(fnspc);
    return new;
}

/** Write the integrand value into the output Field at this element */
bool functional_mapintegrandprocessfn(void *arg) {
    functional_task *task = (functional_task *) arg;
    objectfield *new = (objectfield *) task->out;
    value val = MORPHO_FLOAT(*(double *) task->result);
    return field_setelement(new, task->g, task->id, 0, val);
}

/** Map integrand function, storing the results in a scalar Field on info->g */
bool functional_mapintegrand(vm *v, functional_mapinfo *info, value *out) {
    int success=false;
    int ntask=functional_ntasks(info);
    functional_task task[ntask];
    functional_sumintermediate sums[ntask];
    
    varray_elementid imageids;
    varray_elementidinit(&imageids);
    
    objectfield *new = NULL;
    
    if (!functional_preparetasks(v, info, ntask, task, &imageids)) return false;
    
    new=functional_newintegrandfield(info->mesh, info->g);
    if (new) {
        for (int i=0; i<ntask; i++) {
            task[i].mapfn=(functional_mapfn *) info->integrand;
            task[i].processfn=functional_mapintegrandprocessfn;
            
            task[i].result=(void *) &sums[i].result;
            task[i].out=(void *) new;
        }
        success=functional_map(ntask, task);
    }
    
    if (!success && new) {
        morpho_freeobject(new->fnspc);
        object_free((object *) new);
        new=NULL;
    }
    
    *out = morpho_wrapandbindrecursive(v, (object *) new);
    functional_cleanuptasks(v, ntask, task, &imageids);
    return success;
}

/* ----------------------------
 * Shared gradient combiner
 * ---------------------------- */

/** Add inc into *p. Safe when several workers share an output. */
void functional_accum(double *p, double inc) {
    if (inc==0.0) return;
    MorphoAtomic_adddouble(p, inc);
}

/** Add alpha*b into column col. */
bool functional_addtocolumn(objectmatrix *a, MatrixIdx_t col, double alpha, double *b) {
    if (!a || col<0 || col>=a->ncols) return false;
    double *dest=a->elements+a->nvals*col*a->nrows;
    int n=a->nrows*a->nvals;
    for (int i=0; i<n; i++) MorphoAtomic_madddouble(&dest[i], alpha, b[i]);
    return true;
}

/** Add inc into element (row, col). */
bool functional_addtoelement(objectmatrix *a, MatrixIdx_t row, MatrixIdx_t col, double inc) {
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

/** Compute an analytic field gradient. A NULL fieldgrad yields a zero Field.
 * Maps over info->g only; multigrade Fields (CG2+) need a custom walker. */
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
bool functional_preparenumericalfieldgradientref(vm *v, functional_mapinfo *info, bool clone, functional_numericalfieldgradientref *tref, objectfield **fieldclone) {
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

void functional_clearnumericalfieldgradientref(functional_mapinfo *info, functional_numericalfieldgradientref *tref, objectfield *fieldclone) {
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
 * Fields
 * ********************************************************************** */

void functional_fespaceerror(vm *v, objectfield *field, grade g) {
    char *name = (field ? MORPHO_GETFESPACENAME(field->fnspc) : NULL);

    morpho_runtimeerror(v, FUNC_FESPACE,
                        name ? " with finite element space " : "",
                        name ? name : "",
                        (unsigned int) g);
}

bool functional_preparefespacefield(vm *v, objectfield *field, grade g) {
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

bool functional_preparefieldlist(vm *v, value *fields, int nfields, grade g) {
    for (int i=0; i<nfields; i++) {
        if (!MORPHO_ISFIELD(fields[i])) continue;

        objectfield *field = MORPHO_GETFIELD(fields[i]);
        if (!MORPHO_ISFESPACE(field->fnspc)) MORPHO_FAIL(v, FUNC_NOFESPACE);
        if (!functional_preparefespacefield(v, field, g)) return false;
    }

    return true;
}

bool fieldref_startfn(vm *v, functional_mapinfo *info) {
    fieldref *ref = (fieldref *) info->ref;
    return functional_preparefespacefield(v, ref->field, info->g);
}


void functional_initialize(void) {
    fddelta1 = pow(MORPHO_EPS, 1.0/3.0);
    fddelta2 = pow(MORPHO_EPS, 1.0/4.0);

    functional_gradeproperty=builtin_internsymbolascstring(FUNCTIONAL_GRADE_PROPERTY);
    functional_fieldproperty=builtin_internsymbolascstring(FUNCTIONAL_FIELD_PROPERTY);

    size_initialize();
    scalarpotential_initialize();
    elasticity_initialize();
    hydrogel_initialize();
    equielement_initialize();
    curvature_initialize();
    gradsq_initialize();
    normsq_initialize();
    integral_initialize();
    jump_initialize();
    nematic_initialize();

    morpho_defineerror(FUNC_ELNTFND, ERROR_HALT, FUNC_ELNTFND_MSG);
    morpho_defineerror(FUNC_FESPACE, ERROR_HALT, FUNC_FESPACE_MSG);
    morpho_defineerror(FUNC_NOFESPACE, ERROR_HALT, FUNC_NOFESPACE_MSG);
    morpho_defineerror(FUNCTIONAL_ARGS, ERROR_HALT, FUNCTIONAL_ARGS_MSG);

    functional_poolinitialized = false;

    morpho_addfinalizefn(functional_finalize);
}

void functional_finalize(void) {
    if (functional_poolinitialized) threadpool_clear(&functional_pool);
}

#endif
