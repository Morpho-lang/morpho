/** @file threadpool.h
 *  @author T J Atherton
 *
 *  @brief Thread pool 
 */

#ifndef threadpool_h
#define threadpool_h

#include <stdbool.h>
#include "varray.h"
#include "platform.h"

/* -----------------------------------------
 * Tasks
 * ----------------------------------------- */

/** A task is the basic unit of work that is allocated to the thread pool; it comprises a work function to perform the task, and a single argument. */
 
/** A workfn will be called by the threadpool once a thread is available.
    You must supply all relevant information for both input and output in a single structure passed as an opaque reference. */
typedef bool (* workfn) (void *arg);

typedef struct {
    workfn func;    // Function to call that performs the task
    void *arg;      // Opaque pointer passed as an argument to the work function
} task;

DECLARE_VARRAY(task, task);

/* -----------------------------------------
 * Thread pools
 * ----------------------------------------- */

typedef struct {
    MorphoMutex lock_mutex; /* Lock for access to threadpool structure. */
    MorphoCond work_available_cond; /* Signals that work is available. */
    MorphoCond work_halted_cond; /* Signals when no threads are processing. */
    int nprocessing; /* Number of threads actively processing work */
    int nthreads; /* Number of active threads. */
    bool stop; /* Indicates threads should terminate */

    varray_task queue; /* Queue of tasks lined up */
    varray_MorphoThread threads; /* Threads created by this pool */
} threadpool;

bool threadpool_init(threadpool *pool, int nworkers);
void threadpool_clear(threadpool *pool);
bool threadpool_add_task(threadpool *pool, workfn func, void *arg);
void threadpool_fence(threadpool *pool);
void threadpool_wait(threadpool *pool);

#endif /* threadpool_h */
