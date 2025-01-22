/** @file threadpool.c
 *  @author T J Atherton
 *
 *  @brief Thread pool 
 */

#include <ctype.h>
#include "build.h"
#include "threadpool.h"

/* **********************************************************************
* Thread pools
* ********************************************************************** */

int threadpool_nthreads = MORPHO_DEFAULTTHREADNUMBER;

/** Sets the number of worker threads to use */
void morpho_setthreadnumber(int nthreads) {
    threadpool_nthreads = nthreads;
}

/** Returns the number of worker threads to use */
int morpho_threadnumber(void) {
    return threadpool_nthreads;
}

DEFINE_VARRAY(task, task);

/* Worker thread */
MorphoThreadFnReturnType threadpool_worker(void *ref) {
    threadpool *pool = (threadpool *) ref;
    task t = { .func = NULL, .arg = NULL };

    while (true) {
        /* Await a task */
        MorphoMutex_lock(&pool->lock_mutex);
        while (pool->queue.count == 0 && !pool->stop)
            MorphoCond_wait(&pool->work_available_cond, &pool->lock_mutex);

        if (pool->stop) break; /* Terminate if asked to do so */

        varray_taskpop(&pool->queue, &t); /* Get the task */
        pool->nprocessing++;
        MorphoMutex_unlock(&pool->lock_mutex);

        if (t.func) { (t.func) (t.arg); }; /* Perform the assigned task */

        MorphoMutex_lock(&pool->lock_mutex);
        pool->nprocessing--;
        if (!pool->stop && pool->nprocessing == 0 && pool->queue.count == 0)
            MorphoCond_signal(&pool->work_halted_cond);
        MorphoMutex_unlock(&pool->lock_mutex);
    }

    /* No need to lock here as lock was already obtained */
    pool->nthreads--;
    MorphoCond_signal(&pool->work_halted_cond);
    MorphoMutex_unlock(&pool->lock_mutex);

    return (MorphoThreadFnReturnType) NULL;
}

/* Interface */

/** Initialize a threadpool with n worker threads. */
bool threadpool_init(threadpool *pool, int nworkers) {
    if (nworkers<1) return false;

    varray_taskinit(&pool->queue);
    varray_MorphoThreadinit(&pool->threads);

    MorphoMutex_init(&pool->lock_mutex);
    MorphoCond_init(&pool->work_available_cond);
    MorphoCond_init(&pool->work_halted_cond);

    pool->nthreads=nworkers;
    pool->stop=false;
    pool->nprocessing=0;

    for (int i=0; i<pool->nthreads; i++) {
        MorphoThread thread;
        MorphoThread_create(&thread, threadpool_worker, pool);
        varray_MorphoThreadadd(&pool->threads, &thread, 1);
    }

    return true;
}

/** Clears a threadpool. */
void threadpool_clear(threadpool *pool) {
    MorphoMutex_lock(&pool->lock_mutex);
    varray_taskclear(&pool->queue); /* Erase any remaining tasks */
    pool->stop = true; /* Tell workers to stop */
    MorphoCond_broadcast(&pool->work_available_cond); /* Signal to workers to wake up */
    MorphoMutex_unlock(&pool->lock_mutex);

    for (int i=0; i<pool->threads.count; i++) MorphoThread_join(pool->threads.data[i]);

    MorphoMutex_clear(&pool->lock_mutex);
    MorphoCond_clear(&pool->work_available_cond);
    MorphoCond_clear(&pool->work_halted_cond);

    for (int i=0; i<pool->threads.count; i++) MorphoThread_clear(pool->threads.data[i]);
    
    varray_MorphoThreadclear(&pool->threads);
}

/** Adds a task to the threadpool */
bool threadpool_add_task(threadpool *pool, workfn func, void *arg) {
    bool success=true;
    MorphoMutex_lock(&pool->lock_mutex);

    task t = { .func = func, .arg=arg };
    if (!varray_taskadd(&pool->queue, &t, 1)) success=false; /* Add the task to the queue */

    MorphoCond_broadcast(&pool->work_available_cond); /* Signal there is work to be done */
    MorphoMutex_unlock(&pool->lock_mutex);
    return success;
}

/** Blocks until all tasks in the thread pool are complete */
void threadpool_fence(threadpool *pool) {
    MorphoMutex_lock(&pool->lock_mutex);

    while (true) {
        if ((!pool->stop && (pool->queue.count > 0 || pool->nprocessing>0)) || // If we are simply waiting for tasks to finish
            (pool->stop && pool->nthreads > 0)) { // Or if we have been told to stop
            MorphoCond_wait(&pool->work_halted_cond, &pool->lock_mutex); // Block until working_cond is set
        } else break;
    }

    MorphoMutex_unlock(&pool->lock_mutex);
}
