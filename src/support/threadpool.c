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
void *threadpool_worker(void *ref) {
    threadpool *pool = (threadpool *) ref;
    task t = { .func = NULL, .arg = NULL };

    while (true) {
        /* Await a task */
        pthread_mutex_lock(&pool->lock_mutex);
        while (pool->queue.count == 0 && !pool->stop)
            pthread_cond_wait(&pool->work_available_cond, &pool->lock_mutex);

        if (pool->stop) break; /* Terminate if asked to do so */

        varray_taskpop(&pool->queue, &t); /* Get the task */
        pool->nprocessing++;
        pthread_mutex_unlock(&pool->lock_mutex);

        if (t.func) { (t.func) (t.arg); }; /* Perform the assigned task */

        pthread_mutex_lock(&pool->lock_mutex);
        pool->nprocessing--;
        if (!pool->stop && pool->nprocessing == 0 && pool->queue.count == 0)
            pthread_cond_signal(&pool->work_halted_cond);
        pthread_mutex_unlock(&pool->lock_mutex);
    }

    /* No need to lock here as lock was already obtained */
    pool->nthreads--;
    pthread_cond_signal(&pool->work_halted_cond);
    pthread_mutex_unlock(&pool->lock_mutex);

    return NULL;
}

/* Interface */

/** Initialize a threadpool with n worker threads. */
bool threadpool_init(threadpool *pool, int nworkers) {
    if (nworkers<1) return false;

    varray_taskinit(&pool->queue);

    pthread_mutex_init(&pool->lock_mutex, NULL);
    pthread_cond_init(&pool->work_available_cond, NULL);
    pthread_cond_init(&pool->work_halted_cond, NULL);

    pool->nthreads=nworkers;
    pool->stop=false;
    pool->nprocessing=0;

    for (int i=0; i<pool->nthreads; i++) {
        pthread_t thread;
        pthread_create(&thread, NULL, threadpool_worker, pool);
        pthread_detach(thread);
    }

    return true;
}

/** Clears a threadpool. */
void threadpool_clear(threadpool *pool) {
    pthread_mutex_lock(&pool->lock_mutex);
    varray_taskclear(&pool->queue); /* Erase any remaining tasks */
    pool->stop = true; /* Tell workers to stop */
    pthread_cond_broadcast(&pool->work_available_cond); /* Signal to workers */
    pthread_mutex_unlock(&pool->lock_mutex);

    threadpool_fence(pool); /* Await workers to terminate */

    pthread_mutex_destroy(&pool->lock_mutex);
    pthread_cond_destroy(&pool->work_available_cond);
    pthread_cond_destroy(&pool->work_halted_cond);
}

/** Adds a task to the threadpool */
bool threadpool_add_task(threadpool *pool, workfn func, void *arg) {
    bool success=true;
    pthread_mutex_lock(&pool->lock_mutex);

    task t = { .func = func, .arg=arg };
    if (!varray_taskadd(&pool->queue, &t, 1)) success=false; /* Add the task to the queue */

    pthread_cond_broadcast(&pool->work_available_cond); /* Signal there is work to be done */
    pthread_mutex_unlock(&pool->lock_mutex);
    return success;
}

/** Blocks until all tasks in the thread pool are complete */
void threadpool_fence(threadpool *pool) {
    pthread_mutex_lock(&pool->lock_mutex);

    while (true) {
        if ((!pool->stop && (pool->queue.count > 0 || pool->nprocessing>0)) || // If we are simply waiting for tasks to finish
            (pool->stop && pool->nthreads > 0)) { // Or if we have been told to stop
            pthread_cond_wait(&pool->work_halted_cond, &pool->lock_mutex); // Block until working_cond is set
        } else break;
    }

    pthread_mutex_unlock(&pool->lock_mutex);
}

/*
bool worker(void *arg) {
    int *val = arg;
    int  old = *val;

    *val += 1000;
    printf("tid=%p, old=%d, val=%d\n", pthread_self(), old, *val);

   // if (*val%2)
   //     usleep(100000);

    return false;
}

void threadpool_test(void) {
    threadpool pool;
    int num_items = 100;
    int vals[num_items];

    threadpool_init(&pool, 4);

    for (int i=0; i<num_items; i++) {
        vals[i] = i;
        threadpool_add_task(&pool, worker, vals+i);
    }

    threadpool_fence(&pool);

    for (int i=0; i<num_items; i++) {
        printf("%d\n", vals[i]);
    }

    threadpool_clear(&pool);
}
*/
