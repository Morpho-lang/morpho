/** @file objectpool.c
 *  @author T J Atherton
 *
 *  @brief Pool of Morpho objects
 */

#include "objectpool.h"
#include "memory.h"
#include "morpho.h"

void objectpool_init(objectpool *p, size_t size, objecttype type, object *parent) {
    p->size=size;
    p->nelements=0;
    p->pool=NULL;
    p->type=type;
    p->parent=parent;
    p->ready=false;
}

void objectpool_clear(objectpool *p) {
    if (p->pool) MORPHO_FREE(p->pool);
    p->pool=NULL;
    p->nelements=0;
    p->ready=false;
}

bool objectpool_isalloc(const objectpool *p) {
    return p && p->pool;
}

void *objectpool_get(const objectpool *p, size_t indx) {
    if (!p || !p->pool || indx>=p->nelements) return NULL;
    return (char *) p->pool + indx*p->size;
}

bool objectpool_alloc(objectpool *p, size_t nelements) {
    if (p->pool) return true;
    if (nelements==0) {
        p->nelements=0;
        return true;
    }
    if (p->size<sizeof(object)) return false;
    if (p->size && nelements > ((size_t) -1)/p->size) return false;

    void *mem=MORPHO_MALLOC(nelements*p->size);
    if (!mem) return false;

    p->pool=mem;
    p->nelements=nelements;

    for (size_t i=0; i<nelements; i++) {
        object *obj=(object *) ((char *) mem+i*p->size);
        object_init(obj, p->type);
        if (p->parent) morpho_bindtoparent(obj, p->parent);
    }

    return true;
}
