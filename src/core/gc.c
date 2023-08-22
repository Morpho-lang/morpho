/** @file gc.c
 *  @author T J Atherton
 *
 *  @brief Morpho garbage collector
 */

#include "vm.h"
#include "gc.h"

extern vm *globalvm;

/* **********************************************************************
 * Gray list
 * ********************************************************************** */

/* Initialize the gray list */
void vm_graylistinit(graylist *g) {
    g->graycapacity=0;
    g->graycount=0;
    g->list=NULL;
}

/* Clear the gray list */
void vm_graylistclear(graylist *g) {
    if (g->list) free(g->list);
    vm_graylistinit(g);
}

/* Add an object to the gray list */
void vm_graylistadd(graylist *g, object *obj) {
    if (g->graycount+1>=g->graycapacity) {
        g->graycapacity*=2;
        if (g->graycapacity<8) g->graycapacity=8;
        g->list=realloc(g->list, g->graycapacity*sizeof(object *));
    }

    if (g->list) {
        g->list[g->graycount]=obj;
        g->graycount++;
    }
}

/* **********************************************************************
 * Garbage collector
 * ********************************************************************** */

/** Recalculates the size of bound objects to the VM */
size_t vm_gcrecalculatesize(vm *v) {
    size_t size = 0;
    for (object *ob=v->objects; ob!=NULL; ob=ob->next) {
        size+=object_size(ob);
    }
    return size;
}

/** Marks an object as reachable */
void vm_gcmarkobject(vm *v, object *obj) {
    if (!obj || obj->status!=OBJECT_ISUNMARKED) return;

#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    morpho_printf(v, "Marking %p ", obj);
    morpho_printf(v, MORPHO_OBJECT(obj));
    morpho_printf(v, "\n");
#endif
    obj->status=OBJECT_ISMARKED;

    vm_graylistadd(&v->gray, obj);
}

/** Marks a value as reachable */
void vm_gcmarkvalue(vm *v, value val) {
    if (MORPHO_ISOBJECT(val)) {
        vm_gcmarkobject(v, MORPHO_GETOBJECT(val));
    }
}

/** Marks all entries in a dictionary */
void vm_gcmarkdictionary(vm *v, dictionary *dict) {
    for (unsigned int i=0; i<dict->capacity; i++) {
        if (!MORPHO_ISNIL(dict->contents[i].key)) {
            vm_gcmarkvalue(v, dict->contents[i].key);
            vm_gcmarkvalue(v, dict->contents[i].val);
        }
    }
}

/** Marks all entries in an array */
void vm_gcmarkarray(vm *v, varray_value *array) {
    if (array) for (unsigned int i=0; i<array->count; i++) {
        vm_gcmarkvalue(v, array->data[i]);
    }
}

/** Public veneers */
void morpho_markobject(void *v, object *obj) {
    vm_gcmarkobject((vm *) v, obj);
}

void morpho_markvalue(void *v, value val) {
    vm_gcmarkvalue((vm *) v, val);
}

void morpho_markdictionary(void *v, dictionary *dict) {
    vm_gcmarkdictionary((vm *) v, dict);
}

void morpho_markvarrayvalue(void *v, varray_value *array) {
    vm_gcmarkarray((vm *) v, array);
}

/** Searches a vm for all reachable objects */
void vm_gcmarkroots(vm *v) {
    /** Mark anything on the stack */
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    morpho_printf(v, "> Stack.\n");
#endif
    value *stacktop = v->stack.data+v->fp->roffset+v->fp->function->nregs-1;
    
    /* Find the largest stack position currently in play */
    /*for (callframe *f=v->frame; f<v->fp; f++) {
        value *ftop = v->stack.data+f->roffset+f->function->nregs-1;
        if (ftop>stacktop) stacktop=ftop;
    }*/

    //debug_showstack(v);

    for (value *s=stacktop; s>=v->stack.data; s--) {
        if (MORPHO_ISOBJECT(*s)) vm_gcmarkvalue(v, *s);
    }

#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    morpho_printf(v, "> Globals.\n");
#endif
    for (unsigned int i=0; i<v->globals.count; i++) {
        vm_gcmarkvalue(v, v->globals.data[i]);
    }

    /** Mark closure objects in use */
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    morpho_printf(v, "> Closures.\n");
#endif
    for (callframe *f=v->frame; f && v->fp && f<=v->fp; f++) {
        if (f->closure) vm_gcmarkobject(v, (object *) f->closure);
    }

#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    morpho_printf(v, "> Open upvalues.\n");
#endif
    for (objectupvalue *u=v->openupvalues; u!=NULL; u=u->next) {
        vm_gcmarkobject(v, (object *) u);
    }

#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    morpho_printf(v, "> Thread local storage.\n");
#endif
    for (int i=0; i<v->tlvars.count; i++) {
        vm_gcmarkvalue(v, v->tlvars.data[i]);
    }
    
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    morpho_printf(v, "> End mark roots.\n");
#endif
}

void vm_gcmarkretainobject(vm *v, object *obj) {
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
    morpho_printf(v, "Searching object %p ", (void *) obj);
    morpho_printvalue(v, MORPHO_OBJECT(obj));
    morpho_printf(v, "\n");
#endif
    objecttypedefn *defn=object_getdefn(obj);
    if (defn->markfn) defn->markfn(obj, v);
}

/** Forces the GC to search an unmanaged object */
void morpho_searchunmanagedobject(void *v, object *obj) {
    vm_gcmarkretainobject((vm *) v, obj);
}

/** Trace all objects on the graylist */
void vm_gctrace(vm *v) {
    while (v->gray.graycount>0) {
        object *obj=v->gray.list[v->gray.graycount-1];
        v->gray.graycount--;
        vm_gcmarkretainobject(v, obj);
    }
}

/** Go through the VM's object list and free all unmarked objects */
void vm_gcsweep(vm *v) {
    object *prev=NULL;
    object *obj = v->objects;
    while (obj!=NULL) {
        if (obj->status==OBJECT_ISMARKED) {
            prev=obj;
            obj->status=OBJECT_ISUNMARKED; /* Clear for the next cycle */
            obj=obj->next;
        } else {
            object *unreached = obj;
            size_t size=object_size(obj);
#ifdef MORPHO_DEBUG_GCSIZETRACKING
            value xsize;
            if (dictionary_get(&sizecheck, MORPHO_OBJECT(unreached), &xsize)) {
                size_t isize = MORPHO_GETINTEGERVALUE(xsize);
                if (size!=isize) {
                    morpho_printvalue(MORPHO_OBJECT(unreached));
                    UNREACHABLE("Object doesn't match its declared size");
                }
            }
#endif

            v->bound-=size;

            /* Delink */
            obj=obj->next;
            if (prev!=NULL) {
                prev->next=obj;
            } else {
                v->objects=obj;
            }

#ifndef MORPHO_DEBUG_GCSIZETRACKING
            object_free(unreached);
#endif
        }
    }
}

/** Collects garbage */
void vm_collectgarbage(vm *v) {
#ifdef MORPHO_DEBUG_DISABLEGARBAGECOLLECTOR
    return;
#endif
    vm *vc = (v!=NULL ? v : globalvm);
    if (!vc) return;
    
    if (vc->parent) return; // Don't garbage collect in subkernels
    
#ifdef MORPHO_PROFILER
    vc->status=VM_INGC;
#endif

    if (vc && vc->bound>0) {
        size_t init=vc->bound;
#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
        morpho_printf(v, "--- begin garbage collection ---\n");
#endif
        vm_gcmarkroots(vc);
        vm_gctrace(vc);
        vm_gcsweep(vc);

        if (vc->bound>init) {
#ifdef MORPHO_DEBUG_GCSIZETRACKING
            morpho_printf(v, printf("GC collected %ld bytes (from %zu to %zu) next at %zu.\n", init-vc->bound, init, vc->bound, vc->bound*MORPHO_GCGROWTHFACTOR);
            UNREACHABLE("VM bound object size < 0");
#else
            // This catch has been put in to prevent the garbarge collector from completely seizing up.
            vc->bound=vm_gcrecalculatesize(v);
#endif
        }

        vc->nextgc=vc->bound*MORPHO_GCGROWTHFACTOR;

#ifdef MORPHO_DEBUG_LOGGARBAGECOLLECTOR
        morpho_printf(v, "--- end garbage collection ---\n");
        if (vc) morpho_printf(v, "    collected %ld bytes (from %zu to %zu) next at %zu.\n", init-vc->bound, init, vc->bound, vc->nextgc);
#endif
    }
    
#ifdef MORPHO_PROFILER
    vc->status=VM_RUNNING;
#endif
}
