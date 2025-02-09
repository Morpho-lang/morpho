/** @file error.c
*  @author T J Atherton
*
*  @brief Morpho error data structure and handling
*/

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "error.h"
#include "strng.h"
#include "common.h"
#include "dictionary.h"

/* **********************************************************************
 * Global data
 * ********************************************************************** */

/** A table of errors */
static dictionary error_table;

/** A table of error strings. */
static varray_errordefinition error_messages;

/* **********************************************************************
 * Utility functions
 * ********************************************************************** */

DEFINE_VARRAY(errordefinition, errordefinition)

/** Prints to an error block
 *  @param err      Error struct to fill out
 *  @param cat      The category of error */
static void error_printf(error *err, errorcategory cat, char *file, int line, int posn, char *message, va_list args) {
    
    err->cat=cat;
    err->file=file;
    err->line=line;
    err->posn=posn; 
    
    /* Print the message with requested args */
    vsnprintf(err->msg, MORPHO_ERRORSTRINGSIZE, message, args);
}

/** Clears an error structure
 *  @param err      Error struct to fill out */
void error_clear(error *err) {
    err->cat=ERROR_NONE;
    err->id=NULL;
    err->file=NULL;
    err->line=ERROR_POSNUNIDENTIFIABLE; err->posn=ERROR_POSNUNIDENTIFIABLE;
}

/** Clears an error structure
 *  @param err      Error struct to fill out */
void error_init(error *err) {
    error_clear(err);
}

/** Gets an error definition given an errorid
 *  @param[in]  id     Error to retrieve
 *  @param[out] def   The error definition
 *  @returns true on success */
bool morpho_getdefinitionfromid(errorid id, errordefinition **def) {
    value result;
    int indx;
    bool success=false;
    value key = object_stringfromcstring(id, strlen(id));
    
    if (dictionary_get(&error_table, key, &result)) {
        indx=MORPHO_GETINTEGERVALUE(result);
        *def=&error_messages.data[indx];
        success=true;
    }
    
    object_free(MORPHO_GETOBJECT(key));
    
    return success;
}

/** @brief Writes an error message to an error structure
 *  @param err  The error structure
 *  @param id   The error id.
 *  @param line The line at which the error occurred, if identifiable.
 *  @param posn The position in the line at which the error occurred, if identifiable.
 *  @param args Additional parameters (the data for the printf commands in the message) */
void morpho_writeerrorwithidvalist(error *err, errorid id, char *file, int line, int posn, va_list args) {
    error_init(err);
    errordefinition *def;
    
    err->id=id;
    if (morpho_getdefinitionfromid(id, &def)) {
        /* Print the message with requested args */
        error_printf(err, def->cat, file, line, posn, def->msg, args);
    } else {
        UNREACHABLE("Undefined error generated.");
    }
}

/** @brief Writes an error message to an error structure
 *  @param err  The error structure
 *  @param id   The error id.
 *  @param file The file in which the error ocdured, if relevant.
 *  @param line The line at which the error occurred, if identifiable.
 *  @param posn The position in the line at which the error occurred, if identifiable. 
 *  @param ...  Additional parameters (the data for the printf commands in the message) */
void morpho_writeerrorwithid(error *err, errorid id, char *file, int line, int posn, ...) {
    va_list args;
    va_start(args, posn);
    morpho_writeerrorwithidvalist(err, id, file, line, posn, args);
    va_end(args);
}

/** @brief Writes an error message to an error structure without position information
 *  @param err  The error structure
 *  @param id   The error id.
 *  @param ...  Additional parameters (the data for the printf commands in the message) */
void error_writewithid(error *err, errorid id, ... ) {
    va_list args;
    va_start(args, id);
    morpho_writeerrorwithidvalist(err, id, NULL, ERROR_POSNUNIDENTIFIABLE, ERROR_POSNUNIDENTIFIABLE, args);
    va_end(args);
}

/** @brief Writes a user error to an error structure
 *  @param err  The error structure
 *  @param id   The error id.
 *  @param message Additional parameters (the data for the printf commands in the message) */
void morpho_writeusererror(error *err, errorid id, char *message) {
    err->line=ERROR_POSNUNIDENTIFIABLE;
    err->posn=ERROR_POSNUNIDENTIFIABLE;
    err->cat=ERROR_USER;
    err->id=id;
    size_t length = strlen(message);
    if (length>MORPHO_ERRORSTRINGSIZE-1) length = MORPHO_ERRORSTRINGSIZE-1;
    memcpy(err->msg, message, length);
    err->msg[length]='\0'; // Ensure null termination
}

/** Defines an error
 * @param id       Error struct to fill out
 * @param cat      The category of error
 * @param message  The message string*/
void morpho_defineerror(errorid id, errorcategory cat, char *message) {
    errordefinition new = {.cat = cat, .msg=morpho_strdup(message)};
    
    if (new.msg) {
        value key = object_stringfromcstring(id, strlen(id));
        int indx = varray_errordefinitionwrite(&error_messages, new);
        
        if (dictionary_get(&error_table, key, NULL)) {
            UNREACHABLE("Duplicate error.\n");
        }
        
        dictionary_insert(&error_table, key, MORPHO_INTEGER(indx));
    }
}

/** Gets the id of an error */
errorid morpho_geterrorid(error *err) {
    return err->id;
}

/** Tests if an error struct is showing error id
 * @returns true if the match succeeds and false otherwise */
bool morpho_matcherror(error *err, errorid id) {
    if (err->cat==ERROR_NONE) return false; 
    return (strcmp(err->id, id)==0);
}

/** Tests if an error block is showing an error */
bool morpho_checkerror(error *err) {
    return (err->cat!=ERROR_NONE);
}

/* **********************************************************************
* Unreachable code
* ********************************************************************** */

#ifdef MORPHO_DEBUG
void morpho_unreachable(const char *explanation) {
    fprintf(stderr, "Internal consistency error: Please contact developer. [Explanation: %s].\n", explanation);
    exit(BSD_EX_SOFTWARE);
}
#endif

/* **********************************************************************
* Initialization/Finalization
* ********************************************************************** */

/** Initializes the error handling system */
void error_initialize(void) {
    dictionary_init(&error_table);
    varray_errordefinitioninit(&error_messages);
    
    morpho_addfinalizefn(error_finalize);
}

/** Finalizes the error handling system */
void error_finalize(void) {
    dictionary_freecontents(&error_table, true, false);
    dictionary_clear(&error_table);
    for (unsigned int i=0; i<error_messages.count; i++) {
        MORPHO_FREE(error_messages.data[i].msg);
    }
    varray_errordefinitionclear(&error_messages);
}
