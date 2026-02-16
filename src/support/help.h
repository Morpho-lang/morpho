/** @file help.h
 *  @author T J Atherton and others (see below)
 *
 *  @brief Morpho help
*/

#ifndef help_h
#define help_h

#include <stddef.h>
#include "varray.h"

/** Maximum length of a help query string (used for lookup buffer). */
#define MORPHO_MAX_HELPQUERY_LENGTH 512

/* -------------------------------------------------------
 * Markdown AST: source-backed spans, hierarchical blocks
 * ------------------------------------------------------- */

/** Span into a stored source buffer (offset + length). */
typedef struct {
    size_t start;
    size_t length;
} md_span;

/** Block type. */
typedef enum {
    MD_BLOCK_HEADER,
    MD_BLOCK_PARAGRAPH,
    MD_BLOCK_CODE,
    MD_BLOCK_LIST,
    MD_BLOCK_LINK_DEF,
    MD_BLOCK_BLANK
} md_blocktype;

/** A single block: maps to a span in source; type-specific data; optional hierarchy. */
typedef struct {
    md_blocktype type;
    md_span span;
    union {
        struct { int level; md_span title; } header;
        struct { md_span label; md_span target; } link_def;
        /* paragraph, code, list: span covers content */
    } as;
    int parent;             /* block index of parent, or -1 if top-level */
    varray_int children;    /* indices of child blocks within same file */
} md_block;

DECLARE_VARRAY(md_block, md_block);

/** One source file: owned source text + varray of blocks (with spans into source). */
typedef struct {
    char *source;
    size_t sourcelen;
    varray_md_block blocks;
} md_file;

DECLARE_VARRAY(md_file, md_file);

/** Topic entry: header in the master list, sorted by name for O(log n) lookup. */
typedef struct {
    char *name;             /* lowercase, owned */
    int file_index;
    unsigned int block_index;
    int level;              /* header level 1–3 */
    int parent_topic;       /* topic index of parent, or -1 */
} md_topic;

DECLARE_VARRAY(md_topic, md_topic);

/** Initialize / clear a block (clears children). */
void md_block_clear(md_block *b);

/** Initialize / clear a file (frees source, clears blocks). */
void md_file_init(md_file *f);
void md_file_clear(md_file *f);

/** Topic list: compare by name for bsearch. */
int md_topic_compare(const void *a, const void *b);

/** Sort topic list by name (call after loading so lookup is O(log n)). */
void help_sorttopics(varray_md_topic *topics);

/** Find topic index by name (binary search). Returns -1 if not found. */
int help_findtopic(varray_md_topic *topics, const char *name);

/* -------------------------------------------------------
 * Help topic view (for flexible rendering)
 * ------------------------------------------------------- */

/** A resolved help topic: pointer to the topic entry, its file, and the slice of blocks that form its content (header + following blocks until the next same-level or higher header). Callers can use this to render raw MD, plain text, or terminal-styled output. */
typedef struct {
    const md_topic *topic;
    const md_file *file;
    const md_block *content_blocks;  /* first block of content (header) */
    unsigned int nblocks;
} help_topic;

/** Resolve a query to a help topic. Fills *out and returns true if found; otherwise returns false and *out is unchanged. */
bool morpho_helptopic(const char *query, help_topic *out);

/** Render a help topic as plain text into result (no markdown formatting). Returns true on success. */
bool help_topictotext(const help_topic *t, varray_char *result);

/* -------------------------------------------------------
 * Help API
 * ------------------------------------------------------- */

/** Look up help for query and write plain-text result. Returns true if topic found. (Convenience wrapper for morpho_helptopic + help_topictotext.) */
bool morpho_help(char *query, varray_char *result);

void help_initialize(void);
void help_finalize(void);

#endif /* help_h */
