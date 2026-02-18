/** @file help.h
 *  @author T J Atherton and others (see below)
 *
 *  @brief Morpho help
*/

#ifndef help_h
#define help_h

#ifdef MORPHO_INCLUDE_HELP

#include <stddef.h>
#include "varray.h"
#include "value.h"

/** Maximum length of a help query string (used for lookup buffer). */
#define MORPHO_MAX_HELPQUERY_LENGTH 512

/** Maximum length of a single string for edit-distance (suggestion). */
#define MORPHO_HELP_EDITMAXLEN 128

/** Buffer size for "not found" hint message. */
#define MORPHO_HELP_HINTBUFSIZE 256

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
    MD_BLOCK_THEMATIC_BREAK,
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
    char *filename;  /* Owned filename/path for error reporting */
    bool promote_subtopics;  /* [toplevel]: # directive: include ## from this file in top-level topic list */
    varray_md_block blocks;
} md_file;

DECLARE_VARRAY(md_file, md_file);

/** Topic entry: header in the master list (load order). Hierarchy by parent_topic; lookup by name/alias via single dictionary (value = index or List of indices). */
typedef struct {
    value name;             /* Morpho string (lowercase); not owned */
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

/** Find one topic index by name or alias (first match if multiple). Returns -1 if not found. */
int help_findtopic(const char *name);

/** Find all topic indices with given name. Fills indices[] up to max, returns count. */
int help_findallbyname(const char *name, int indices[], int max);

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

/** Render a help topic as plain text into result (no markdown formatting). Returns true on success. */
bool help_topictotext(const help_topic *t, varray_char *result);

/** Append a help topic's raw markdown from source into result. Returns true on success. */
bool help_topicrawmd(const help_topic *t, varray_char *result);

/* -------------------------------------------------------
 * Markdown parser error codes
 * ------------------------------------------------------- */

#define MD_UNCLOSEDITALIC         "MDUnclItal"
#define MD_UNCLOSEDITALIC_MSG     "Unclosed italic (missing closing * or _)."

#define MD_UNCLOSEDINLINECODE     "MDUnclCode"
#define MD_UNCLOSEDINLINECODE_MSG "Unclosed inline code (missing closing `)."

#define MD_EXPECTLINEEND          "MDExpLnEnd"
#define MD_EXPECTLINEEND_MSG      "Expected end of line."

#define MD_EXPECTHEADERTEXT       "MDExpHdrTxt"
#define MD_EXPECTHEADERTEXT_MSG   "Expected header text after #."

#define MD_LINKEXPECTTEXT         "MDLnkExpTxt"
#define MD_LINKEXPECTTEXT_MSG     "Link definition expects label text after [."

#define MD_LINKEXPECTBRACKET      "MDLnkExpBr"
#define MD_LINKEXPECTBRACKET_MSG  "Link definition expects ] after label."

#define MD_LINKEXPECTCOLON        "MDLnkExpCol"
#define MD_LINKEXPECTCOLON_MSG    "Link definition expects : after ]."

#define MD_UNEXPECTEDTOKEN        "MDUnexpTok"
#define MD_UNEXPECTEDTOKEN_MSG    "Unexpected markdown token."

#define MORPHO_HELP_NOTFOUND      "Topic not found."

/* -------------------------------------------------------
 * Help API
 * ------------------------------------------------------- */

/** Resolve a query to a help topic. Fills *out and returns true if found; otherwise returns false and *out is unchanged. */
bool morpho_helpastopic(const char *query, help_topic *out);

/** Look up help for query and write plain-text result. Returns true if topic found. (Convenience wrapper for morpho_helpastopic + help_topictotext.) */
bool morpho_helpastext(const char *query, varray_char *result);

/** Look up help for query and write raw markdown into result. Returns true if topic found. (Convenience wrapper for morpho_helpastopic + help_topicrawmd.) */
bool morpho_helpasmd(const char *query, varray_char *result);

/** Fill a varray_value with top-level topic names (Morpho string values). Loads help on first use. Caller must initialize/clear the varray. */
void morpho_helptopics(varray_value *out);

/** Build a hint for a failed query */
void help_queryhint(const char *query, varray_char *result);

void help_initialize(void);
void help_finalize(void);

#endif

#endif /* help_h */
