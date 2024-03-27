/** @file format.h
 *  @author T J Atherton 
 *
 *  @brief Formatting of values
*/

#ifndef format_h
#define format_h

#define FORMAT_FLOATTYPES "efgEG"
#define FORMAT_INTTYPES "ioxX"

bool format_printtobuffer(value v, char *format, varray_char *out);

#endif /* format_h */
