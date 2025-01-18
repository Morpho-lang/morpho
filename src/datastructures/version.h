/** @file version.h
 *  @author T J Atherton
 *
 *  @brief Semantic version comparison
*/

#ifndef version_h
#define version_h

#include <stdbool.h>

#define VERSION_MAXSTRINGLENGTH 64

typedef struct {
    int major; // when you make incompatible API changes
    int minor; // when you add functionality in a backward compatible manner
    int patch; // when you make backward compatible bug fixes
} version;

#define VERSION_STATIC(maj, min, ptch) { .major = maj, .minor = min, .patch = ptch }

/** @brief Initialize a version structure
 * @param[in] v - Version structure to intialize
 * @param[in] major }
 * @param[in] minor } version
 * @param[in] patch } */
void version_init(version *v, int major, int minor, int patch);

/** @brief Compare two versions;  returns < 0 if a<b; 0 if a==b and >0 if a>b */
int version_cmp(version *a, version *b);

/** @brief Test whether a version is compatible with minimum and maximum version constraints
 * @param[in] v - Version structure to test
 * @param[in] min - (optional) minimum version number
 * @param[in] max - (optional) maximum version number
 * @returns true if min <= v <= max */
bool version_check(version *v, version *min, version *max);

/** @brief Convert a version to a string
 * @param[in] v - Version structure to intialize
 * @param[in] n - size of output buffer
 * @param[out] str - output string - recommend creating with VERSION_MAXSTRINGLENGTH */
void version_tostring(version *v, size_t n, char *str);

/** @brief Sets a version to the current morpho version */
void morpho_version(version *v);

#endif
