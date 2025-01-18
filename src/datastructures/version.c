/** @file version.c
 *  @author T J Atherton
 *
 *  @brief Version comparison
*/

#include <stdio.h>

#include "build.h"
#include "version.h"

/** @brief Initialize a version structure
 * @param[in] v - Version structure to intialize
 * @param[in] major }
 * @param[in] minor } version
 * @param[in] patch } */
void version_init(version *v, int major, int minor, int patch) {
    v->major=major;
    v->minor=minor;
    v->patch=patch;
}

/** @brief Compare two versions;  returns < 0 if a<b; 0 if a==b and >0 if a>b */
int version_cmp(version *a, version *b) {
    int d[3] = { a->major-b->major, a->minor-b->minor, a->patch-b->patch };
    for (int i=0; i<3; i++) if (d[i]!=0) return d[i];
    
    return 0;
}

/** @brief Test whether a version is compatible with minimum and maximum version constraints
 * @param[in] v - Version structure to test
 * @param[in] min - (optional) minimum version number
 * @param[in] max - (optional) maximum version number
 * @returns true if min <= v <= max */
bool version_check(version *v, version *min, version *max) {
    int l = (min ? version_cmp(min, v) : 0);
    int r = (max ? version_cmp(v, max) : 0);
        
    return (l>=0 && r<=0);
}

/** @brief Convert a version to a string
 * @param[in] v - Version structure to intialize
 * @param[in] n - size of output buffer
 * @param[out] str - output string - recommend creating with VERSION_MAXSTRINGLENGTH */
void version_tostring(version *v, size_t size, char *str) {
    snprintf(str, size, "%i.%i.%i", v->major, v->minor, v->patch);
}

/** @brief Sets a version to the current morpho version */
void morpho_version(version *v) {
    version_init(v, MORPHO_VERSION_MAJOR, MORPHO_VERSION_MINOR, MORPHO_VERSION_PATCH);
}
