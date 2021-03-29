/** @file matrix3d.h
 *  @author T J Atherton
 *
 *  @brief Minimal matrix math for 3d graphics; uses BLAS and LAPACK
 */

#ifndef matrix3d_h
#define matrix3d_h

#include <stdio.h>

typedef float mat4x4[16];
typedef float vec4[4];
typedef float mat3x3[9];
typedef float vec3[3];

void mat3d_vectornormalize(vec3 in, vec3 out);

void mat3d_identity4x4(mat4x4 out);
void mat3d_identity3x3(mat4x4 out);

void mat3d_mul4x4(mat4x4 a, mat4x4 b, mat4x4 out);
void mat3d_mul3x3(mat3x3 a, mat3x3 b, mat3x3 out);
void mat3d_addscale3x3(mat3x3 a, float alpha, mat3x3 b, mat3x3 out);

void mat3d_copy4x4(mat4x4 a, mat4x4 out);

void mat3d_invert4x4(mat4x4 a, mat4x4 out);

void mat3d_print3x3(mat3x3 in);
void mat3d_print4x4(mat4x4 in);

void mat3d_translate(mat4x4 in, vec3 vec, mat4x4 out);
void mat3d_scale(mat4x4 in, float scale, mat4x4 out);
void mat3d_rotate(mat4x4 in, vec3 axis, float angle, mat4x4 out);

void mat3d_ortho(mat4x4 in, mat4x4 out, float left, float right, float bottom, float top, float near, float far);
void mat3d_frustum(mat4x4 in, mat4x4 out, float left, float right, float bottom, float top, float near, float far);

#endif /* matrix3d_h */
