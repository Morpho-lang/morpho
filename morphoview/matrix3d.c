/** @file matrix3d.c
 *  @author T J Atherton
 *
 *  @brief Matrix math for 3d graphics
 */

#include "matrix3d.h"
#include <math.h>
#include <string.h> 

/** Use Apple's Accelerate library for LAPACK and BLAS */
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#include <lapacke.h>
#define USE_LAPACKE
#endif

#define EPS 1e-16

/** @brief Normalizes a vector
 * @param[in] in - input vector
 * @param[out] out - output vector. */
void mat3d_vectornormalize(vec3 in, vec3 out) {
    float norm=cblas_snrm2(3, in, 1);
    if (norm>EPS) norm = 1.0/norm;
    if (out!=in) cblas_scopy(3, in, 1, out, 1);
    cblas_sscal(3, norm, out, 1);
}

/** @brief Stores the identity matrix in out.
 * @param[out] out - output matrix. */
void mat3d_identity4x4(mat4x4 out) {
    static float ident[] = { 1.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 1.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 1.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 1.0f };
    cblas_scopy(16, ident, 1, out, 1);
}

/** @brief Stores the identity matrix in out.
 * @param[out] out - output matrix. */
void mat3d_identity3x3(mat4x4 out) {
    static float ident[] = { 1.0f, 0.0f, 0.0f,
                             0.0f, 1.0f, 0.0f,
                             0.0f, 0.0f, 1.0f };
    cblas_scopy(9, ident, 1, out, 1);
}

/** @brief Multiply out = a*b
 * @param[in] a input matrix
 * @param[in] b input matrix
 * @param[out] out filled with a*b
 * @warning: out must be distinct from a and b */
void mat3d_mul4x4(mat4x4 a, mat4x4 b, mat4x4 out) {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, 4, 4, 1.0, a, 4, b, 4, 0.0, out, 4);
}

/** @brief Multiply: out = a*b
 * @param[in] a input matrix
 * @param[in] b input matrix
 * @param[out] out filled with a*b
 * @warning: out must be distinct from a and b */
void mat3d_mul3x3(mat3x3 a, mat3x3 b, mat3x3 out) {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, a, 3, b, 3, 0.0, out, 3);
}

/** @brief Add with scale: out = a + alpha*b
 * @param[in] a input matrix
 * @param[in] b input matrix
 * @param[out] out filled with a + alpha*b  */
void mat3d_addscale3x3(mat3x3 a, float alpha, mat3x3 b, mat3x3 out) {
    if (a!=out) cblas_scopy(9, a, 1, out, 1);
    cblas_saxpy(9, alpha, b, 1, out, 1);
}

/** @brief Copy: out = a
 * @param[in] a input matrix
 * @param[out] out filled with a*b */
void mat3d_copy4x4(mat4x4 a, mat4x4 out) {
    cblas_scopy(16, a, 1, out, 1);
}

/** @brief Matrix inversion
 * @param[in] a input matrix
 * @param[out] out filled with inverse(a)  */
void mat3d_invert4x4(mat4x4 a, mat4x4 out) {
    int m = 4, n = 4;
    int piv[4];
    int info;
    /* Copy a into out */
    memcpy(out, a, sizeof(float)*16);
    /* Compute LU decomposition, storing result in place */
#ifdef USE_LAPACKE
    info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, m, n, out, m, piv);
#else
    sgetrf_(&m, &n, out, &m, piv, &info);
#endif
    
    if (!info) {
        /* Now compute inverse */
#ifdef USE_LAPACKE
        info=LAPACKE_sgetri(LAPACK_COL_MAJOR, n, out, n, piv);
#else
        float work[16];
        int lwork=16;
        sgetri_(&n, out, &n, piv, work, &lwork, &info);
#endif
    }
}

/** @brief Convert a 3x3 matrix to a 4x4 matrix
 * @param[in] in input matrix
 * @param[out] out filled with inverse(a)  */
void mat3d_lift(mat3x3 in, mat4x4 out) {
    mat4x4 new = { in[0], in[1], in[2], 0.0f, // Col major order!
                   in[3], in[4], in[5], 0.0f,
                   in[6], in[7], in[8], 0.0f,
                    0.0f,  0.0f,  0.0f, 1.0f };
    cblas_scopy(16, new, 1, out, 1);
}

/** @brief Print a 3x3 matrix */
void mat3d_print3x3(mat3x3 in) {
    for (unsigned int j=0; j<3; j++) { // row
        printf("[ ");
        for (unsigned int i=0; i<3; i++) { // column
            printf("%g ", in[i*3+j]);
        }
        printf("]\n");
    }
}

/** @brief Print a 3x3 matrix */
void mat3d_print4x4(mat4x4 in) {
    for (unsigned int j=0; j<4; j++) { // row
        printf("[ ");
        for (unsigned int i=0; i<4; i++) { // column
            printf("%g ", in[i*4+j]);
        }
        printf("]\n");
    }
}

/** @brief Translate by a vector
 * @param[in] in input matrix
 * @param[in] vec translation vector
 * @param[out] out on output, contains T*in where T is the translation matrix computed from vec */
void mat3d_translate(mat4x4 in, vec3 vec, mat4x4 out) {
    mat4x4 tr = { 1.0f, 0.0f, 0.0f, 0.0f, // Col major order!
                  0.0f, 1.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 1.0f, 0.0f,
                  vec[0], vec[1], vec[2], 1.0f };
    mat4x4 in2;
    if (in==out) mat3d_copy4x4(in, in2); /* Use a copy if in and out are the same matrix */
    if (in) mat3d_mul4x4(tr, (in==out ? in2 : in), out);
    else mat3d_copy4x4(tr, out);
}

/** @brief Scale by a factor
 * @param[in] in input matrix
 * @param[in] scale scale factor
 * @param[out] out on output, contains T*in where T is the translation matrix computed from vec */
void mat3d_scale(mat4x4 in, float scale, mat4x4 out) {
    mat4x4 tr = { scale, 0.0f, 0.0f, 0.0f, // Col major order!
                  0.0f, scale, 0.0f, 0.0f,
                  0.0f, 0.0f, scale, 0.0f,
                  0.0f, 0.0f,  0.0f, 1.0f };
    mat4x4 in2;
    if (in==out) mat3d_copy4x4(in, in2); /* Use a copy if in and out are the same matrix */
    if (in) mat3d_mul4x4(tr, (in==out ? in2 : in), out);
    else mat3d_copy4x4(tr, out);
}

/** @brief Rotate by angle around an axis
 * @param[in] in input matrix
 * @param[in] axis rotation axis
 * @param[in] angle rotation angle
 * @param[out] out on output, contains R*in where R is the translation matrix computed from vec */
void mat3d_rotate(mat4x4 in, vec3 axis, float angle, mat4x4 out) {
    vec3 u;
    mat3x3 rot;
    mat4x4 rot4;
    mat4x4 in2;
    if (in==out) mat3d_copy4x4(in, in2); /* Use a copy if in and out are the same matrix */
    
    /* Construct rotation matrix from Rodrigues formula */
    mat3d_vectornormalize(axis, u);
    mat3x3 w = { 0.0f,  u[2], -u[1],  // Col major order
                -u[2],  0.0f,  u[0],
                 u[1], -u[0],  0.0f };
    mat3x3 w2;
    
    /* Rodrigues formula: R = I + sin(a) W + 2 sin(a/2)^2 W^2 */
    mat3d_identity3x3(rot);
    mat3d_addscale3x3(rot, sin(angle), w, rot);
    mat3d_mul3x3(w, w, w2);
    float phi = sin(angle/2);
    mat3d_addscale3x3(rot, 2*phi*phi, w2, rot);
    
    /* Convert to 4x4 matrix */
    mat3d_lift(rot, rot4);

    /* Multiply the input matrix by this */
    if (in) mat3d_mul4x4(rot4, (in==out ? in2 : in), out);
    else mat3d_copy4x4(rot4, out);
}

/** @brief Orthographic projection matrix
 * @param[in] in input matrix
 * @param[in] left     } Bounds of the viewing area
 * @param[in] right   }
 * @param[in] bottom }
 * @param[in] top        }
 * @param[in] near      }
 * @param[in] far        }
 * @param[out] out on output, contains R*in where R is the translation matrix computed from vec */
void mat3d_ortho(mat4x4 in, mat4x4 out, float left, float right, float bottom, float top, float near, float far) {
    mat4x4 pr = { 2.0f/(right-left), 0.0f, 0.0f, 0.0f, // Col major order!
                  0.0f, 2.0f/(top-bottom), 0.0f, 0.0f,
                  0.0f, 0.0f, -2.0f/(far-near), 0.0f,
                  0.0f, 0.0f, 0.0f, 1.0f };
    mat4x4 in2;
    if (in==out) mat3d_copy4x4(in, in2); /* Use a copy if in and out are the same matrix */
    
    /* Multiply the input matrix by this */
    if (in) mat3d_mul4x4(pr, (in==out ? in2 : in), out);
    else mat3d_copy4x4(pr, out);
}

/** @brief Perspective projection matrix
 * @param[in] in input matrix
 * @param[in] left     } Bounds of the viewing area
 * @param[in] right   }
 * @param[in] bottom }
 * @param[in] top        }
 * @param[in] near      }
 * @param[in] far        }
 * @param[out] out on output, contains R*in where R is the translation matrix computed from vec */
void mat3d_frustum(mat4x4 in, mat4x4 out, float left, float right, float bottom, float top, float near, float far) {
    mat4x4 pr = { 2*near/(right-left), 0.0f, 0.0f, 0.0f, // Col major order!
                  0.0f, 2*near/(top-bottom), 0.0f, 0.0f,
                  (right+left)/(right-left), (top+bottom)/(top-bottom), -(far+near)/(far-near), -1.0f,
                  0.0f, 0.0f, -2*far*near/(far-near), 0.0f };
    mat4x4 in2;
    if (in==out) mat3d_copy4x4(in, in2); /* Use a copy if in and out are the same matrix */
    
    /* Multiply the input matrix by this */
    if (in) mat3d_mul4x4(pr, (in==out ? in2 : in), out);
    else mat3d_copy4x4(pr, out);
}
