/**
 * @file kernals.h
 * @author Danny Goldstein 
 * @brief This file contains the macroized kernals for CUDA and openCL
 * to be included or complied respectivly
 * @version 0.1
 * @date 2022-04-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */

__global__ void ScalarAddition(double *matrix ,double scalar, double *out, unsigned int size){
    // a += b * lambda
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<size){
        *(out+i) = *(matrix + i) + scalar;
    }
}
