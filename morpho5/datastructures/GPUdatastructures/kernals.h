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

__global__ void transposeDiagonal(double *odata, double *idata, int width, int height) {
    __shared__ double tile[TILE_DIM][TILE_DIM+1];
    int blockIdx_x, blockIdx_y;
    // diagonal reordering
    if (width == height) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid%gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }
    int xIndex = blockIdx_x*TILE_DIM + threadIdx.x;
    int yIndex = blockIdx_y*TILE_DIM + threadIdx.y;
    int index_in = yIndex + xIndex*height;
    bool outOfBounds = yIndex<height && xIndex<width;
    if (outOfBounds) {
        for (int i=0; i<TILE_DIM && (i+blockIdx_x*TILE_DIM) < width; i+=BLOCK_COLS) {
            tile[threadIdx.y][threadIdx.x+i] = idata[index_in+i*height];
        }
    }
    xIndex = blockIdx_y*TILE_DIM + threadIdx.x;
    yIndex = blockIdx_x*TILE_DIM + threadIdx.y;
    int index_out = yIndex + (xIndex)*width;
    outOfBounds = xIndex<height && yIndex<width;
    __syncthreads();
    if (outOfBounds) {
        for (int i=0; i<TILE_DIM  && i+blockIdx_y*TILE_DIM < height; i+=BLOCK_COLS) {
            odata[index_out+i*width] = tile[threadIdx.x+i][threadIdx.y];
        }
    }
}
