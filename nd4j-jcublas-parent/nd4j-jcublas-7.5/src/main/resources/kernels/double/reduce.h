extern "C"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "deeplearning4j.h"


//an op for the kernel
__device__ double op(double d1,double *extraParams);

//calculate an update of the reduce operation
__device__ double update(double old,double opOutput,double *extraParams);
//invoked when combining two kernels
__device__ double merge(double f1, double f2,double *extraParams);

//post process result (for things like means etc)
__device__ double postProcess(double reduction,int n,int xOffset,double *dx,int incx,double *extraParams,double *result);




/**

Perform a reduction
@param n the number of elements
@param xOffset the starting offset
@param dx the data to perform the reduction on
@param incx the increment on which to perform the reduction
@param extraParams extra parameters used for calculations
@param result where to store the result of the reduction
 */
__device__ void transform(int n, int xOffset,double *dx,int incx,double *extraParams,double *result,int i2,int blockSize) {
	extern volatile __shared__ double sdata[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;
    int nIsPow2 = n % 2 == 0;
	double reductionResult = extraParams[0];

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		reductionResult += dx[i * incx];

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i * incx + blockSize < n)
			reductionResult  = update(op(dx[i * incx  + blockSize],extraParams),reductionResult,extraParams);

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = reductionResult;
	__syncthreads();


	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = reductionResult = update(op(sdata[tid + 256],extraParams),reductionResult,extraParams);
	}

	__syncthreads();

	if ((blockSize >= 256) &&(tid < 128))
	{
		sdata[tid] = reductionResult = update(op(sdata[tid + 128],extraParams),reductionResult,extraParams);
	}

	__syncthreads();

	if ((blockSize >= 128) && (tid <  64))
	{
		sdata[tid] = reductionResult = update(op(sdata[tid +  64],extraParams),reductionResult,extraParams);
	}

	__syncthreads();

#if (__CUDA_ARCH__ >= 300 )
	if ( tid < 32 )
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64) 
		reductionResult = update(op(sdata[tid + 32],extraParams),reductionResult,extraParams);
		// Reduce final warp using shuffle
		for (int offset = warpSize / 2; offset > 0; offset /= 2)
		{
			reductionResult = update(op(__shfl_down(reductionResult, offset),extraParams),reductionResult,extraParams);
		}
	}
#else
	// fully unroll reduction within a single warp
	if ((blockSize >=  64) && (tid < 32))
	{
		sdata[tid] = reductionResult = update(op(sdata[tid + 32],extraParams),reductionResult,extraParams);
	}

	__syncthreads();

	if ((blockSize >=  32) && (tid < 16))
	{
		sdata[tid] = reductionResult = update(op(sdata[tid + 16],extraParams),reductionResult,extraParams);
	}

	__syncthreads();

	if ((blockSize >=  16) && (tid <  8))
	{
		sdata[tid] = reductionResult = update(op(sdata[tid + 8],extraParams),reductionResult,extraParams);
	}

	__syncthreads();

	if ((blockSize >=   8) && (tid <  4))
	{
		sdata[tid] = reductionResult = update(op(sdata[tid + 4],extraParams),reductionResult,extraParams);
	}

	__syncthreads();

	if ((blockSize >=   4) && (tid <  2))
	{
		sdata[tid] = reductionResult = update(op(sdata[tid + 2],extraParams),reductionResult,extraParams);
	}

	__syncthreads();

	if ((blockSize >=   2) && ( tid <  1))
	{
		sdata[tid] = reductionResult = update(op(sdata[tid + 1],extraParams),reductionResult,extraParams);
	}

	__syncthreads();
#endif

// write result for this block to global mem
if (tid == 0)
	result[blockIdx.x] = postProcess(reductionResult,n,xOffset,dx,incx,extraParams,result);

}
