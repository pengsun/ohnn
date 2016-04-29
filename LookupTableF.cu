#include "THC.h"
#include "common.h"
#include <cstdio>
#include <cusparse.h>


/// Helper
// TODO: make it a template fun and move to a separate file
__global__ void OHNN_CudaLookupTableF_updateOutput_kernel(
		float *inputInd, float *weight, int weightStride, int B, int M, int V, int C,
		float *output, int outputStride)
{
	int iFet = blockIdx.x * blockDim.x + threadIdx.x;
	int iWord = blockIdx.y * blockDim.y + threadIdx.y;
	if (iFet < C && iWord < B*M) {
		int iVocab = (int)(inputInd[iWord] - 1); // C zero base <- lua one base
		int nSrc = iVocab * weightStride + iFet;
		int nDst = iWord * outputStride + iFet;
		output[nDst] = weight[nSrc];
	}
	/*
	printf("blockId = (%d, %d); threadId = (%d, %d), iFet = %d, iWord = %d\n",
			blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, iFet, iWord);
	*/
}


/// Expose
extern "C"
void OHNN_CudaLookupTableF_updateOutput(
		THCState *state,
		// In
        THCudaTensor *input,
        THCudaTensor *weight,
        // Out
        THCudaTensor *output)
{
	DEBUG_PRINT(("in OHNN_CudaLookupTable2_updateOutput\n"));
	THAssert(THCudaTensor_checkGPU(state, 3, input, weight, output));
	// TODO: arg check?

	// input: B, M (,V)
	// weight: V, C
	// output: B, M, C
	int B = THCudaTensor_size(state, input, 0);
	int M = THCudaTensor_size(state, input, 1);
	int V = THCudaTensor_size(state, weight, 0);
	int C = THCudaTensor_size(state, weight, 1);
	DEBUG_PRINT(("B = %d, M = %d, V = %d, C = %d\n", B, M, V, C));

	// prepare data
	THCudaTensor_resize2d(state, output, B*M, C);
	int outputStride = output->stride[0];
	int weightStride = weight->stride[0];
	DEBUG_PRINT(("outputStride = %d\n", outputStride));
	DEBUG_PRINT(("weightStride = %d\n", weightStride));

	// update output
	cudaStream_t stream = THCState_getCurrentStream(state);
	dim3 grid(DIV_CEIL(C, 32), DIV_CEIL(B*M, 32));
	dim3 block(32, 32);
	OHNN_CudaLookupTableF_updateOutput_kernel<<<grid, block, 0, stream>>>(
			THCudaTensor_data(state, input),
			THCudaTensor_data(state, weight), weightStride,
			B, M, V, C,
			THCudaTensor_data(state, output), outputStride
	);

	// post process
	THCudaTensor_resize3d(state, output, B, M, C);

	// check error
	DEBUG_PRINT(("checking cuda error\n"));
	THCudaCheck(cudaGetLastError());
	DEBUG_PRINT(("done, no cuda error\n"));

	DEBUG_PRINT(("leaving OHNN_CudaLookupTable2_updateOutput\n"));
}
