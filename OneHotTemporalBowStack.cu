#include "THC.h"
#include "common.h"
#include <cstdio>
#include <cusparse.h>


/// Helper
// TODO: make it a template fun and move to a separate file
__global__ void temporal_bow_statck_kernelV1(
		float *input, int B, int M, int p, int padVocabInd,
		float *output)
{
	// a Naive impl
	int nSrc = blockIdx.x * blockDim.x + threadIdx.x;
	if (nSrc >= B*M) return;

	int iBatch = nSrc / M;
	int iWord = nSrc % M;
	for (int i = 0; i < p; ++i) { // scan each word in the window
		int iCurWord = iWord - p/2 + i; // TODO: check?
		int curVocabInd = padVocabInd;
		int nDst = nSrc*p + i;

		if (iCurWord >= 0 && iCurWord < M) { // inside the iBatch sequence
			curVocabInd = (int)input[iBatch*M + iCurWord];
			if (curVocabInd != padVocabInd) { // a normal word
				// scan previous word in the same window, remove any duplicate
				// Warp divergence here?
				for (int k = 1; k <= i; ++k) {
					if (curVocabInd == output[nDst-k]) {
						curVocabInd = padVocabInd;
						break;
					}
				}
			}
		}

		output[nDst] = curVocabInd;
	}
}


/// Expose
extern "C"
void OHNN_CudaOneHotTemporalBowStack_updateOutput(
		THCState *state,
		// In
        THCudaTensor *input,
        double p,
        double padVocabInd,
        // Out
        THCudaTensor *output)
{
	DEBUG_PRINT(("in OHNN_CudaOneHotTemporalBowStack_updateOutput\n"));
	THAssert(THCudaTensor_checkGPU(state, 2, input, output));
	// TODO: arg check?

	// input: B, M (,V)
	// output: B, Mp, C
	int B = THCudaTensor_size(state, input, 0);
	int M = THCudaTensor_size(state, input, 1);
	int BM = B*M;
	int Mp = M*(int(p));
	DEBUG_PRINT(("B = %d, M = %d, p = %d, padVocabInd = %d\n", B, M, (int)p, (int)padVocabInd));

	// prepare data
	THCudaTensor_resize2d(state, output, B, Mp);

	// stack bow input
	cudaStream_t stream = THCState_getCurrentStream(state);
	dim3 grid(DIV_CEIL(BM, CUDA_NUM_THREADS));
	dim3 block(CUDA_NUM_THREADS);
	temporal_bow_statck_kernelV1<<<grid, block, 0, stream>>>(
			THCudaTensor_data(state, input), B, M, (int)p, (int)padVocabInd,
			THCudaTensor_data(state, output)
	);

	// check error
	DEBUG_PRINT(("checking cuda error\n"));
	THCudaCheck(cudaGetLastError());
	DEBUG_PRINT(("done, no cuda error\n"));

	DEBUG_PRINT(("leaving OHNN_CudaOneHotTemporalBowStack_updateOutput\n"));
}
