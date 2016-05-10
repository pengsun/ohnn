#include "THC.h"
#include "common.h"
#include <cstdio>
#include <cusparse.h>


/// Helper
// TODO: make it a template fun and move to a separate file
__global__ void temporal_bow_statck_kernelV2(
		float *input, int B, int M, int p, int padBegLen, int padEndLen, int padIndValue,
		float *output)
{
	// a Naive impl
	int Md = (M + padBegLen + padEndLen) - p + 1;

	int nWinDst = blockIdx.x * blockDim.x + threadIdx.x;
	if (nWinDst >= B*Md) return;

	int iBat = nWinDst / Md;
	int iWin = nWinDst % Md;
	for (int i = 0; i < p; ++i) { // scan each word in the window
		int iCurWord = iWin - padBegLen + i;
		int curVocabInd = padIndValue;
		int nDst = nWinDst*p + i;

		if (iCurWord >= 0 && iCurWord < M) { // inside the source iBat sequence
			curVocabInd = (int)input[iBat*M + iCurWord];

			if (curVocabInd != padIndValue) { // encounter a normal word
				// scan previous word in the same window (Warp divergence here?)
				for (int k = 1; k <= i; ++k) {
					if (curVocabInd == output[nDst-k]) { // encounter a duplicate
						curVocabInd = padIndValue;
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
        double padBegLen,
        double padEndLen,
        double padIndValue,
        // Out
        THCudaTensor *output)
{
	DEBUG_PRINT(("in OHNN_CudaOneHotTemporalBowStack_updateOutput\n"));
	THAssert(THCudaTensor_checkGPU(state, 2, input, output));
	// TODO: arg check?

	// input: B, M (,V)
	// output: B, M'*p (,V)
	int B = THCudaTensor_size(state, input, 0);
	int M = THCudaTensor_size(state, input, 1);
	int Md = (M + (int)padBegLen + (int)padEndLen) - (int)p + 1; // output seq length
	int Mdp = Md*(int)p;
	int BMd = B*Md;
	DEBUG_PRINT(("B = %d, M = %d, Md = %d, p = %d\n", B, M, Md, (int)p));
	DEBUG_PRINT(("padBegLen = %d, padEndLen = %d, padVocabInd = %d\n", (int)padBegLen, (int)padEndLen, (int)padIndValue));
	DEBUG_PRINT(("Mdp = %d, BMd = %d\n", Mdp, BMd));

	// prepare data
	THCudaTensor_resize2d(state, output, B, Mdp);

	// stack bow input
	cudaStream_t stream = THCState_getCurrentStream(state);
	dim3 grid(DIV_CEIL(BMd, CUDA_NUM_THREADS));
	dim3 block(CUDA_NUM_THREADS);
	temporal_bow_statck_kernelV2<<<grid, block, 0, stream>>>(
			THCudaTensor_data(state, input), B, M, (int)p,
			(int)padBegLen, (int)padEndLen, (int)padIndValue,
			THCudaTensor_data(state, output)
	);

	// check error
	DEBUG_PRINT(("checking cuda error\n"));
	THCudaCheck(cudaGetLastError());
	DEBUG_PRINT(("done, no cuda error\n"));

	DEBUG_PRINT(("leaving OHNN_CudaOneHotTemporalBowStack_updateOutput\n"));
}
