local ffi = require 'ffi'

local libpath = package.searchpath('libohnn', package.cpath)
if not libpath then return end

require 'cunn'

ffi.cdef[[
void OHNN_CudaOneHotTemporalBowStack_updateOutput(
		THCState *state,
		// In
        THCudaTensor *input,
        double p,
        double padVocabInd,
        // Out
        THCudaTensor *output);
void OHNN_CudaLookupTable2_updateOutput(
		THCState *state,
		// In
        THCudaTensor *input,
        THCudaTensor *weight,
        // Out
        THCudaTensor *output);
]]

return ffi.load(libpath)