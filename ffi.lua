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
]]

return ffi.load(libpath)
