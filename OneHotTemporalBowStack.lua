-- One-hot Temporal Bag-Of-Word Stacking classdef
-- Tensor size flow:
-- Input: B, M (,V)
-- Output: B, Mp (,V)
-- where
--   B = batch size
--   M = nInputFrame = sequence length
--   V = inputFrameSize = vocabulary size
--   p = convolution kernel size = region size
--

require'cutorch'
require'nn'

-- main methods
local OneHotTemporalBowStack, parent = torch.class(
    'ohnn.OneHotTemporalBowStack',
    'nn.Module')
local C = ohnn.C

function OneHotTemporalBowStack:__init(p, padBegLen, padEndLen, padIndValue, opt)
    parent.__init(self)

    local function check_arg()
        self.p = p or error('no p: must specify kernel window size')
        assert(self.p >= 1)

        self.padBegLen = padBegLen or error('no padBegLen')
        self.padEndLen = padEndLen or error('no padEndLen')

        self.padIndValue = padIndValue or
            error('no padIndValue: must specify a padding vocabulary index')

        opt = opt or {}
    end
    check_arg()

    -- output
    self.output = torch.Tensor()
end

function OneHotTemporalBowStack:updateOutput(input)
    C.OHNN_CudaOneHotTemporalBowStack_updateOutput(
        cutorch.getState(),
        -- in
        input:cdata(),
        self.p,
        self.padBegLen,
        self.padEndLen,
        self.padIndValue,
        -- out
        self.output:cdata()
    )
    return self.output
end

function OneHotTemporalBowStack:updateGradInput(input, gradOutput)
    -- not implemented yet, pass a null tensor
    self.gradInput = self.gradInput or gradOutput.new()
    return self.gradInput
end

function OneHotTemporalBowStack:__tostring__()
    local s = string.format('%s(%d) padBegLen=%d padEndLen=%d padIndValue=%d',
        torch.type(self), self.p, self.padBegLen, self.padEndLen, self.padIndValue)
    return s
end

