--- One-hot Temporal Sum-Of-Word Convolution classdef
-- Tensor size flow:
-- B, M (,V)
--     V, C
-- B, M, C
-- where
--   B = batch size
--   M = nInputFrame = sequence length
--   V = inputFrameSize = vocabulary size
--   C = outputFrameSize = "embedding" size
--
-- TODO: shortcut for internal module getter

-- main methods
local OneHotTemporalSowConvolution, parent = torch.class(
    'ohnn.OneHotTemporalSowConvolution',
    'ohnn.OneHotTemporalConvolution')

-- Okay with base class constructor

function OneHotTemporalSowConvolution:checkArg(V, C, p, opt)
    parent.checkArg(self, V, C, p, opt)

    assert(self.padBegLen == self.padEndLen,
        "Sum-of-Word conv requires padBegLen == padEndLen!")
    assert(self.padIndValue == nil,
        "Sum-of-Word conv ignores padIndValue and always pads zero values " ..
        "at outputs... set padIndValue as nil.")
end

function OneHotTemporalSowConvolution:makeModel()
    local p = self.p
    local V = self.V
    local C = self.C
    local pad = self.padBegLen

    -- B, M (,V)
    self:add( ohnn.LookupTableExt(V, C) )
    -- B, M, C
    self:add( nn.Unsqueeze(1, 2) )
    -- B, 1, M, C
    self:add( cudnn.SpatialAveragePooling(1,p, 1,1, 0,pad) )
    self:add( nn.MulConstant(p, true) )
    self:add( nn.Squeeze(1, 3) )
    -- B, M, C
    if true == self.hasBias then
        self:add( ohnn.TemporalAddBias(C, true) )
    end
    -- B, M, C
end

function OneHotTemporalSowConvolution:updateOutput(input)
    assert(input:dim()==2, "input size must be dim 2: B, M")
    return parent.updateOutput(self, input)
end

-- Okay with default backward(), which calls each module's backward()

