--- One-hot Temporal Bag-Of-Word Convolution classdef
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
local OneHotTemporalBowConvolution, parent = torch.class(
    'ohnn.OneHotTemporalBowConvolution',
    'ohnn.OneHotTemporalConvolution')

function OneHotTemporalBowConvolution:checkArg(V, C, p, opt)
    parent.checkArg(self, V, C, p, opt)


    if self.padIndValue == nil then
        error("bow conv requires padIndValue be specified, " ..
            "where any duplicate words in local window will be replaced with padIndValue."
        )
    end

end

function OneHotTemporalBowConvolution:makeModel()
    local padIndValue = self.padIndValue
    local padBegLen = self.padBegLen
    local padEndLen = self.padEndLen
    local p = self.p
    local V = self.V
    local C = self.C

    -- B, M (,V)
    self:add( ohnn.OneHotTemporalBowStack(p, padBegLen, padEndLen, padIndValue) )
    -- B, M''*p (,V)
    self:add( ohnn.LookupTableExt(V, C) )
    -- B, M''*p, HU
    self:add( nn.Unsqueeze(1, 2) )
    -- B, 1, M''*p, HU
    self:add( cudnn.SpatialAveragePooling(1,p, 1,p, 0,0) )
    -- B, 1, M'', HU
    self:add( nn.MulConstant(p, true) )
    self:add( nn.Squeeze(1, 3) )
    -- B, M'', HU
    if true == self.hasBias then
        self:add( ohnn.TemporalAddBias(C, true) )
    end

end

function OneHotTemporalBowConvolution:updateOutput(input)
    assert(input:dim()==2, "input size must be dim 2: B, M")
    return parent.updateOutput(self, input)
end

-- Okay with default backward(), which calls each module's backward()

