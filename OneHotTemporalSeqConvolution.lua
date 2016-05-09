--- One-hot Temporal Sequential Convolution classdef
-- Tensor size flow:
-- B, M (,V)
--     V, C, p
-- B, (M+pL+pR)-p+1, C
-- where
--   B = batch size
--   M = nInputFrame = sequence length
--   V = inputFrameSize = vocabulary size
--   C = outputFrameSize = "embedding" size
--   p = region size = convolution kernel width
--   pL, pR = padding Left, Right length
--
-- TODO: shortcut for internal module getter

local OneHotTemporalSeqConvolution, parent = torch.class(
    'ohnn.OneHotTemporalSeqConvolution',
    'ohnn.OneHotTemporalConvolution')

-- Okay with base class constructor

function OneHotTemporalSeqConvolution:makeModel()
    local V = self.V
    local C = self.C
    local p = self.p

    -- padding at beginning, end
    local dimPad, dimInputFM = 1, 1
    local mPadBeg= nn.Padding(dimPad, -self.padBegLen, dimInputFM, self.padIndValue)
    local mPadEnd = nn.Padding(dimPad, self.padEndLen, dimInputFM, self.padIndValue)
    local function null_updateGradInput(self, input, gradOutput)
        -- turn off gradInput for padding module
        self.gradInput = torch.Tensor():typeAs(gradOutput)
        return self.gradInput
    end
    mPadBeg.updateGradInput = null_updateGradInput
    mPadEnd.updateGradInput = null_updateGradInput

    -- submodules: narrow + lookuptable
    local submds = {}
    for i = 1, p do
        local offset = i
        local length = 1 -- set it as (M'' - p + 1) at runtime
        submds[i] = nn.Sequential()
        -- B, M'' (,V)
        :add(ohnn.OneHotNarrowExt(V, 2,offset,length))
        -- B, M''-p+1 (,V)
        :add(ohnn.LookupTableExt(V,C))
        -- B, M''-p+1, C
    end

    -- multiplexer: send input to each submodule
    local ct = nn.ConcatTable()
    for i = 1, p do
        ct:add(submds[i])
    end

    -- the container to be returned
    self.iModConcat = 1
    local inplace = true
    -- B, M (,V)
    if self.padBegLen > 0 then
        self:add(mPadBeg)
        self.iModConcat = self.iModConcat + 1
    end
    -- B, M' (,V)
    if self.padEndLen > 0 then
        self:add(mPadEnd)
        self.iModConcat = self.iModConcat + 1
    end
    -- B, M'' (,V)
    self:add(ct)
    -- {B, M''-p+1, C}, {B, M''-p+1, C}
    self:add(nn.CAddTable(inplace))
    -- B, M''-p+1, C
    if self.hasBias == true then
        self:add(ohnn.TemporalAddBias(C, inplace))
    end
    -- B, M''-p+1, C
end

function OneHotTemporalSeqConvolution:updateOutput(input)
    assert(input:dim()==2, "input size must be dim 2: B, M")

    -- need to the seq length for current input batch
    local M = input:size(2)
    assert(M >= self.p,
        ("kernel size %d > seq length %d, failed"):format(self.p, M)
    )
    self:_reset_seq_length(M)

    return parent.updateOutput(self, input)
end

-- Okay with default backward(), which calls each module's backward()

-- priviate
function OneHotTemporalSeqConvolution:_reset_seq_length(M)
    local contable = self.modules[self.iModConcat]
    local p = #contable.modules
    local length = (M+self.padBegLen+self.padEndLen) - p + 1
    for i = 1, p do
        -- reset nn.Narrow length
        contable.modules[i].modules[1].length = length
    end
end