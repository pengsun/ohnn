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

require'torch'
require'nn'

-- main methods
local OneHotTemporalBowConvolution, parent = torch.class(
    'ohnn.OneHotTemporalBowConvolution',
    'nn.Sequential'
)

function OneHotTemporalBowConvolution:__init(V, C, p, opt)
    parent.__init(self)

    local function check_arg()
        assert(V>0 and C>0 and p >0)
        self.V = V
        self.C = C
        self.p = p
        assert(p %2 == 1, "region size (kernel width) p must be an even number!")

        opt = opt or {}
        self.hasBias = opt.hasBias or true
        self.vocabIndPad = opt.vocabIndPad or 1
        self.isStrictBow = opt.isStrictBow or false
    end
    check_arg()

    -- which kind: Bag-Of-Word or Sum-Of-Word
    if true == self.isStrictBow then
        self:makeBagOfWord()
    else
        self:makeSumOfWord()
    end

    -- vocabulary index padding
    self:setVocabIndPad(self.vocabIndPad)
end

function OneHotTemporalBowConvolution:makeBagOfWord()
    local indUnknown = self.vocabIndPad
    local p = self.p
    local V = self.V
    local C = self.C

    -- B, M (,V)
    self:add( ohnn.OneHotTemporalBowStack(p, indUnknown) )
    -- B, Mp (,V)
    self:add( ohnn.LookupTableExt(V, C) )
    -- B, Mp, HU
    self:add( nn.Unsqueeze(1, 2) )
    -- B, 1, Mp, HU
    self:add( cudnn.SpatialAveragePooling(1,p, 1,p, 0,0) )
    -- B, 1, M, HU
    self:add( nn.MulConstant(p, true) )
    self:add( nn.Squeeze(1, 3) )
    -- B, M, HU
    if true == self.hasBias then
        self:add( ohnn.TemporalAddBias(C) )
    end

end

function OneHotTemporalBowConvolution:makeSumOfWord()
    local p = self.p
    local V = self.V
    local C = self.C
    local pad = (p -1)/2

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
        self:add( ohnn.TemporalAddBias(C) )
    end
    -- B, M, C
end

function OneHotTemporalBowConvolution:updateOutput(input)
    assert(input:dim()==2, "input size must be dim 2: B, M")
    return parent.updateOutput(self, input)
end

-- Okay with default backward(), which calls each module's backward()

-- additional methods
function OneHotTemporalBowConvolution:setVocabIndPad(vip)
    self.vocabIndPad = vip

    local ms = self:findModules('ohnn.LookupTableExt')
    assert(#ms > 0)
    for _, m in ipairs(ms) do
        m:setPadding(vip)
    end
    return self
end

function OneHotTemporalBowConvolution:zeroVocabIndPadWeight()
    local ms = self:findModules('ohnn.LookupTableExt')
    assert(#ms > 0)
    for _, m in ipairs(ms) do
        local vocabIndPad = m.paddingValue or error('currupted code... no paddingValue')
        m.weight:select(1, vocabIndPad):fill(0)
    end
    return self
end

function OneHotTemporalBowConvolution:shouldUpdateGradInput(flag)
    assert(flag==true or flag==false, "flag must be boolean!")

    if true == flag then
        error('updateGradInput() not yet implemented for bow conv...')
    end

    -- set each submoule
    local function set_each_flag(mods)
        for _, md in ipairs(mods) do
            md:shouldUpdateGradInput(flag)
        end
    end
    set_each_flag( self:findModules('ohnn.LookupTableExt') )
end

function OneHotTemporalBowConvolution:__tostring__()
    local s = string.format('%s(%d -> %d, %d',
        torch.type(self),  self.V, self.C, self.p)
    return s .. ')'
end
