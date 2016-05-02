--- One-hot Temporal Sequential Convolution classdef
-- Tensor size flow:
-- B, M (,V)
--     V, C, kW
-- B, M-kW+1, C
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
local OneHotTemporalSeqConvolution, parent = torch.class('ohnn.OneHotTemporalSeqConvolution', 'nn.Sequential')

function OneHotTemporalSeqConvolution:__init(V, C, kW, opt)
    parent.__init(self)

    local function check_arg()
        assert(V>0 and C>0 and kW>0)
        self.V = V
        self.C = C
        self.kW = kW

        opt = opt or {}
        self.hasBias = opt.hasBias or false -- default no Bias
    end
    check_arg()

    -- submodules: narrow + lookuptable
    local submds = {}
    for i = 1, kW do
        local offset = i
        local length = 1 -- set it as (M - kW + 1) at runtime
        submds[i] = nn.Sequential()
            -- B, M (,V)
            :add(ohnn.OneHotNarrowExt(V, 2,offset,length))
            -- B, M-kW+1 (,V)
            :add(ohnn.LookupTableExt(V,C))
            -- B, M-kW+1, C
    end

    -- multiplexer: send input to each submodule
    local ct = nn.ConcatTable()
    for i = 1, kW do
        ct:add(submds[i])
    end

    -- the container to be returned
    local inplace = true
    -- B, M (,V)
    self:add(ct)
    -- {B, M-kW+1, C}, {B, M-kW+1, C}, ...
    self:add(nn.CAddTable(inplace))
    -- B, M-kW+1, C
    if self.hasBias == true then
        self:add(ohnn.TemporalAddBias(C, inplace))
    end
    -- B, M-kW+1, C
end

function OneHotTemporalSeqConvolution:setPadding(pv)
    local ms = self:findModules('nn.LookupTableExt')
    for _, m in ipairs(ms) do
        m:setPadding(pv)
    end
    return self
end

function OneHotTemporalSeqConvolution:zeroPaddingWeight()
    local ms = self:findModules('nn.LookupTableExt')
    for _, m in ipairs(ms) do
        local paddingInd = m.paddingValue
        if paddingInd > 0 then
            m.weight:select(1, paddingInd):fill(0)
        end
    end
    return self
end

function OneHotTemporalSeqConvolution:updateOutput(input)
    assert(input:dim()==2, "input size must be dim 2: B, M")

    -- need to the seq length for current input batch
    local M = input:size(2)
    assert(M >= self.kW,
        ("kernel size %d > seq length %d, failed"):format(self.kW, M)
    )
    self:_reset_seq_length(M)

    return parent.updateOutput(self, input)
end

--[[ Okay with default backward(), which calls each module's backward() ]]--

function OneHotTemporalSeqConvolution:shouldUpdateGradInput(flag)
    assert(flag==true or flag==false, "flag must be boolean!")

    -- set each submoule
    local function set_each_flag(mods)
        for _, md in ipairs(mods) do
            md:shouldUpdateGradInput(flag)
        end
    end
    local ms = self:findModules('ohnn.OneHotNarrowExt')
    local mms = self:findModules('ohnn.LookupTableExt')
    set_each_flag(ms)
    set_each_flag(mms)
end

function OneHotTemporalSeqConvolution:__tostring__()
    local s = string.format('%s(%d -> %d, %d',
        torch.type(self),  self.V, self.C, self.kW)
    return s .. ')'
end

-- helpers
function OneHotTemporalSeqConvolution:_reset_seq_length(M)
    local contable = self.modules[1]
    local kW = #contable.modules
    for i = 1, kW do
        -- reset nn.Narrow length
        local length = M -kW + 1
        contable.modules[i].modules[1].length = length
    end
end