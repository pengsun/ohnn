--- One-hot Temporal Sequential Convolution, base classdef

local OneHotTemporalConvolution, parent = torch.class(
    'ohnn.OneHotTemporalConvolution',
    'nn.Sequential')

function OneHotTemporalConvolution:__init(V, C, p, opt)
    parent.__init(self)
    self:checkArg(V, C, p, opt)
    self:makeModel()
    self:setDummyVocabInd(self.dummyVocabInd)
end

function OneHotTemporalConvolution:checkArg(V, C, p, opt)
    assert(V>0 and C>0 and p >0)
    self.V = V
    self.C = C
    self.p = p

    opt = opt or {}
    self.hasBias = (true == opt.hasBias) and true or false -- default no Bias
    self.padBegLen = opt.padBegLen or 0
    self.padEndLen = opt.padEndLen or 0
    self.padIndValue = opt.padIndValue or nil
    self.dummyVocabInd = opt.dummyVocabInd or 0 -- default no dummy vocabular index
end

function OneHotTemporalConvolution:makeModel()
    -- make the concrete model in derived clas
end

function OneHotTemporalConvolution:setDummyVocabInd(dvi)
    self.dummyVocabInd = dvi

    local ms = self:findModules('ohnn.LookupTableExt'); assert(#ms > 0);
    for _, m in ipairs(ms) do
        m:setPadding(dvi)
    end
    return self
end

function OneHotTemporalConvolution:zeroDummyVocabIndWeight()
    if (not self.dummyVocabInd) or (self.dummyVocabInd < 1) then
        error('dummyVocabInd not properly set, cannot zero the corresponding weights.')
    end

    local ms = self:findModules('ohnn.LookupTableExt'); assert(#ms > 0);
    for _, m in ipairs(ms) do
        local dvi = m.paddingValue; assert(dvi==self.dummyVocabInd);
        m.weight:select(1, dvi):fill(0)
    end
    return self
end

function OneHotTemporalConvolution:shouldUpdateGradInput(flag)
    assert(flag==true or flag==false, "flag must be boolean!")
    if true == flag then
        error('gradInput not implemented yet...')
    end

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

function OneHotTemporalConvolution:__tostring__()
    local s = string.format('%s(%d -> %d, %d)',
        torch.type(self),  self.V, self.C, self.p
    )
    s = s .. ' hasBias=' .. tostring(self.hasBias)
    s = s .. ' padBegLen=' .. self.padBegLen
    s = s .. ' padEndLen=' .. self.padEndLen
    s = s .. ' padIndValue=' .. tostring(self.padIndValue)
    s = s .. ' dummyVocabInd=' .. tostring(self.dummyVocabInd)
    return s
end