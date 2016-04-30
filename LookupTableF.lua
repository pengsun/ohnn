--- ohnn.LookupTable2
-- fprop with cu code

local LookupTableF, parent = torch.class('ohnn.LookupTableF', 'nn.LookupTable')
local C = ohnn.C

-- class def
function LookupTableF:__init(...)
    parent.__init(self, ...)
end

function LookupTableF:updateOutput(input)
    if torch.type(input) ~= 'torch.CudaTensor' then -- cpu data
        return parent.updateOutput(self, input)
    end
    if (input:dim() == 1) then -- single batch
        return parent.updateOutput(self, input)
    end

    assert(input:dim() == 2)
    input = self:makeInputContiguous(input)

    C.OHNN_CudaLookupTable2_updateOutput(
        cutorch.getState(),
        -- in
        input:cdata(),
        self.weight:cdata(),
        -- out
        self.output:cdata()
    )

    return self.output
end