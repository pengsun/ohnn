--- nn.LookupTable2
-- fprop with cu code

local LookupTable2, parent = torch.class('ohnn.LookupTable2', 'nn.LookupTable')
local C = ohnn.C

-- class def
function LookupTable2:__init(...)
    parent.__init(self, ...)
end

function LookupTable2:updateOutput(input)
    if torch.type(input) ~= 'torch.CudaTensor' then
        return parent.updateOutput(self, input)
    end

    input = self:makeInputContiguous(input)
    assert(input:dim() == 2)

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