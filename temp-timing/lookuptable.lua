--[[
th -e "require'ohnn.temp_bow_timming'"
--]]
require'cunn'
require'cudnn'

V = 30000 + 1 -- vocabulary size
C = 500
M = 80 -- seq length
p = 9
B = 100 -- #batches
padVocabInd = 1
MP = M * p

nloop = 2

-- onehot input
input = torch.LongTensor(B, MP):random(V):cuda()


function timing_module(input, m)
    cutorch.synchronize()
    local time

    -- fprop
    m:forward(input) -- warm up
    time = torch.tic()
    for i = 1, nloop do
        m:forward(input)
        cutorch.synchronize()
    end

    time = torch.toc(time)
    print(torch.type(m) .. ' fprop time ' .. time/nloop)

end

-- lookuptable
m = nn.LookupTable(V, C):cuda()
m:setPadding(padVocabInd)
print('lookuptable')
--print(m)
timing_module(input, m)

output = m:forward(input)