--[[
th -e "require'ohnn.temp_bow_timming'"
--]]
require 'ohnn'

V = 30000 + 1 -- vocabulary size
M = 80 -- seq length
p = 9
B = 100 -- #batches
padIndValue = 1
MP = M * p

nloop = 3

-- onehot input
input = torch.LongTensor(B, M):random(V):cuda()


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

-- new one
m = ohnn.OneHotTemporalBowStack(p, 0, 0, padIndValue):cuda()
print('new one')
--print(m)
timing_module(input, m)

output = m:forward(input)