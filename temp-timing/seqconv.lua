require'cunn'
require'cudnn'
require'ohnn'
require'onehot-temp-conv'

V = 30000 + 1 -- vocabulary size
C = 500
M = 95 -- seq length
p = 3
B = 100 -- #batches
vocabIndPad = 1

nloop = 3

-- onehot input
input = torch.LongTensor(B, M):random(V):cuda()
weight = torch.CudaTensor(V, C):normal()

-- gradOutput
gOutput = torch.CudaTensor(B, M-p+1, C):normal()

function timing_module(input, gOutput, m)
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

    -- bprop
    m:backward(input, gOutput)
    time = torch.tic()
    for i = 1, nloop do
        m:backward(input, gOutput)
        cutorch.synchronize()
    end

    time = torch.toc(time)
    print(torch.type(m) .. ' bprop time ' .. time/nloop)

end

-- new
m1 = ohnn.OneHotTemporalSeqConvolution(V, C, p, {hasBias = true}):cuda()
m1:setVocabIndPad(vocabIndPad)

-- old
m2 = nn.OneHotTemporalConvolution(V, C, p, {hasBias = true}):cuda()
m2:setPadding(vocabIndPad)

-- common weights bias
local function enforce_param(m1, m2)
    p1 = m1:getParameters()
    p2 = m2:getParameters()
    assert(p1:nElement() == p2:nElement())
    p1:copy(p2)
end
enforce_param(m1, m2)
cutorch.synchronize()

-- do the timing
print('new')
timing_module(input, gOutput, m1)
output1 = m1:forward(input)

print('old')
timing_module(input, gOutput, m2)
output2 = m2:forward(input)

-- verify?
function calc_diff(a, b)
    local c = a:view(-1) - b:view(-1)
    return c:abs():max()
end
d = calc_diff(output1, output2)
print( ('diff = %f'):format(d) )