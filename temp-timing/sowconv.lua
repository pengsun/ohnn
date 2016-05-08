require'cunn'
require'cudnn'
require'ohnn'
require'onehot-temp-conv'

V = 30000 + 1 -- vocabulary size
C = 500
M = 95 -- seq length
p = 3
B = 100 -- #batches
vocabIndPad = 7

nloop = 3

-- onehot input
input = torch.LongTensor(B, M):random(V):cuda()
weight = torch.CudaTensor(V, C):normal()

-- gradOutput
gOutput = torch.CudaTensor(B, M, C):normal()

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
m1 = ohnn.OneHotTemporalBowConvolution(V, C, p,
    {hasBias = true, isStrictBow = false}
)
m1:cuda()
m1:setVocabIndPad(vocabIndPad)

-- old
local function make_old(V, C, p)
    local function get_pad()
        assert(p %2 == 1)
        return (p -1)/2
    end
    local pad = get_pad(p)
    local stride = 1

    --local m = nn.OneHotTemporalConvolution(V, C, 1, {hasBias = false})
    --m:setPadding(vocabIndPad):zeroPaddingWeight()

    local m = nn.LookupTable(V, C)
    m:setPadding(VocabIndPad)

--    local m = ohnn.LookupTableExt(V, C)
--    m:setPadding(VocabIndPad)

    local md = nn.Sequential()
    -- B, M (,V)
    md:add( m )
    -- B, M, HU
    md:add( nn.Unsqueeze(1, 2) )
    -- B, 1, M, HU
    md:add( cudnn.SpatialAveragePooling(1,p, 1,stride, 0,pad) )
    md:add( nn.MulConstant(p, true) )
    md:add( nn.Squeeze(1, 3) )
    -- B, M, HU
    md:add( nn.TemporalAddBias(C) )
    -- B, M, HU

    md:cuda()

    return md
end
m2 = make_old(V, C, p)

-- common weights bias
local function enforce_param(m1, m2)
    p1 = m1:getParameters()
    p2 = m2:getParameters()
    assert(p1:nElement() == p2:nElement())
    p1:copy(p2)
end
enforce_param(m2, m1)
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