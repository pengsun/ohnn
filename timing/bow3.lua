--[[
th -e "require'ohnn.temp_bow_timming'"
--]]
require'cudnn'
require'onehot-temp-conv'
require 'ohnn'

V = 30000 + 1 -- vocabulary size
C = 500 -- output dim
M = 80 -- seq length
p = 9
MP = M * p
B = 100 -- #batches
padVocabInd = 1

nloop = 3

-- onehot input
input = torch.LongTensor(B, M):random(V):cuda()

-- dense grad output
gOutput = torch.CudaTensor(B, M, C):normal():cuda()

function timing_module(input, gOutput, m)
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

-- old one
function create_old_bowConv(V, C, p, padVocabInd)
    local mcontrol = nn.OneHotTemporalConvolution(V, C, 1, {hasBias = false})
    --local mcontrol = nn.LookupTable(V, C)

    local function get_pad()
        assert(p %2 == 1)
        return (p -1)/2
    end
    local pad = get_pad(p)
    local stride = 1

    local md = nn.Sequential()
    -- B, M (,V)
    md:add( mcontrol )
    -- B, M, C
    md:add( nn.Unsqueeze(1, 2) )
    -- B, 1, M, C
    md:add( cudnn.SpatialAveragePooling(1,p, 1,stride, 0,pad) )
    md:add( nn.MulConstant(p, true) )
    md:add( nn.Squeeze(1, 3) )
    -- B, M, C
    md:add( nn.TemporalAddBias(C) )
    -- B, M, C

    mcontrol:setPadding(padVocabInd):zeroPaddingWeight()
    md = md:cuda()
    return md
end
mold = create_old_bowConv(V, C, p, padVocabInd)
print('old one')
--print(mold)
timing_module(input, gOutput, mold)

-- comb
function create_comb_bowConv(V, C, p, padVocabInd)
    local mcontrol = nn.OneHotTemporalConvolution(V, C, 1, {hasBias = false})
    --local mcontrol = nn.LookupTable(V, C)

    local md = nn.Sequential()
    -- B, Mp (,V)
    md:add( mcontrol )
    -- B, Mp, C
    md:add( nn.Unsqueeze(1, 2) )
    -- B, 1, Mp, C
    md:add( cudnn.SpatialAveragePooling(1,p) )
    -- B, 1, M, C
    md:add( nn.MulConstant(p, true) )
    md:add( nn.Squeeze(1, 3) )
    -- B, M, C
    md:add( nn.TemporalAddBias(C) )
    -- B, M, C

    mcontrol:setPadding(padVocabInd):zeroPaddingWeight()
    md = md:cuda()
    return md
end
mm = create_comb_bowConv(V, C, p, padVocabInd)
print('comb one')
--print(m)
input2 = input:repeatTensor(1, p):clone()
timing_module(input2, gOutput, mm)

-- new one
function create_new_bowConv(V, C, p, padVocabInd)
    local mcontrol = nn.OneHotTemporalConvolution(V, C, 1, {hasBias = false})
    --local mcontrol = nn.LookupTable(V, C)

    local md = nn.Sequential()
    -- B, M (,V)
    md:add( ohnn.OneHotTemporalBowStack(p, padVocabInd) )
    -- B, Mp (,V)
    md:add( mcontrol )
    -- B, Mp, C
    md:add( nn.Unsqueeze(1, 2) )
    -- B, 1, Mp, C
    md:add( cudnn.SpatialAveragePooling(1,p) )
    -- B, 1, M, C
    md:add( nn.MulConstant(p, true) )
    md:add( nn.Squeeze(1, 3) )
    -- B, M, C
    md:add( nn.TemporalAddBias(C) )
    -- B, M, C

    mcontrol:setPadding(padVocabInd):zeroPaddingWeight()
    md = md:cuda()
    return md
end
m = create_new_bowConv(V, C, p, padVocabInd)
print('new one')
--print(m)
timing_module(input, gOutput, m)