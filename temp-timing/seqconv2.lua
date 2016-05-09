require'cunn'
require'cudnn'
require'ohnn'
require'onehot-temp-conv'

V = 30000 + 1 -- vocabulary size
C = 500
M = 95 -- seq length
p = 3
B = 100 -- #batches
dummyVocabInd = 1

pBeg = 1 -- padding beginning
pEnd = 3 -- padding end
Mdd = M + pBeg + pEnd

nloop = 3

-- onehot input
input = torch.LongTensor(B, M):random(V):cuda()
weight = torch.CudaTensor(V, C):normal()

-- gradOutput
gOutput = torch.CudaTensor(B, Mdd-p+1, C):normal()

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
m1 = ohnn.OneHotTemporalSeqConvolution(V, C, p, {hasBias = true,
        padBegLen = pBeg, padEndLen = pEnd, padIndValue = dummyVocabInd,
        dummyVocabInd = dummyVocabInd}
):cuda()

-- old
local function create_old()
    -- pad for input
    local inputDim = 1
    local mPadLeft = nn.Padding(1, -pBeg, inputDim, dummyVocabInd)
    local mPadRight = nn.Padding(1, pEnd, inputDim, dummyVocabInd)
    -- turn off gradInput for padding module
    local function null_updateGradInput(self, input, gradOutput)
        self.gradInput = torch.Tensor():typeAs(gradOutput)
        return self.gradInput
    end
    mPadLeft.updateGradInput = null_updateGradInput
    mPadRight.updateGradInput = null_updateGradInput


    local md = nn.Sequential()
    md:add( mPadLeft )
    md:add( mPadRight )
    md:add( nn.OneHotTemporalConvolution(V, C, p, {hasBias = true}):setPadding(dummyVocabInd) )

    md:cuda()
    return md
end
m2 = create_old()

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
print(m1)
timing_module(input, gOutput, m1)
output1 = m1:forward(input)

print('old')
--print(m2)
timing_module(input, gOutput, m2)
output2 = m2:forward(input)

-- verify?
function calc_diff(a, b)
    local c = a:view(-1) - b:view(-1)
    return c:abs():max()
end
d = calc_diff(output1, output2)
print( ('diff = %f'):format(d) )