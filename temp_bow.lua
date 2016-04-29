--[[
th -e "require'ohnn.temp_bow'"
--]]
require 'ohnn'

B, M, V, C = 2, 4, 6, 2
p = 3
padVocabInd = 1

-- input
--input = torch.LongTensor(B, M):random(1,V):cuda()
tmp = {6, 3, 1, 6, 4, 6, 3, 3}
input = torch.LongTensor(tmp):resize(B, M):cuda()
print('input = ')
print(input)

-- module
m = ohnn.OneHotTemporalBowConvolution(V, C, p, padVocabInd)
--ww = torch.range(1,V):resize(1,V):repeatTensor(C,1)
tmp = {1,2,3,4,5,6, 2,5,8,7,1,3}
ww = torch.Tensor(tmp):resize(C,V):contiguous():cuda()
m.weight:copy(ww)

print('weight = ')
print(m.weight)
print('bias = ')
print(m.bias)
print('gradWeight = ')
print(m.gradWeight)
print('gradBias = ')
print(m.gradBias)

-- fprop
output = m:forward(input)
print('output')
print(output)

-- cahced input
print('bowInputVal:')
print(m.bowInputVal)
print('bowInputRowPtr:')
print(m.bowInputRowPtr)
print('bowInputColInd:')
print(m.bowInputColInd)

-- bprop
--gradOutput = torch.CudaTensor(B, M, C):fill(1)
tmp = torch.range(1,C):resize(1, C):repeatTensor(B*M,1):resize(B,M,C)
gradOutput = tmp:cuda():clone()
print('gradOutput = ')
print(gradOutput)

m.gradWeight:zero()
m.gradBias:zero()

m:backward(input, gradOutput)
print('gradWeight = ')
print(m.gradWeight)
print('gradBias = ')
print(m.gradBias)