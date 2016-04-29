require 'ohnn'


V = 3001 -- vocabulary size
C = 500 -- output dim
M = 80 -- seq length
p = 9
B = 10 -- #batches
padVocabInd = 1

-- input
input = torch.LongTensor(B, M):random(V):cuda()
print('input = ')
print(input:size())

-- module
m = ohnn.OneHotTemporalBowConvolution(V, C, p, padVocabInd)

print('weight = ')
print(m.weight:size())
print('bias = ')
print(m.bias:size())
print('gradWeight = ')
print(m.gradWeight:size())
print('gradBias = ')
print(m.gradBias:size())

-- fprop
output = m:forward(input)
print('output')
print(output:size())

-- cahced input
print('bowInputVal:')
print(m.bowInputVal:size())
print('bowInputRowPtr:')
print(m.bowInputRowPtr:size())
print('bowInputColInd:')
print(m.bowInputColInd:size())


-- bprop
gradOutput = torch.CudaTensor(B, M, C):fill(1)
print('gradOutput = ')
print(gradOutput:size())

m:backward(input, gradOutput)
print('gradWeight = ')
print(m.gradWeight:size())
print('gradBias = ')
print(m.gradBias:size())