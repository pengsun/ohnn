require 'ohnn'

B, M, V, C = 2, 4, 6, 2
p = 3
padBegLen = 0
padEndLen = 0
padVocabInd = 1

-- input
--input = torch.LongTensor(B, M):random(1,V):cuda()
--tmp = {6, 3, 1, 6, 4, 6, 3, 3 }
tmp = {6, 2, 1, 2, 4, 6, 3, 3}
input = torch.LongTensor(tmp):resize(B, M):cuda()
print('input = ')
print(input)

-- module
m = ohnn.OneHotTemporalBowStack(p, padBegLen, padEndLen, padVocabInd):cuda()

-- fprop
output = m:forward(input)
print('output')
print(output)
