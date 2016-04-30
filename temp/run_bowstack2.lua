require 'ohnn'

tmp = {1, 3, 4, 4, 3, 5, 6, 2, 3}

B = 1
M = #tmp
V = tablex.reduce(function (a,b) return (a > b) and a or b end, tmp)
p = 5
padVocabInd = 1

-- input
input = torch.LongTensor(tmp):resize(B, M):cuda()
print('input = ')
print(input)

-- module
m = ohnn.OneHotTemporalBowStack(p, padVocabInd):cuda()

-- fprop
output = m:forward(input)
print('output')
print(output)
