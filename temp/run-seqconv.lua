require'cunn'
require'ohnn'

B = 200 -- batch size
M = 45 -- sequence length (#words)
V = 12333 -- inputFrameSize (vocabulary size)
C = 300 -- outputFrameSize (#output feature maps, or embedding size)
p = 5 -- convolution kernel size (width)
dummyVocabInd = 1 -- the first word in vocabulary is seen as padding word

-- inputs: the one-hot vector as index in set {1,2,...,V}. size: B, M
inputs = torch.LongTensor(B, M):random(1,V):cuda()

-- the 1d seq conv module
m = ohnn.OneHotTemporalSeqConvolution(V, C, p,
    {hasBias = true, dummyVocabInd = dummyVocabInd}
):cuda()

-- outputs: the dense tensor. size: B, M-kW+1, C
outputs = m:forward(inputs)

-- back prop: the gradients w.r.t. parameters
gradOutputs = outputs:clone():normal()
m:backward(inputs, gradOutputs)
