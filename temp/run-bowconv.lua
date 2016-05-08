require'cunn'
require'ohnn'

B = 200 -- batch size
M = 45 -- sequence length (#words)
V = 12333 -- inputFrameSize (vocabulary size)
C = 300 -- outputFrameSize (#output feature maps, or embedding size)
p = 5 -- window size for bag-of-word
vocabPadInd = 1 -- the first word in vocabulary is seen as padding word

-- inputs: the one-hot vector as index in set {1,2,...,V}. size: B, M
inputs = torch.LongTensor(B, M):random(1,V):cuda()


-- the 1d bag-of-word conv module
m1 = ohnn.OneHotTemporalSeqConvolution(V, C, p,
    {hasBias = true, vocabPadInd = vocabPadInd, isStrictBOW = true}
):cuda()
-- outputs: the dense tensor. size: B, M-kW+1, C
outputs1 = m1:forward(inputs)
-- back prop: the gradients w.r.t. parameters
gradOutputs1 = outputs1:clone():normal()
m1:backward(inputs, gradOutputs1)


-- the 1d sum-of-word conv module
m2 = ohnn.OneHotTemporalSeqConvolution(V, C, p,
    {hasBias = true, vocabPadInd = vocabPadInd, isStrictBOW = false}
):cuda()
-- outputs: the dense tensor. size: B, M-kW+1, C
outputs2 = m2:forward(inputs)

-- back prop: the gradients w.r.t. parameters
gradOutputs2 = outputs2:clone():normal()
m2:backward(inputs, gradOutputs2)
