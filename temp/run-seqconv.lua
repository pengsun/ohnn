require'cunn'
require'ohnn'

B = 200 -- batch size
M = 45 -- sequence length (#words)
V = 12333 -- inputFrameSize (vocabulary size)
C = 300 -- outputFrameSize (#output feature maps, or embedding size)
kW = 5 -- convolution kernel size (width)

-- inputs: the one-hot vector as index in set {1,2,...,V}. size: B, M
inputs = torch.LongTensor(B, M):random(1,V):cuda()

-- the 1d conv module
tf = ohnn.OneHotTemporalSeqConvolution(V, C, kW, {hasBias = true}):cuda()

-- outputs: the dense tensor. size: B, M-kW+1, C
outputs = tf:forward(inputs)

-- back prop: the gradients w.r.t. parameters
gradOutputs = outputs:clone():normal()
tf:backward(inputs, gradOutputs)
