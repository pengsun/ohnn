# ohnn

Implement the OneHot Temporal Convolution (oh-conv) defined in [1]. 
For NLP task, it directly applies the convolution over the high-dimensional one-hot word vector sequence.
This way, pre-trained word embedding is unnecessary [1], although unsupervised learning on un-labeled data can further boost the performance [2, 3].
This is a re-implementation as Torch 7 `nn` module (See [3] for the original C++/CUDA C code).
Currently it must work with GPU.

Hereafter, we use the following notations compatible with `nn.TemporalConvolution`:
```
  B = batch size
  M = sequence length = nInputFrame = #words
  V = inputFrameSize = vocabulary size
  C = outputFrameSize = #output feature maps = #hidden units = embedding size
  kW = convolution kernel size = kernel width
```

The input data layout for `ohnn` is compatible with that for `nn.LookupTable`, 
i.e., input should be a `B, M` sized tensor with each element indicating a vocabulary index.
In another word, the input can be understood as a `B, M, V` sized tensor consisting of up to `B*M` one-hot vectors. 

## Requirements
* Torch
* cunn
* cudnn


## Install
git clone the code, cd to the directory, and run command ```luarocks make```.

Then the lib will ba installed to your torch 7 directory. Delete the git-cloned source directory if you like.


## Usage
See TODO for basic examples.

See TODO for examples of text classification using oh-conv.

#### ohnn.OneHotTemporalSeqConvolution
```lua
module = ohnn.OneHotTemporalSeqConvolution(V, C, p [,opt])
```
Construct class for seq-conv [1], which means sequential convolution that is exactly a conventional temporal convolution over high-dimensional one-hot vector sequence. 
Always reduce the sequence length. Expect tensor size:
```
input: B, M (,V)
   kernel: V, C, p
output: B, M-p+1, C
```
The option `opt` can have the following fields:
`hasBias`: true means there is a bias term. Default true.
`vocabIndPad`: padding index of the vocabulary for unknown or filling words. Default 1.


#### ohnn.OneHotTemporalBowConvolution
```lua
module = ohnn.OneHotTemporalSeqConvolution(V, C, p [,opt])
```
Construct class for bow-conv [1], which means bag-of-word convolution that can be seen as shared-weight linear regression over bag-of-word extracted from local window. 
`p` must be a even number. Always keep the sequence length. Expect tensor size:
```
input: B, M (,V)
   kernel: V, C
output: B, M, C
```
The option `opt` can have the following fields:
`hasBias`: true means there is a bias term. Default true.
`vocabIndPad`: padding index of the vocabulary for unknown or filling words. Default 1.
`isStrictBow`: true means strict bag-of-word (multiple word occurrence is only counted 1 time), false means sum-of-word (count as many times a word can occur). 
SOM should have similar performance with BOW for natural word sequence. 
Also, due to implementation issue, SOW is much faster than BOW. 
Default true.
`edgeVocabIndPad`: when `isStrictBow = true`, this field must be specified to indicate what vocabulary index is padded at both word-sequence edges. 
When `isStrictBow = false`, this field is ignored.

##Reference
[1] Rie Johnson and Tong Zhang. Effective use of word order for text categorization with convolutional neural networks. NAACL-HLT 2015. 

[2] Rie Johnson and Tong Zhang. Semi-supervised convolutional neural networks for text categorization via region embedding. NIPS 2015.

[3] http://riejohnson.com/cnn_download.html
