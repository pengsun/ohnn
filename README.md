# ohnn

Implement the OneHot Temporal Convolution (oh-conv) defined in [1]. 
For NLP task, it directly applies the convolution over the high-dimensional one-hot word vector sequence.
This way, pre-trained word embedding is unnecessary [1], although semi-supervised learning with additional un-labeled data can further boost the performance [2, 3].
This is a re-implementation as Torch 7 `nn` module (See [3] for the original C++/CUDA C code).
Currently it must work with GPU.

Hereafter, we use the following notations compatible with `nn.TemporalConvolution`:
```
  B = batch size
  M = sequence length = nInputFrame = #words
  V = inputFrameSize = vocabulary size
  C = outputFrameSize = #output feature maps = #hidden units = embedding size
  p = convolution kernel size = kernel width
  pB = padding length at sequence beginning
  pE = padding length at sequence end
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
The following classes are exposed:
* `ohnn.OneHotTemporalSeqConvolution`: convolution over one-hot word vector sequence.
* `ohnn.OneHotTemporalBowConvolution`: Bag-of-Word convolution.
* `ohnn.OneHotTemporalSowConvolution`: Sum-of-Word convolution. 

See TODO for basic examples.

See TODO for examples of text classification using oh-conv.

#### ohnn.OneHotTemporalSeqConvolution
```lua
module = ohnn.OneHotTemporalSeqConvolution(V, C, p [,opt])
```
Construct class for seq-conv [1], which means sequential convolution that is exactly a conventional temporal convolution over high-dimensional one-hot vector sequence. 
Expect tensor size:
```
input: B, M (,V)
   kernel: V, C, p
output: B, (M+pB+pE)-p+1, C
```
The option `opt` can have the following fields:
`hasBias`: true means there is a bias term. Default true.
`padBegLen`: padding length at sequence beginning. Default 0. 
`padEndLen`: padding length at sequence end. Default 0. 
`padIndValue`: padded index value (over the vocabulary) for the padding at sequence beginning and end. Default nil. 
`dummyVocabInd`: vocabulary index seen as dummy one (e.g., for out-of-vocabulary word). Default nil. 

#### ohnn.OneHotTemporalBowConvolution
```lua
module = ohnn.OneHotTemporalSeqConvolution(V, C, p [,opt])
```
Construct class for bow-conv [1], which means bag-of-word convolution that can be seen as shared-weight linear regression over bag-of-word extracted from local window. 
Expect tensor size:
```
input: B, M (,V)
   kernel: V, C
output: B, (M+pB+pE)-p+1, C
```
The option `opt` can have the following fields:
`hasBias`: true means there is a bias term. Default true.
`padBegLen`: padding length at sequence beginning. Default 0. 
`padEndLen`: padding length at sequence end. Default 0. 
`padIndValue`: padded index value (over the vocabulary) for the padding at sequence beginning and end. Default nil. 
`dummyVocabInd`: vocabulary index seen as dummy one (e.g., for out-of-vocabulary word). Default nil. 

#### ohnn.OneHotTemporalSowConvolution
```lua
module = ohnn.OneHotTemporalSeqConvolution(V, C, p [,opt])
```
Construct class for sum-of-word (SOW) convolution. 
SOW conv is much like BOW conv. 
The difference lies in that BOW conv counts duplicate words only one time, while SOW conv counts as many times as a word can show up.
Expect tensor size:
```
input: B, M (,V)
   kernel: V, C
output: B, (M+pB+pE)-p+1, C
```
The option `opt` can have the following fields:
`hasBias`: true means there is a bias term. Default true.
`padBegLen`: padding length at sequence beginning. Default 0. 
`padEndLen`: padding length at sequence end. MUST equal to `padBegLen`. Default 0. 
`padIndValue`: padded index value (over the vocabulary) for the padding at sequence beginning and end. Default nil. 
`dummyVocabInd`: vocabulary index seen as dummy one (e.g., for out-of-vocabulary word). Default nil. 

##Reference
[1] Rie Johnson and Tong Zhang. Effective use of word order for text categorization with convolutional neural networks. NAACL-HLT 2015. 

[2] Rie Johnson and Tong Zhang. Semi-supervised convolutional neural networks for text categorization via region embedding. NIPS 2015.

[3] http://riejohnson.com/cnn_download.html
