require'nn'
require'cunn'
require'cudnn'
ohnn = {}
ohnn.C = require 'ohnn.ffi'

require('ohnn.OneHotNarrowExt')
require('ohnn.LookupTableExt')
require('ohnn.TemporalAddBias')

require('ohnn.OneHotTemporalConvolution')
require('ohnn.OneHotTemporalBowStack')
require('ohnn.OneHotTemporalBowConvolution')
require('ohnn.OneHotTemporalSowConvolution')
require('ohnn.OneHotTemporalSeqConvolution')

return ohnn