require 'cunn'
ohnn = {}
ohnn.C = require 'ohnn.ffi'

require('ohnn.OneHotNarrowExt')
require('ohnn.LookupTableExt')
require('ohnn.TemporalAddBias')

require('ohnn.OneHotTemporalBowStack')
require('ohnn.OneHotTemporalBowConvolution')
require('ohnn.OneHotTemporalSeqConvolution')

return ohnn