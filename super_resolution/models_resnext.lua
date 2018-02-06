require 'nn'
require 'cunn'
require 'nngraph'

local utils = require 'super_resolution.utils'

local M = {}

 -- The shortcut layer is either identity or 1x1 convolution
 local function shortcut(nInputPlane, nOutputPlane, stride)
  local useConv = shortcutType == 'C' or
  (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
  if useConv then
       -- 1x1 convolution
       return nn.Sequential()
       :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
       :add(SBatchNorm(nOutputPlane))
       elseif nInputPlane ~= nOutputPlane then
       -- Strided, zero-padded identity shortcut
       return nn.Sequential()
       :add(nn.SpatialAveragePooling(1, 1, stride, stride))
       :add(nn.Concat(2)
         :add(nn.Identity())
         :add(nn.MulConstant(0)))
     else
       return nn.Identity()
     end
   end

 -- The aggregated residual transformation bottleneck layer, Form (B)
--[[
 local function split(nInputPlane, d, c, stride)
  local cat = nn.ConcatTable()
  for i=1,c do
   local s = nn.Sequential()
   s:add(Convolution(nInputPlane,d,1,1,1,1,0,0))
   s:add(SBatchNorm(d))
   s:add(ReLU(true))
   s:add(Convolution(d,d,3,3,stride,stride,1,1))
   s:add(SBatchNorm(d))
   s:add(ReLU(true))
   cat:add(s)
 end
 return cat
end
]]

--原残差模块
--[[
local function bottleneck()
  local convs=nn.Sequential()
  convs:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
  convs:add(nn.SpatialBatchNormalization(64))
  convs:add(nn.ReLU(true))
  convs:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
  convs:add(nn.SpatialBatchNormalization(64))
  local shortcut=nn.Identity()
  return nn.Sequential():add(nn.ConcatTable():add(convs):add(shortcut)):add(nn.CAddTable(true))
end
]]





--model:add(layer(block,16,8,1))

function M.build_model(opt)
    --resnext
    local Convolution = nn.SpatialConvolution
    local Avg = nn.SpatialAveragePooling
    local ReLU = nn.ReLU
    local Max = nn.SpatialMaxPooling
    local SBatchNorm = nn.SpatialBatchNormalization

    --local function createModel(opt)
    local depth = opt.depth
    local shortcutType = opt.shortcutType or 'B'
    local iChannels

    -- The aggregated residual transformation bottleneck layer, Form (C)
  local function resnext_bottleneck_C(n, stride)
    local nInputPlane = iChannels
    iChannels = n * 4

    local D = math.floor(n * (opt.baseWidth/64))
    local C = opt.cardinality

    local s = nn.Sequential()
    s:add(nn.SpatialConvolution(nInputPlane,D*C,1,1,1,1,0,0))
    s:add(nn.SpatialBatchNormalization(D*C))
    s:add(nn.ReLU(true))
    s:add(nn.SpatialConvolution(D*C,D*C,3,3,stride,stride,1,1,C))
    s:add(nn.SpatialBatchNormalization(D*C))
    s:add(nn.ReLU(true))
    s:add(nn.SpatialConvolution(D*C,n*4,1,1,1,1,0,0))
    s:add(nn.SpatialBatchNormalization(n*4))

    return nn.Sequential()
    :add(nn.ConcatTable()
      :add(s)
      :add(shortcut(nInputPlane, n * 4, stride)))
    :add(nn.CAddTable(true))
    :add(ReLU(true))
  end

  local bottleneck
  if opt.bottleneckType == 'resnet' then 
    bottleneck = resnet_bottleneck
    print('Deploying ResNet bottleneck block')
    elseif opt.bottleneckType == 'resnext_B' then 
      bottleneck = resnext_bottleneck_B
      print('Deploying ResNeXt bottleneck block form B')
      elseif opt.bottleneckType == 'resnext_C' then 
        bottleneck = resnext_bottleneck_C
        print('Deploying ResNeXt bottleneck block form C (group convolution)')
      else
        error('invalid bottleneck type: ' .. opt.bottleneckType)
  end


 -- Creates count residual blocks with specified number of features
  local function layer(block, features, count, stride)
   local s = nn.Sequential()
   for i=1,count do
   s:add(block(features, i == 1 and stride or 1))
   end
   return s
  end

 --create model
  iChannels=128
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(3,64,3,3,1,1,1,1))
  model:add(nn.ReLU(true))
  model:add(nn.SpatialConvolution(64,128,3,3,1,1,1,1))
  model:add(nn.ReLU(true))
  --model:add(layer(opt.residual_blocks))
  model:add(layer(bottleneck,64,8,1))
  if opt.deconvolution_type == 'sub_pixel' then   ----train4将pixelshuffle 换成nearest upsample 减少棋盘噪声
    model:add( nn.SpatialUpSamplingNearest(2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.ReLU(true))
    model:add(nn.SpatialConvolution(256,128,3,3,1,1,1,1))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.ReLU(true))
    model:add( nn.SpatialUpSamplingNearest(2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.ReLU(true))
    model:add(nn.SpatialConvolution(128,32,3,3,1,1,1,1))
    model:add(nn.SpatialBatchNormalization(32))
  else
    model:add(nn.SpatialFullConvolution(64,64,4,4,2,2,1,1))
    model:add(nn.ReLU(true))
    model:add(nn.SpatialFullConvolution(64,64,4,4,2,2,1,1))
  end
  model:add(nn.ReLU(true))
  model:add(nn.SpatialConvolution(32,3,3,3,1,1,1,1))
  if opt.use_tanh then
    print('Use tanh to scale output')
    model:add(nn.Tanh())
    model:add(nn.AddConstant(1))
    model:add(nn.MulConstant(1/2))
  end
  utils.init_msra(model)
  utils.init_BN(model)

  return model
end

function M.build_discriminator(opt)
  local model=nn.Sequential()
  model:add(nn.SpatialConvolution(3,64,3,3,1,1,1,1))
  model:add(nn.LeakyReLU(0.2))
  model:add(nn.SpatialConvolution(64,64,3,3,2,2,1,1))
  model:add(nn.LeakyReLU(0.2))
  model:add(nn.SpatialBatchNormalization(64))
  model:add(nn.SpatialConvolution(64,128,3,3,1,1,1,1))
  model:add(nn.LeakyReLU(0.2))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.SpatialConvolution(128,128,3,3,2,2,1,1))
  model:add(nn.LeakyReLU(0.2))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.SpatialConvolution(128,256,3,3,1,1,1,1))
  model:add(nn.LeakyReLU(0.2))
  model:add(nn.SpatialBatchNormalization(256))
  model:add(nn.SpatialConvolution(256,256,3,3,2,2,1,1))
  model:add(nn.LeakyReLU(0.2))
  model:add(nn.SpatialBatchNormalization(256))
  model:add(nn.SpatialConvolution(256,512,3,3,1,1,1,1))
  model:add(nn.LeakyReLU(0.2))
  model:add(nn.SpatialBatchNormalization(512))
  model:add(nn.SpatialConvolution(512,512,3,3,2,2,1,1))
  model:add(nn.LeakyReLU(0.2))
  model:add(nn.SpatialBatchNormalization(512))
  model:add(nn.View(opt.train_size / 16 * opt.train_size / 16 *512))
  model:add(nn.Linear(opt.train_size / 16 * opt.train_size / 16 *512, 1024))
  model:add(nn.LeakyReLU(0.2))
  model:add(nn.Linear(1024, 1))
  model:add(nn.Sigmoid())

  utils.init_msra(model)
  utils.init_BN(model)
  return model
end
return M
