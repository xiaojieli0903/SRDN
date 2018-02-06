require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
local utils = require 'super_resolution.utils'

local M = {}


function M.build_model(opt)
    --growth rate
    local growthRate = 32

    --dropout rate, set it to 0 to disable dropout, non-zero number to enable dropout and set drop rate
    local dropRate = 0

    --#channels before entering the first denseblock
    local nChannels = 2 * growthRate

    --compression rate at transition layers
    local reduction = 0.5

    --whether to use bottleneck structures
    local bottleneck = true

    --In our paper, a DenseNet-BC uses compression rate 0.5 with bottleneck structures
    --a default DenseNet uses compression rate 1 without bottleneck structures

    --N: #transformations in each denseblock
    local N = (opt.depth - 4)/3
    if bottleneck then N = N/2 end
    print('N='..N)
    --depth=10,N=1

    --non-bottleneck transformation
    local function addSingleLayer(model, nChannels, nOutChannels, dropRate)
      concate = nn.Concat(2)
      concate:add(nn.Identity())

      convFactory = nn.Sequential()
      convFactory:add(nn.SpatialBatchNormalization(nChannels))
      convFactory:add(nn.ReLU(true))
      convFactory:add(nn.SpatialConvolution(nChannels, nOutChannels, 3, 3, 1, 1, 1,1))
      if dropRate>0 then
        convFactory:add(nn.Dropout(dropRate))
      end
      concate:add(convFactory)
      model:add(concate)
    end


    --bottleneck transformation
    local function addBottleneckLayer(model, nChannels, nOutChannels, dropRate)
      concate = nn.Concat(2)
      concate:add(nn.Identity())

      local interChannels = 4 * nOutChannels

      convFactory = nn.Sequential()
      convFactory:add(nn.SpatialBatchNormalization(nChannels))
      convFactory:add(nn.ReLU(true))
      convFactory:add(nn.SpatialConvolution(nChannels, interChannels, 1, 1, 1, 1, 0, 0))
      if dropRate>0 then
        convFactory:add(nn.Dropout(dropRate))
      end

      convFactory:add(nn.SpatialBatchNormalization(interChannels))
      convFactory:add(nn.ReLU(true))
      convFactory:add(nn.SpatialConvolution(interChannels, nOutChannels, 3, 3, 1, 1, 1, 1))
      if dropRate>0 then
        convFactory:add(nn.Dropout(dropRate))
      end

      concate:add(convFactory)
      model:add(concate)
    end

    if bottleneck then
      add = addBottleneckLayer
    else
      add = addSingleLayer
    end

    local function addTransition(model, nChannels, nOutChannels, dropRate)
      model:add(nn.SpatialBatchNormalization(nChannels))
      model:add(nn.ReLU(true))
      model:add(nn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0))
      if dropRate>0 then
        model:add(nn.Dropout(dropRate))
      end
     --model:add(nn.SpatialAveragePooling(2, 2))
    end
------------------------------------------------------------------------------------------
    --构建网络
    model = nn.Sequential()

    --first conv before any dense blocks -nChannels = 2 * growthRate=2*12=24
    model:add(nn.SpatialConvolution(3, nChannels, 3,3, 1,1, 1,1))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(nn.ReLU(true))

    --1st dense block and transition
    for i=1, N do 
      add(model, nChannels, growthRate, dropRate)
      nChannels = nChannels + growthRate
    end
    addTransition(model, nChannels, math.floor(nChannels*reduction), dropRate)
    nChannels = math.floor(nChannels*reduction)

    --2nd dense block and transition
    for i=1, N do
      add(model, nChannels, growthRate, dropRate)
      nChannels = nChannels + growthRate
    end
    addTransition(model, nChannels, math.floor(nChannels*reduction), dropRate)
    nChannels = math.floor(nChannels*reduction)

    --3rd dense block
    for i=1, N do
      add(model, nChannels, growthRate, dropRate)
      nChannels = nChannels + growthRate
    end
      --[[
    addTransition(model, nChannels, math.floor(nChannels*reduction), dropRate)
    nChannels = math.floor(nChannels*reduction)
    --4rd dense block
  
    for i=1, N do
      add(model, nChannels, growthRate, dropRate)
      nChannels = nChannels + growthRate
    end
]]


    if opt.deconvolution_type == 'sub_pixel' then   ----train4将pixelshuffle 换成nearest upsample 减少棋盘噪声
    model:add( nn.SpatialUpSamplingNearest(2))
   -- model:add(nn.SpatialUpSamplingBilinear(2))
    model:add(nn.SpatialBatchNormalization(352))
    model:add(nn.ReLU(true))
    model:add(nn.SpatialConvolution(352,64,3,3,1,1,1,1))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(nn.ReLU(true))
    model:add( nn.SpatialUpSamplingNearest(2))
   -- model:add(nn.SpatialUpSamplingBilinear(2))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(nn.ReLU(true))
    model:add(nn.SpatialConvolution(64,32,3,3,1,1,1,1))
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

  --print(model)
  return model
end

return M
