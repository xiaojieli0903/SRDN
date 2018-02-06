require 'nn'
require 'nngraph'

local utils = require 'super_resolution.utils'

local M = {}


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

local function layer(count)
      local s=nn.Sequential()
      for i=1,count do
        s:add(bottleneck())
      end
      return s
    end

function M.build_model(opt)
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(3,64,3,3,1,1,1,1))
  model:add(nn.ReLU(true))
  model:add(layer(opt.residual_blocks))
  if opt.deconvolution_type == 'sub_pixel' then   ----train4将pixelshuffle 换成nearest upsample 减少棋盘噪声
    model:add( nn.SpatialUpSamplingNearest(2))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(nn.ReLU(true))
    model:add(nn.SpatialConvolution(64,32,3,3,1,1,1,1))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(nn.ReLU(true))
    model:add( nn.SpatialUpSamplingNearest(2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(nn.ReLU(true))
    model:add(nn.SpatialConvolution(32,32,3,3,1,1,1,1))
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