require 'image'
require 'torch'


require 'super_resolution.InstanceNormalization'
require 'math'

local utils = require 'super_resolution.utils'
local gm = require 'graphicsmagick'
local cmd = torch.CmdLine()
-- Generic options
cmd:option('-test_dataset',"/home/lixiaojie/Data/Set5/image_SRF_4/")
-- Super-resolution options
cmd:option('-use_tanh', false)--true
-- Checkpointing
cmd:option('-model_path', '/mnt/lvm/xiaojie/workspace/torch-srgan/checkpoints/checkpoint_resnet18_pixel')
-- Backend options
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
cmd:option('-backend', 'cuda')
-------------------------------------------------------------------------------------------------
local lfs = require"lfs"
  --获取文件名  
function getFileName(str)  
    local idx = str:match(".+()%.%w+$")  
    if(idx) then  
        return str:sub(1, idx-1)  
    else  
        return str  
    end  
end  
  
--获取扩展名  
function getExtension(str)  
    return str:match(".+%.(%w+)$")  
end  

------------------------------------------------------------------------------------------

-- Calcul du PSNR entre 2 images
function PSNR(true_frame, pred)

   local eps = 0.0001
   local prediction_error = 0
   for i = 1,pred:size(1) do
     for j = 1, pred:size(2) do
        for k = 1, pred:size(3) do
            prediction_error = prediction_error +(pred[i][j][k] - true_frame[i][j][k])^2
        end
     end
   end
   --MSE
   prediction_error=128*128*prediction_error/(pred:size(2)*pred:size(3))
   --PSNR
   if prediction_error>eps then
      prediction_error = 10*torch.log((255*255)/prediction_error)/torch.log(10)
   else
      prediction_error = 10*torch.log((255*255)/ eps)/torch.log(10)
   end
   return prediction_error
end

--------------------------------------------------------------------------------
-- Calcul du SSIM
function SSIM(img1, img2)

   -- place images between 0 and 255.
   img1:add(1):div(2):mul(255)
   img2:add(1):div(2):mul(255)

   local K1 = 0.01;
   local K2 = 0.03;
   local L = 255;

   local C1 = (K1*L)^2;
   local C2 = (K2*L)^2;
   local window = image.gaussian(11, 1.5/11,0.0708);

   local window = window:div(torch.sum(window));

   local mu1 = image.convolve(img1, window,'full')
   local mu2 = image.convolve(img2, window, 'full')

   local mu1_sq = torch.cmul(mu1,mu1);
   local mu2_sq = torch.cmul(mu2,mu2);
   local mu1_mu2 = torch.cmul(mu1,mu2);

   local sigma1_sq = image.convolve(torch.cmul(img1,img1),window,'full')-mu1_sq
   local sigma2_sq = image.convolve(torch.cmul(img2,img2),window,'full')-mu2_sq
   local sigma12 =  image.convolve(torch.cmul(img1,img2),window,'full')-mu1_mu2

   local ssim_map = torch.cdiv( torch.cmul((mu1_mu2*2 + C1),(sigma12*2 + C2)) ,
     torch.cmul((mu1_sq + mu2_sq + C1),(sigma1_sq + sigma2_sq + C2)));
   local mssim = torch.mean(ssim_map);
   return mssim
end




function main()
  local opt = cmd:parse(arg)
  local path=opt.model_path
  local img_path = opt.test_dataset
  for file in lfs.dir(path) do
    if getExtension(file)=="t7" then
      print("-------model--------"..file)
      local model_file=path..'/'..file
      -- Figure out the backend
      local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn)
      local criterion= nn.MSECriterion():type(dtype)
      local psnr_all = 0
      local ssim_all = 0
      -- Build the model
      print('Loading model ...')
      model = torch.load(model_file):type(dtype)
      if use_cudnn then cudnn.convert(model, cudnn) end
        cudnn.benchmark = false
      model:evaluate()
      for im in lfs.dir(img_path) do
        if getExtension(im)=="png" and im:match("H.*") == "HR.png" then
          local img_hr= img_path..im
          local img_lr = img_path..im:match("^.-F").."_4_LR.png"
          -- print(im)
          local inp_img = gm.Image(img_lr):colorspace('RGB')
          local input = inp_img:toTensor('float','RGB','DHW')
          local img_origin = image.load(img_hr)

          input = torch.reshape(input,1,input:size(1),input:size(2),input:size(3))
          if opt.use_tanh then
            input = input:mul(2.0):add(-1.0)
          end
          local output = model:forward(input:type(dtype)):double()
          local out_img = gm.Image(output[1]:float():clamp(0, 1),'RGB','DHW')
          if img_origin:size()[1]==1 then
            local origin = torch.zeros(output[1]:size()[1],output[1]:size()[2],output[1]:size()[3])
            origin[{1,{},{}}] = img_origin[{1,{},{}}]
            origin[{2,{},{}}] = img_origin[{1,{},{}}]
            origin[{3,{},{}}] = img_origin[{1,{},{}}]
            img_origin = origin
          end
          local psnr=PSNR(output[1],img_origin)
          local ssim=SSIM(output[1],img_origin)
          local output_file=opt.model_path..'/'..getFileName(file)..'_'..psnr..'_'..ssim..'_'..im
          psnr_all = psnr_all + psnr
          ssim_all = ssim_all + ssim
          -- print(psnr)
          -- print(ssim)
          -- print(output_file)
          out_img:save(output_file)
        end
      end
      print("-------panr--------"..psnr_all/14.0)
      print("-------ssim--------"..ssim_all/14.0)
    end
  end
end

main()