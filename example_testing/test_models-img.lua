require 'image'
require 'torch'


require 'super_resolution.InstanceNormalization'
require 'math'

local utils = require 'super_resolution.utils'
local gm = require 'graphicsmagick'
local cmd = torch.CmdLine()
-- Generic options
cmd:option('-img','/home/lixiaojie/Data/Set14/image_SRF_4/img_005_SRF_4_LR.png')
cmd:option('-img_hr','/home/lixiaojie/Data/Set14/image_SRF_4/img_005_SRF_4_HR.png')
--cmd:option('-img_out',"/home/lixiaojie/torch-srgan/paper_test_img/resnet")
-- Super-resolution options
-- Super-resolution options
cmd:option('-use_tanh', true)--true
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
-------------------------------------------------------------------------------------------

-- Calcul du PSNR entre 2 images
function PSNR(true_frame, pred)

   local eps = 0.0001
  -- print(true_frame:size())
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
   --print(torch.log(100)/torch.log(10))
   --PSNR
   if prediction_error>eps then
      prediction_error = 10*torch.log((255*255)/prediction_error)/torch.log(10)
   --prediction_error = 20*torch.log(255)-10*torch.log(prediction_error)
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
for file in lfs.dir(path) do
	if getExtension(file)=="t7" then
		print(file)
		local model_file=path..'/'..file

		-- Figure out the backend
		local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn)
         local criterion= nn.MSECriterion():type(dtype)
		-- Build the model
		print('Loading model ...')
		model = torch.load(model_file):type(dtype)
		print(dtype)
		print(model)
		if use_cudnn then cudnn.convert(model, cudnn) end
			cudnn.benchmark = false
    -- model.evaluate()
		--model:training()

		local img = gm.Image(opt.img):colorspace('RGB')
		local input = img:toTensor('float','RGB','DHW')
		--local img_hr= gm.Image(opt.img_hr):colorspace('RGB')
		--local img_origin = img_hr:toTensor('float','RGB','DHW')
    local img_origin = image.load(opt.img_hr)
    local origin = img_origin:clone()
    origin[{1,{},{}}] = img_origin[{3,{},{}}]
    origin[{3,{},{}}] = img_origin[{1,{},{}}]
    --local imggggg =image.load("/home/lixiaojie/Data/Set14/image_SRF_4/img_005_SRF_4_bicubic.png")

		input = torch.reshape(input,1,input:size(1),input:size(2),input:size(3))
		if opt.use_tanh then
			input = input:mul(2.0):add(-1.0)
		end
		local output = model:forward(input:type(dtype)):double()
    --output = output:add(1):div(2)
		local oimage = gm.Image(output[1]:float():clamp(0, 1),'RGB','DHW')
		-- oimage = oimage:add(1):div(2)
		--model:training()
    --print (img_origin:type())
    --print('process conv')
		local psnr=PSNR(output[1],img_origin)
		local ssim=SSIM(output[1],img_origin)
		--local psnr=PSNR(imggggg,img_origin)
		--local ssim=SSIM(imggggg,img_origin)
		--print(output[1]:size())
		--print(output[1]:type())

		local output_file="/mnt/lvm/xiaojie/workspace/torch-srgan/paper_test"..'/'..getFileName(file)..'_'..psnr..'_'..ssim..'.'..'png'
		print(psnr)
		print(ssim)
		oimage:save(output_file)
		end
	end
end

main()