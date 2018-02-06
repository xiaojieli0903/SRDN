require 'image'
require 'torch'
require 'math'

local gm = require 'graphicsmagick'
local cmd = torch.CmdLine()
local util = require 'util.util'

-- cmd:option('-img_hr','/home/lixiaojie/Data/Set14/comic_HR.jpg')
cmd:option('-img_hr','/home/lixiaojie/Data/Set14/image_SRF_4/img_005_SRF_4_HR.png')
cmd:option('-img_dir','/home/lixiaojie/workplace/pix2pix/test_img/')
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
   img1:mul(255)
   img2:mul(255)

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
  for file in lfs.dir(opt.img_dir) do
    if getExtension(file)=="png" then
      print(file)
      local img_file=opt.img_dir..file
      local img_sr = image.load(img_file)--0-1     
      local img_hr = image.load(opt.img_hr)--0-1
      local psnr=PSNR(img_sr,img_hr)
      local ssim=SSIM(img_sr,img_hr)
     -- local output_file=opt.output_dir..'/'..getFileName(file)..'5'..'_'..psnr..'_'..ssim..'.'..'png'
      print(psnr)
      print(ssim)
    end
  end
end
main()