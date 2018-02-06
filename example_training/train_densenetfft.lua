require 'torch'
require 'optim'
require 'image'
require 'xlua'
require 'super_resolution.DataLoader'

local utils = require 'super_resolution.utils'
local models = require 'super_resolution.models'
local models_new = require 'super_resolution.models_new'
local models_densenet=require 'super_resolution.models_densenet'
local models_densenet_pretrain=require 'densenet_pretrain'
local gm = require 'graphicsmagick'
local c = require 'trepl.colorize'

local cmd = torch.CmdLine()
-- Generic options
cmd:option('-h5_file','/mnt/lvm/xiaojie/imagenet-val-192_multi.h5')
--test1用了加强及细节后的数据集，test2用了原数据集 
cmd:option('-val_img','./imgs/comic_input.bmp')
cmd:option('-val_output','./Retest/densenetfft_1-6_conv2_2/val/')
cmd:option('-residual_blocks', 8)
cmd:option('-deconvolution_type','sub_pixel','sub_pixel|fullconvolution')
cmd:option('-debug', false)
-- Super-resolution options
cmd:option('-loss', 'percep', 'pixel|percep')
cmd:option('-percep_layer', 'conv2_2', 'conv2_2|conv5_4')
cmd:option('-percep_model', './models/VGG19.t7')
cmd:option('-use_tanh', true)
-- Optimization
cmd:option('-num_epoch', 50)
cmd:option('-batch_size',8)
cmd:option('-learning_rate', 1e-3)
cmd:option('-lr_decay_every', 10)
cmd:option('-lr_decay_factor', 0.95)

cmd:option('-beta1', 0.9)
cmd:option('-weight_decay',1e-3)
cmd:option('-random_flip', true)
-- Checkpointing
cmd:option('-resume_from_checkpoint','')--'./checkpoint_densenet40+fft1-8/densenet_fft_63.t7')
cmd:option('-resume_epoch',0)
cmd:option('-checkpoint_name', './Retest/densenetfft_1-6_conv2_2')
cmd:option('-checkpoint_every', 1)
-- Backend options
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
cmd:option('-backend', 'cuda')
-- dense net options
cmd:option('-depth', 40)
cmd:option('-dataset', 'cifar10')

cmd:option('-pixel_loss_weight', 0)
cmd:option('-percep_loss_weight', 1)


function main()
	local opt = cmd:parse(arg)

	-- Figure out the backend
	local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn)
	print('dtype')
	print(dtype)

	-- Build the model
	local model = nil
	if opt.resume_from_checkpoint ~= '' then
		print('Loading checkpoint from ' .. opt.resume_from_checkpoint)
		model = torch.load(opt.resume_from_checkpoint):type(dtype)
	else
		print('Initializing model from scratch')
		model = models_densenet.createModel(opt):type(dtype)
		--model=models_densenet_pretrain.build_densenet_pretrain()
	end
	--if use_cudnn then cudnn.convert(model, cudnn) end
	model:training()
	print(model)

	local criterion_percep = nn.MSECriterion():type(dtype)
	local criterion_pixel = nn.MSECriterion():type(dtype)

	local loader = DataLoader(opt)
	local params, grad_params = model:getParameters()

	-- Load percep model
	local percep_model = nil
	local params_percep, grad_params_percep = nil
	if opt.loss == 'percep' then
		print('Training with perceptual loss of layer ' .. opt.percep_layer)
		print('Loading VGG19 model')
		percep_model = torch.load(opt.percep_model)
		if opt.percep_layer == 'conv2_2' then
			for _ = 1,27 do
				percep_model:remove()
			end
		end
		percep_model:type(dtype)
		if use_cudnn then cudnn.convert(percep_model, cudnn) end
		percep_model:evaluate()
		params_percep, grad_params_percep = percep_model:getParameters()
		print(percep_model)
	end

	local function f(x)
		assert(x == params)
		grad_params:zero()

		--- Load data and label
		local x, y, z = loader:getBatch('train')
		if opt.use_tanh then
			x = x:mul(2.0):add(-1.0)
		end
		x, y ,z= x:type(dtype), y:type(dtype), z:type(dtype)
		-- Run model forward
		local out = model:forward(x)
		local grad_out = nil
		local grad_out_pix=nil
		local grad_out_percep=nil

		-- Compute loss and loss gradient
		local loss = 0
		local loss_pix = 0
		local loss_percep = 0
		if opt.loss == 'pixel' then
			loss_pix = criterion_pixel:forward(out, y)
			loss_pix = loss_pix*opt.pixel_loss_weight
		    grad_out_pix = criterion_pixel:backward(out, y)
		    if grad_out then
		    	grad_out:add(opt.pixel_loss_weight, grad_out_pix)
		    else
		    	grad_out_pix:mul(opt.pixel_loss_weight)
		    	grad_out = grad_out_pix
		    end

		elseif opt.loss == 'percep' then
			grad_params_percep:zero()
			local input_real_percep = utils.vgg_preprocess(y)
			local input_sr_percep = utils.vgg_preprocess(out)
			local output_real_percep = percep_model:forward(input_real_percep):clone()
			local output_sr_percep = percep_model:forward(input_sr_percep)
			loss_percep = criterion_percep:forward(output_sr_percep, output_real_percep)
			loss_percep =loss_percep *opt.percep_loss_weight
			local percep_grad_mse_out = criterion_percep:backward(output_sr_percep, output_real_percep)
			local percep_grad_out = percep_model:backward(input_sr_percep, percep_grad_mse_out)
			local z= z:expandAs(y)
			local percep_grad_out_ =torch.cmul(percep_grad_out,z)
			grad_out = percep_grad_out_:mul(255.0):index(2, torch.LongTensor{3,2,1})
	       -- grad_out_percep:mul(opt.percep_loss_weight)
	        --grad_out:add(grad_out_percep)
	        loss=loss_percep+loss_pix--pixel+percep loss
        end
		  model:backward(x, grad_out)
		  grad_params:add(opt.weight_decay, params)
		return loss, grad_params
	end


	local optim_state = {learningRate=opt.learning_rate,
						beta1 = opt.beta1,
						weightDecay = opt.weight_decay,
						}
	local train_loss_history = {}
	local val_loss_history = {}
	
	-- Training
	for epoch = opt.resume_epoch + 1, opt.num_epoch do
		local tic = torch.tic()
		print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batch_size .. ' lr = ' .. optim_state.learningRate .. ']')
		local loss_epoch = 0
		for t = 1, loader.num_minibatches['train'] do
			local _, loss_batch = optim.adam(f, params, optim_state)
			loss_epoch = loss_epoch + loss_batch[1]
			if opt.debug then
				print(string.format('Epoch %d, Iteration %d / %d, loss = %f ', 
							epoch, t, loader.num_minibatches['train'], loss_batch[1]), optim_state.learningRate)
			else
				xlua.progress(t, loader.num_minibatches['train'])
			end
		end
		loss_epoch = loss_epoch / loader.num_minibatches['train']
		print(('Train loss: '..c.cyan'%.6f'..' \t time: %.2f s'):format(loss_epoch, torch.toc(tic)))
        table.insert(train_loss_history, loss_epoch)

		-- Testing
        if epoch % opt.checkpoint_every == 0 then
        	-- Check loss on the validation set
        	loader:reset('val')
			model:evaluate()
			local val_loss = 0
			print('Running on validation set')
			local val_batches = loader.num_minibatches['val']
			print('val batches')
			--print(val_batches)
			for j = 1, val_batches do
				local x, y,z = loader:getBatch('val')
				if opt.use_tanh then
					x = x:mul(2.0):add(-1.0)
				end
				x, y,z= x:type(dtype), y:type(dtype),z:type(dtype)
				local out = model:forward(x)
				local loss = 0
				if opt.loss == 'pixel' then
					loss = criterion_pixel:forward(out, y)
				elseif opt.loss == 'percep' then
					local input_real_percep = utils.vgg_preprocess(y)
					local input_sr_percep = utils.vgg_preprocess(out)
					local output_real_percep = percep_model:forward(input_real_percep):clone()
					local output_sr_percep = percep_model:forward(input_sr_percep)
					loss = criterion_percep:forward(output_sr_percep, output_real_percep)
				end
				val_loss = val_loss + loss
			end
			val_loss = val_loss / val_batches
			print(('Val loss: '..c.cyan'%.6f'):format(val_loss))
			table.insert(val_loss_history, val_loss)
			-- Save log
        	local log = {opt = opt, 
						train_loss_history = train_loss_history,
						val_loss_history = val_loss_history,
						}
			local filename = string.format('%s.json',opt.checkpoint_name)
			paths.mkdir(paths.dirname(filename))
			utils.write_json(filename, log)

			-- Check performance on the val img
			local val_img = gm.Image(opt.val_img):colorspace('RGB')
			local input = val_img:toTensor('float','RGB','DHW')
			input = torch.reshape(input,1,input:size(1),input:size(2),input:size(3))
			if opt.use_tanh then
				input = input:mul(2.0):add(-1.0)
			end
			local output = model:forward(input:type(dtype))
			local image = gm.Image(output[1]:float(),'RGB','DHW')
			image:save(opt.val_output .. 'outputs_' .. epoch .. '.png')

			-- Save model
			model:clearState()
			if use_cudnn then 
				cudnn.convert(model, nn)
			end
			model:float()
			filename = string.format('%s_%d.t7',opt.checkpoint_name,epoch)
			torch.save(filename,model)
			model:type(dtype)
			if use_cudnn then
				cudnn.convert(model,cudnn)
			end
			params, grad_params = model:getParameters()
		end

		if opt.lr_decay_every > 0 and epoch % opt.lr_decay_every == 0 then
                  local new_lr = opt.lr_decay_factor * optim_state.learningRate
                  optim_state = {learningRate = new_lr}
                end

		model:training()
	end
end

main()


