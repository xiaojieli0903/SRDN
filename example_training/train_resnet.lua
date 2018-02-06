require 'torch'
require 'optim'
require 'image'
require 'xlua'
require 'super_resolution.DataLoader_origin'

local utils = require 'super_resolution.utils'
local models = require 'super_resolution.models'
local models_resnet = require 'super_resolution.models_resnet'
local models_densenet=require 'super_resolution.models_densenet'
local gm = require 'graphicsmagick'
local c = require 'trepl.colorize'

local cmd = torch.CmdLine()
-- Generic options
cmd:option('-h5_file','/mnt/lvm/xiaojie/workspace/imagenet-val-19.h5')
--test1用了加强及细节后的数据集，test2用了原数据集
cmd:option('-val_img','./images/imgs/comic_input.bmp')
cmd:option('-val_output','./checkpoints/checkpoint_resnet18_pixel/')
cmd:option('-residual_blocks', 18)
cmd:option('-deconvolution_type','sub_pixel','sub_pixel|fullconvolution')
cmd:option('-debug', true)
-- Super-resolution options
cmd:option('-loss', 'pixel', 'pixel|percep')
cmd:option('-percep_layer', 'conv5_4', 'conv2_2|conv5_4')
cmd:option('-percep_model', './models/VGG19.t7')
cmd:option('-use_tanh', true)
-- Optimization
cmd:option('-num_epoch', 100)
cmd:option('-batch_size',16)
cmd:option('-learning_rate', 1e-3)
cmd:option('-lr_decay_every', 10)
cmd:option('-lr_decay_factor', 0.95)

cmd:option('-beta1', 0.9)
cmd:option('-weight_decay',1e-3)
cmd:option('-random_flip', true)
-- Checkpointing
cmd:option('-resume_from_checkpoint', '')
cmd:option('-resume_epoch',0)
cmd:option('-checkpoint_name', './checkpoints/checkpoint_resnet18_pixel/resnet18_pixel')
cmd:option('-checkpoint_every', 5)
-- Backend options
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
cmd:option('-backend', 'cuda')
-- dense net options
cmd:option('-depth', 40)
cmd:option('-dataset', 'cifar10')



function main()
	local opt = cmd:parse(arg)

	-- Figure out the backend
	local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn)

	-- Build the model
	local model = nil
	if opt.resume_from_checkpoint ~= '' then
		print('Loading checkpoint from ' .. opt.resume_from_checkpoint)
		model = torch.load(opt.resume_from_checkpoint):type(dtype)
	else
		print('Initializing model from scratch')
		model = models_resnet.build_model(opt):type(dtype)

	end
	--if use_cudnn then cudnn.convert(model, cudnn) end
	model:training()
	print(model)

	local criterion = nn.MSECriterion():type(dtype)

	local loader = DataLoader(opt)
	local params, grad_params = model:getParameters()
    print(params:size(1))
    assert(1)
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

		-- Load data and label
		local x, y, z = loader:getBatch('train')
		if opt.use_tanh then
			x = x:mul(2.0):add(-1.0)
		end
		x, y= x:type(dtype), y:type(dtype)
		--print(x:size())
		--print(y:size())
		--print(z:size())
		-- Run model forward
		local out = model:forward(x) -- -1~1
		--print(out:size())
		local grad_out = nil

		-- Compute loss and loss gradient
		local loss = 0
		if opt.loss == 'pixel' then
			loss = criterion:forward(out, y)
			grad_out = criterion:backward(out, y)
		elseif opt.loss == 'percep' then
			grad_params_percep:zero()
			local input_real_percep = utils.vgg_preprocess(y)
			local input_sr_percep = utils.vgg_preprocess(out)
			local output_real_percep = percep_model:forward(input_real_percep):clone()
			local output_sr_percep = percep_model:forward(input_sr_percep)
			loss = criterion:forward(output_sr_percep, output_real_percep)
			local percep_grad_out = criterion:backward(output_sr_percep, output_real_percep)
			local percep_grad_in = percep_model:backward(input_sr_percep, percep_grad_out)
			--local z= z:expandAs(y)
			--local percep_grad_in_ =torch.cmul(percep_grad_in,z)
			--percep_grad_in.size(10*3*192*192)
			--*weight
			--print('percep_grad_in')
			--print(percep_grad_out:size())
			grad_out = percep_grad_in:mul(255.0):index(2, torch.LongTensor{3,2,1})
		end
		model:backward(x, grad_out)
		return loss, grad_params
	end


	local optim_state = {learningRate=opt.learning_rate,
						beta1 = opt.beta1,
						weightDecay = opt.weight_decay,
						}
	local train_loss_epoch_history = {}
	local val_loss_history = {}
	local train_loss_batch_history = {}
	
	-- Training
	for epoch = opt.resume_epoch + 1, opt.num_epoch do
		local tic = torch.tic()
		print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batch_size .. ' lr = ' .. optim_state.learningRate .. ']')
		local loss_epoch = 0
		for t = 1, loader.num_minibatches['train'] do
			local _, loss_batch = optim.adam(f, params, optim_state)
			loss_epoch = loss_epoch + loss_batch[1]
            table.insert(train_loss_batch_history, loss_batch[1])
			if opt.debug then
				print(string.format('Epoch %d, Iteration %d / %d, loss = %f ', 
							epoch, t, loader.num_minibatches['train'], loss_batch[1]), optim_state.learningRate)
			else
				xlua.progress(t, loader.num_minibatches['train'])
			end
		end
		loss_epoch = loss_epoch / loader.num_minibatches['train']
		print(('Train loss: '..c.cyan'%.6f'..' \t time: %.2f s'):format(loss_epoch, torch.toc(tic)))
        table.insert(train_loss_epoch_history, loss_epoch)

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
				local x, y = loader:getBatch('val')
			--	print(x:shape())
			--	print(y:shape())
			--	print(x[1][1])
			--	print(y[1][1])
				if opt.use_tanh then
					x = x:mul(2.0):add(-1.0)
				end
				x, y = x:type(dtype), y:type(dtype)
				local out = model:forward(x)
				local loss = 0
				if opt.loss == 'pixel' then
					loss = criterion:forward(out, y)
				elseif opt.loss == 'percep' then
					local input_real_percep = utils.vgg_preprocess(y)
					local input_sr_percep = utils.vgg_preprocess(out)
					local output_real_percep = percep_model:forward(input_real_percep):clone()
					local output_sr_percep = percep_model:forward(input_sr_percep)
					loss = criterion:forward(output_sr_percep, output_real_percep)
				end
				val_loss = val_loss + loss
			end
			val_loss = val_loss / val_batches
			print(('Val loss: '..c.cyan'%.6f'):format(val_loss))
			table.insert(val_loss_history, val_loss)
			-- Save log
        	local log = {opt = opt, 
						train_loss_epoch_history = train_loss_epoch_history,
						val_loss_history = val_loss_history,
						train_loss_batch_history = train_loss_batch_history,
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


