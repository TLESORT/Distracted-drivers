require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'cutorch'
require 'cunn'

require 'GetBatchData'


function save_model(model,path)
	path=path or "model2.t7"
	model:cuda()
	parameters, gradParameters = model:getParameters()
	local lightModel = model:clone('weight','bias','running_mean','running_std')
	torch.save(path,lightModel)
end

function train_Random_batchs(im_list, BatchSize,MaxBatch,classes,image_width,image_height, criterion, Timnet, LR)
	for NumBatch=1, MaxBatch do
		-- Data ---------------------------------------------------------------
		trainData=getRandomDataTrain(im_list, BatchSize,image_width,image_height)
		trainData.data=trainData.data:cuda()
		trainData.label=trainData.label:cuda()

		local input=trainData.data
		local output=trainData.label
		-----------------------------------------------------------------------
		print('loss '.. criterion:forward(Timnet:forward(input), output).." batch : "..NumBatch)
		-- Stochastique Gradient Descent
		Timnet:zeroGradParameters()
		Timnet:backward(input, criterion:backward(Timnet.output, output))
		Timnet:updateParameters(LR)
	end
end

function train_epochs(train_list,test_list, BatchSize,classes,image_width,image_height, criterion, Timnet, LR, MaxEpoch)
	
	local nbBatch=math.floor(#train_list.data/BatchSize)+1
	for epochs=0, MaxEpoch do
		print(" Epochs :  "..epochs)
		print(nbBatch .. " Batchs per Epochs ")
		for numBatch=1, nbBatch do	
			-- Data ---------------------------------------------------------------
			trainData=getBatch(train_list, BatchSize,image_width,image_height,numBatch-1,'TRAIN')
			trainData.data=trainData.data:cuda()
			trainData.label=trainData.label:cuda()
			--image.display(trainData.data)
			local predictions=Timnet:forward(trainData.data)
			local loss=criterion:forward(predictions, trainData.label)
			-----------------------------------------------------------------------
			--!--print('loss '.. loss .." batch : "..numBatch.."\n")
			-- Stochastique Gradient Descent
			Timnet:zeroGradParameters()
			Timnet:backward(trainData.data, criterion:backward(Timnet.output, trainData.label))
			Timnet:updateParameters(LR)
		end 
		save_model(Timnet,"modele16.t7")
		--Testing
		print_performance(train_list, BatchSize , Timnet,criterion, classes, image_width, image_height,"TRAIN")
		print_performance(test_list, BatchSize , Timnet,criterion, classes, image_width, image_height,"VALID")
	end
end

function count_error(truth, prediction,class_Accuracy)
	local errors=0
	maxs, indices = torch.max(prediction,2)
	for i=1,truth:size(1) do
	    if truth[i] ~= indices[i][1] then
		--print(truth[i].." vs "..indices[i][1])
		errors = errors + 1
	    else
		--print(truth[i].." = "..indices[i][1])
		class_Accuracy[truth[i]] = class_Accuracy[truth[i]] + 1
	    end
	end
	return errors, class_Accuracy
end

function print_performance(im_list, Batchsize, net, criterion, classes, image_width, image_height,Type)
	local correct = 0
	local nbBatch=math.floor(#im_list.label/Batchsize)+1
	local class_Accuracy = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	local errors_tot=0
	local loss_tot=0
	print(nbBatch.. " Test Batchs ")
	for i=1, nbBatch do
	    local Data=getBatch(im_list, Batchsize, image_width, image_height, i-1, Type)
	    Data.data= Data.data:cuda()
	    Data.label= Data.label:cuda()
	    local groundtruth = Data.label
	    --image.display(Data.data)
	    local prediction = net:forward(Data.data)--one image per batch
	    local loss=criterion:forward(prediction, groundtruth)
	    loss_tot=loss_tot+loss
	    --!--local confidences, indices = torch.sort(prediction, true)  --true -> sort in descending order
	    errors, class_Accuracy=count_error(groundtruth, prediction,class_Accuracy)
	    errors_tot=errors_tot+errors
	end
	print("----------------"..Type.."-----------------")
	print('loss_tot '.. loss_tot/nbBatch) 
	print("nb errors : ".. errors_tot)
	print("errors : "..100*errors_tot/(nbBatch*Batchsize).. ' % ')
	--print("Classe Accuracy: ")
	--for i=1,#classes do
	--    print(classes[i], 100*class_Accuracy[i]/(nbBatch*Batchsize) .. ' %')
	--end
	print("-------------------------------------")
end

-- Training Parameters -------------------------------------------------------

local LR=0.01
local MaxEpoch=30
local classes={"c0","c1","c2","c3","c4","c5","c6","c7","c8","c9"}
local datapath="/home/lesort/TrainTorch/Kaggle/imgs/train/"
local BatchSize=5
local image_width=200
local image_height=200
local modele_name='model2.t7'
local reload=false

--!--local MaxBatch=1
--!--local Testsize=1000
------------------------------------------------------------------------------

if reload==true then
	Timnet = torch.load(modele_name):double()
	print('Timnet\n' .. Timnet:__tostring());
else
	modele_file='modele2'
	require(modele_file)
	Timnet = getNet(image_width,image_height)
end
Timnet=Timnet:cuda()
criterion = nn.ClassNLLCriterion() -- a negative log-likelihood for multi-class classification
criterion = criterion:cuda()
----------------------------------------------------------------

local trainList, testList= GetImageTrainAndTestList(datapath, classes, 80)

trainList=shuffleDataList(trainList)
testList=shuffleDataList(testList)

-- Mini-Batch Training -------------------------------------------------------
--[[
--!--train_Random_batchs(trainList,
		BatchSize,
		MaxBatch,
		classes,image_width,image_height, criterion, Timnet, LR)
--]]

-- Epoch Training -------------------------------------------------------
train_epochs(trainList, testList, BatchSize,classes,image_width,image_height, criterion, Timnet, LR, MaxEpoch)


------------------------------------------------------------------------------
--trainer = nn.StochasticGradient(Timnet, criterion)
--trainer.learningRate = 0.01
--trainer.maxIteration = 20000 -- just do n epochs of training.
--trainer:train(trainData)
------------------------------------------------------------------------------


-- Testing -------------------------------------------------------
--!--local testData=getRandomDataTest(testList, Testsize,image_width,image_height)
--!--testData.data=testData.data:cuda()
--!--testData.label=testData.label:cuda()

--!--print_performance(testList, BatchSize , Timnet,criterion, classes, image_width, image_height,'VALID')
------------------------------------------------------------------


---- Save Model ---------------------------------------------------------------------
save_model(Timnet)
-------------------------------------------------------------------------------------

