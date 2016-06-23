require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'cutorch'
require 'cunn'
require 'gnuplot'

require 'GetBatchData'
require 'GetDataFromCsv'
require 'printing'


function save_model(model,path)
	path=path or "./Save/model2.t7"
	print("Saved at : "..path)
	model:cuda()
	parameters, gradParameters = model:getParameters()
	local lightModel = model:clone('weight','bias','running_mean','running_std'):double()
	torch.save(path,model)
end

function train_epochs(train_list,test_list, BatchSize,classes,image_width,image_height, criterion, Timnet, LR, MaxEpoch)
	
	local list_error_train={}
	local list_error_test={}
	local list_loss_train={}
	local list_loss_test={}
	local nbBatch=math.floor(#train_list.data/BatchSize)+1
	local timer = torch.Timer()

	for epochs=0, MaxEpoch do
		print(" Epochs :  "..epochs)
		print(nbBatch .. " Batchs per Epochs ")
		for numBatch=1, nbBatch do	
			
			if numBatch%500==0 then
			 print('Numbatch : '..numBatch.." - time : "..os.date("%X"))
			 end
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
			local visu=criterion:backward(Timnet.output, trainData.label)
			Timnet:backward(trainData.data, visu)
			Timnet:updateParameters(LR)
		end 
		save_model(Timnet,"./Save/Savemodele22.t7")
		--Testing
		error_train, loss_train=print_performance(train_list, BatchSize , 					Timnet,criterion, classes, image_width, image_height,"TRAIN")
		error_test, loss_test=print_performance(test_list, BatchSize , Timnet,criterion, 						classes, image_width, image_height,"VALID")


		table.insert(list_error_train,error_train)
		table.insert(list_error_test,error_test)
		table.insert(list_loss_train,loss_train)
		table.insert(list_loss_test,loss_test)

		show_figure(list_error_train,list_error_test,list_loss_train,list_loss_test)
	end
end



-- Training Parameters -------------------------------------------------------

local LR=0.01
local MaxEpoch=10
local classes={"c0","c1","c2","c3","c4","c5","c6","c7","c8","c9"}
local datapath="/home/lesort/TrainTorch/Kaggle/imgs/train/"
local BatchSize=5
local image_width=200
local image_height=200
local save_name='./Save/model2.t7'
local reload=false

--!--local MaxBatch=1
--!--local Testsize=1000
------------------------------------------------------------------------------

if reload==true then
	Timnet = torch.load(modele_name):double()
	print('Timnet\n' .. Timnet:__tostring());
else
	modele_file='./models/modele2'
	print("------------------------------------------------------------------------------")
	print("------------------------------------------------------------------------------")
	print("-----------------------------"..modele_file.."--------------------------------")
	print("------------------------------------------------------------------------------")
	print("------------------------------------------------------------------------------")
	require(modele_file)
	Timnet = getNet(image_width,image_height)
end
Timnet=Timnet:cuda()
criterion = nn.ClassNLLCriterion() -- a negative log-likelihood for multi-class classification
criterion = criterion:cuda()
----------------------------------------------------------------

--local trainList, testList= GetImageTrainAndTestList(datapath, classes, 80)
local csv="/home/lesort/TrainTorch/Kaggle/driver_imgs_list.csv"
local trainList, testList=GetTestAndTrain(csv,datapath, 80)

trainList=shuffleDataList(trainList)
testList=shuffleDataList(testList)

-- Epoch Training -------------------------------------------------------
train_epochs(trainList, testList, BatchSize,classes,image_width,image_height, criterion, Timnet, LR, MaxEpoch)


---- Save Model ---------------------------------------------------------------------
--save_model(Timnet)
-------------------------------------------------------------------------------------

