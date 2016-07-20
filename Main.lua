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
	print("Saved at : "..path)
	model:cuda()
	parameters, gradParameters = model:getParameters()
	local lightModel = model:clone('weight','bias','running_mean','running_std'):double()
	torch.save(path,lightModel)
end

function bootStrap(train_list,test_list, BatchSize,classes,image_width,image_height, criterion, Timnet, LR, MaxEpoch, usePreprocessedData)

		Timnet:training()
		nbBatch=20
	for epoch=1,20 do
		for numBatch=1, nbBatch do
			trainData=getBatch(train_list, BatchSize,image_width,image_height,numBatch-1,'TRAIN', usePreprocessedData)

			trainData.data=trainData.data:cuda()
			trainData.label=trainData.label:cuda()
			local predictions=Timnet:forward(trainData.data)
			local loss=criterion:forward(predictions, trainData.label)
			Timnet:zeroGradParameters()
			Timnet:backward(trainData.data,criterion:backward(Timnet.output, trainData.label) )
			Timnet:updateParameters(LR)
			xlua.progress(numBatch, nbBatch)
		end 
		Timnet:evaluate()
		error_train, loss_train=print_performance(train_list, BatchSize , 					Timnet,criterion, classes, image_width, image_height,'TRAIN', usePreprocessedData, nbBatch)
	end

end

function train_epochs(train_list,test_list, BatchSize,classes,image_width,image_height, criterion, Timnet, LR, MaxEpoch, usePreprocessedData)
	
	local list_error_train={}
	local list_error_test={}
	local list_loss_train={}
	local list_loss_test={}
	local timer = torch.Timer()
	local best_loss =100000
	local save_best=string.gsub(save_name, ".t7", "_best.t7")

	if usePreprocessedData then
		train_list, test_list=GetTestAndTrain(csv,PPdatapath.."0/", 80,TestPPdatapath)
		train_list=shuffleDataList(train_list)
		test_list=shuffleDataList(test_list)
	end
		
	bootStrap(train_list,test_list, BatchSize,classes,image_width,image_height, criterion, Timnet, LR, MaxEpoch, usePreprocessedData)


	for epochs=0, MaxEpoch do
		local folder=epochs%10
		if usePreprocessedData then
			train_list, test_list=GetTestAndTrain(csv,PPdatapath..folder.."/", 80,TestPPdatapath)
			train_list=shuffleDataList(train_list)
			test_list=shuffleDataList(test_list)
		end
		
		local nbBatch=math.floor(#train_list.data/BatchSize)+1
		Timnet:training()
		print(" Epochs :  "..epochs.. " - time : "..os.date("%X"))
		for numBatch=1, nbBatch do	
			-- Data ---------------------------------------------------------------
			trainData=getBatch(train_list, BatchSize,image_width,image_height,numBatch-1,'TRAIN', usePreprocessedData)

			trainData.data=trainData.data:cuda()
			trainData.label=trainData.label:cuda()

			local predictions=Timnet:forward(trainData.data)
			local loss=criterion:forward(predictions, trainData.label)
			-----------------------------------------------------------------------
			-- Stochastique Gradient Descent
			Timnet:zeroGradParameters()
			Timnet:backward(trainData.data,criterion:backward(Timnet.output, trainData.label) )
			Timnet:updateParameters(LR)

			xlua.progress(numBatch, nbBatch)
		end 
		--save_model(Timnet,"./Save/Savemodele08_07.t7")
		--Testing
		Timnet:evaluate()
		error_train, loss_train=print_performance(train_list, BatchSize , 					Timnet,criterion, classes, image_width, image_height,'TRAIN', usePreprocessedData)

		error_test, loss_test=print_performance(test_list, BatchSize , Timnet,criterion, 						classes, image_width, image_height,'VALID', usePreprocessedData)

		if loss_test<best_loss then
			best_loss=loss_test
			save_model(Timnet,save_best)
		elseif epochs==MaxEpoch then
			save_model(Timnet,save_name)
		end


		table.insert(list_error_train,error_train)
		table.insert(list_error_test,error_test)
		table.insert(list_loss_train,loss_train*100)
		table.insert(list_loss_test,loss_test*100)

		show_figure(list_error_train,list_error_test,list_loss_train,list_loss_test,save_name)
	end
end



-- Training Parameters -------------------------------------------------------

local LR=0.001
local MaxEpoch=20
local classes={"c0","c1","c2","c3","c4","c5","c6","c7","c8","c9"}
home=paths.home
datapath=home.."/Kaggle/imgs/train/"
PPdatapath=home.."/Kaggle/PreprocessedData/Train/epoch"
TestPPdatapath=home.."/Kaggle/PreprocessedData/Test/"
local BatchSize=5
local image_width=200
local image_height=200
save_name='./Save/Savemodele20_07.t7'
load_name='./Save/Savemodele20_07_best.t7'
local reload=true
local usePreprocessedData=true
local UseSecondeGPU=true

------------------------------------------------------------------------------

if UseSecondeGPU then
	cutorch.setDevice(2) 
end

if reload==true then
	Timnet = torch.load(load_name):double()
	print('Timnet\n' .. Timnet:__tostring());
	print(load_name.." Loaded")
else
	modele_file='./models/modele2'
	print("------------------------------------------------------------------------------")
	print("------------------------------------------------------------------------------")
	print("-----------------------------"..modele_file.."--------------------------------")
	print("------------------------------------------------------------------------------")
	print("------------------------------------------------------------------------------")
	require(modele_file)

	Timnet = nn.Sequential()
	Timnet:add(GetModel(image_width,image_height))
end

collectgarbage()
Timnet=Timnet:cuda()
criterion = nn.ClassNLLCriterion() -- a negative log-likelihood for multi-class classification
criterion = criterion:cuda()
----------------------------------------------------------------
csv=home.."/Kaggle/driver_imgs_list.csv"

local trainList={}
local testList={}
if not usePreprocessedData then
	trainList, testList=GetTestAndTrain(csv,datapath, 80)
	trainList=shuffleDataList(trainList)
	testList=shuffleDataList(testList)
end

-- Epoch Training -------------------------------------------------------
train_epochs(trainList, testList, BatchSize,classes,image_width,image_height, criterion, Timnet, LR, MaxEpoch,usePreprocessedData)


