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

function train_epochs(train_list,test_list, BatchSize,classes,image_width,image_height, criterion, Timnet, LR, MaxEpoch, usePreprocessedData)
	
	local list_error_train={}
	local list_error_test={}
	local list_loss_train={}
	local list_loss_test={}
	local nbBatch=math.floor(#train_list.data/BatchSize)+1
	local timer = torch.Timer()
	local best_loss =100000
	--local save_name="./Save/Savemodele08_07.t7"
	local save_best=string.gsub(save_name, ".t7", "_best.t7")
		
	for epochs=0, MaxEpoch do

		if usePreprocessedData then
			train_list, test_list=GetTestAndTrain(csv,PPdatapath..epochs.."/", 80,TestPPdatapath)
		end
		Timnet:training()
		print(" Epochs :  "..epochs.. " - time : "..os.date("%X"))
		for numBatch=1, nbBatch do	
			-- Data ---------------------------------------------------------------
			trainData=getBatch(train_list, BatchSize,image_width,image_height,numBatch-1,'TRAIN', usePreprocessedData)

			trainData.data=trainData.data:cuda()
			trainData.label=trainData.label:cuda()

			--image.display(trainData.data)
			local predictions=Timnet:forward(trainData.data)
			local loss=criterion:forward(predictions, trainData.label)
			-----------------------------------------------------------------------
			--!--print('loss '.. loss .." batch : "..numBatch.."\n")
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

local LR=0.01
local MaxEpoch=15
local classes={"c0","c1","c2","c3","c4","c5","c6","c7","c8","c9"}
home=paths.home
datapath=home.."/Kaggle/imgs/train/"
PPdatapath=home.."/Kaggle/PreprocessedData/Train/epoch"
TestPPdatapath=home.."/Kaggle/PreprocessedData/Test/"
local BatchSize=12
local image_width=200
local image_height=200
save_name='./Save/Savemodele18_07.t7'
local reload=false
local usePreprocessedData=true
local multiGPU=true

------------------------------------------------------------------------------

if multiGPU then
	-- 2 GPU from my side but you could have more
	cutorch.setDevice(2) 
end

if reload==true then
	Timnet = torch.load(save_name):double()
	print('Timnet\n' .. Timnet:__tostring());
	print(save_name.." Loaded")
else
	modele_file='./models/modele_dropout'
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
local csv=home.."/Kaggle/driver_imgs_list.csv"

local trainList={}
local testList={}
if usePreprocessedData then
	trainList, testList=GetTestAndTrain(csv,PPdatapath.."0/", 80,TestPPdatapath)
else
	trainList, testList=GetTestAndTrain(csv,datapath, 80)
end

trainList=shuffleDataList(trainList)
testList=shuffleDataList(testList)

-- Epoch Training -------------------------------------------------------
train_epochs(trainList, testList, BatchSize,classes,image_width,image_height, criterion, Timnet, LR, MaxEpoch,usePreprocessedData)


