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


local classes={"c0","c1","c2","c3","c4","c5","c6","c7","c8","c9"}
local datapath="/home/lesort/TrainTorch/Kaggle/imgs/train/"

local csv="/home/lesort/TrainTorch/Kaggle/driver_imgs_list.csv"
local trainList, testList=GetTestAndTrain(csv,datapath, 80)

train_list=shuffleDataList(trainList)
test_list=shuffleDataList(testList)

local MaxEpoch=10
local BatchSize=5
local image_width=200
local image_height=200
local nbBatch=math.floor(#train_list.data/BatchSize)+1

for epochs=0, MaxEpoch do
        for numBatch=1, nbBatch do
                trainData=getBatch(train_list, BatchSize,image_width,image_height,numBatch-1,'TRAIN')

		local indice=0
		if (numBatch)*lenght>=#train_list.data then
			indice=#train_list.data-lenght
		else
			indice=lenght*numBatch
		end	

		for i=1, BatchSize do
			local newFolder="PrepocessedData/Train/Epoch"..epochs.."/"
			local filename= string.gsub((numBatch*BatchSize+i), "imgs/train/", newFolder)
			image.save(filename,trainData.data[i])
		end
        end
end

for numBatch=1, nbBatch do
        testData=getBatch(test_list, BatchSize,image_width,image_height,numBatch-1,'TRAIN')
        for i=1, BatchSize do
		local newFolder="PrepocessedData/Test/"
		local filename= string.gsub((numBatch*BatchSize+i), "imgs/train/", newFolder)
	        image.save(filename,testData.data[i])
        end
end
