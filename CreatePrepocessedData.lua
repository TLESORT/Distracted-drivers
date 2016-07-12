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
local PPdatapath="/home/lesort/TrainTorch/Kaggle/PreprocessedData/Train/epoch0/"
local TestPPdatapath="/home/lesort/TrainTorch/Kaggle/PreprocessedData/Test/"
local csv="/home/lesort/TrainTorch/Kaggle/driver_imgs_list.csv"

local newList = false

if newList then
	train_list, test_list=GetTestAndTrain(csv,datapath, 80)
else
	train_list, test_list=GetTestAndTrain(csv,PPdatapath, 80, datapath)
end
local MaxEpoch=10
local BatchSize=5
local image_width=200
local image_height=200
local nbBatch=math.floor(#train_list.data/BatchSize)+1

local function clampImage(tensor)
   if tensor:type() == 'torch.ByteTensor' then
      return tensor
   end
    
   local a = torch.Tensor():resize(tensor:size()):copy(tensor)
   min=a:min()
   max=a:max()
   a:add(-min)
   a:mul(1/(max-min))         -- remap to [0-1]
   return a
end

--[[
for epochs=0, MaxEpoch do
        for numBatch=1, nbBatch do
                trainData=getBatch(train_list,BatchSize,image_width,image_height,numBatch-1,'TRAIN',false)
		local indice=0
		if (numBatch+1)*BatchSize>=#train_list.data then
			indice=#train_list.data-BatchSize
		else
			indice=BatchSize*numBatch
		end	
		for i=1, BatchSize do
			local newFolder="PreprocessedData/Train/epoch"..epochs.."/"
			local filename= string.gsub(train_list.data[indice+i],"imgs/train/", newFolder)
			tensor= clampImage(trainData.data[i])
			image.save(filename,tensor)
		end
		xlua.progress(numBatch, nbBatch)
        end
end
--]]

nbBatch=math.floor(#test_list.data/BatchSize)+1
for numBatch=1, nbBatch do
        testData=getBatch(test_list, BatchSize,image_width,image_height,numBatch-1,'VALID', false)
 
       if (numBatch+1)*BatchSize>=#test_list.data then
              indice=#test_list.data-BatchSize
        else
              indice=BatchSize*numBatch
        end

	for i=1, BatchSize do
		local newFolder="PreprocessedData/Test/"
		local filename= string.gsub(test_list.data[indice+i], "imgs/train/", newFolder)
		--image.display{image=testData.data[i], legend=filename}
               
		tensor= clampImage(testData.data[i])
                image.save(filename,tensor)

        end
	xlua.progress(numBatch, nbBatch)
end

