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
local home=paths.home
local datapath=home.."/Kaggle/imgs/train/"
local PPdatapath=home.."/Kaggle/PreprocessedData/Train/epoch0/"
local TestPPdatapath=home.."/Kaggle/PreprocessedData/Test/"
local csv=home.."/Kaggle/driver_imgs_list.csv"

local newList = false

if newList then
	train_list, test_list=GetTestAndTrain(csv,datapath, 80)
else
	train_list, test_list=GetTestAndTrain(csv,PPdatapath, 80, datapath, true)
end
local MaxEpoch=10
local BatchSize=5
local image_width=200
local image_height=200

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

nbBatch=math.floor(#test_list.data/BatchSize)+1
for numBatch=0, nbBatch-1 do
        testData=getBatch(test_list, BatchSize,image_width,image_height,numBatch,'VALID', false)
 
       if (numBatch+1)*BatchSize>=#test_list.data then
              indice=#test_list.data-BatchSize
       else
              indice=BatchSize*numBatch
       end

	for i=1, BatchSize do
		local newFolder="PreprocessedData/Test/"
		local filename= string.gsub(test_list.data[indice+i], "imgs/train/", newFolder)
               
		tensor= clampImage(testData.data[i])
                image.save(filename,tensor)

        end
	xlua.progress(numBatch, nbBatch)
end

nbBatch=math.floor(#train_list.data/BatchSize)+1
for epochs=0, MaxEpoch do
        for numBatch=0, nbBatch-1 do
                trainData=getBatch(train_list,BatchSize,image_width,image_height,numBatch,'TRAIN',false)
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
