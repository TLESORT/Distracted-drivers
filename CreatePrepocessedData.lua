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

local MaxEpoch=1
local BatchSize=5
local image_width=200
local image_height=200
local nbBatch=math.floor(#train_list.data/BatchSize)+1

local function clampImage(tensor)
   if tensor:type() == 'torch.ByteTensor' then
      return tensor
   end
   local a = torch.Tensor():resize(tensor:size()):copy(tensor)
   a.image.saturate(a) -- bound btwn 0 and 1
   a:mul(255)          -- remap to [0..255]
   return a
end


for epochs=0, MaxEpoch do
        for numBatch=1, nbBatch do
                trainData=getBatch(train_list, BatchSize,image_width,image_height,numBatch-1,'TRAIN')

		local indice=0
		if (numBatch)*BatchSize>=#train_list.data then
			indice=#train_list.data-BatchSize
		else
			indice=BatchSize*numBatch
		end	

		for i=1, BatchSize do
			local newFolder="PreprocessedData/Train/epoch"..epochs.."/"
			local filename= string.gsub(train_list.data[indice+i], "imgs/train/", newFolder)
			--im=image.compressJPG(trainData.data[i])
			--print(im:size())
			--image.display(trainData.data[i])
			tensor= clampImage(trainData.data[i])
			--image.save(filename,tensor)
			tensor.libjpeg.save(filename,tensor,1,75)
		end
		xlua.progress(numBatch, nbBatch)
        end
end

nbBatch=math.floor(#test_list.data/BatchSize)+1

for numBatch=1, nbBatch do
        testData=getBatch(test_list, BatchSize,image_width,image_height,numBatch-1,'TRAIN')
 
       if (numBatch)*BatchSize>=#test_list.data then
              indice=#test_list.data-BatchSize
        else
              indice=BatchSize*numBatch
        end

	for i=1, BatchSize do
		local newFolder="PreprocessedData/Test/"
		local filename= string.gsub(test_list.data[indice+i], "imgs/train/", newFolder)
               tensor= clampImage(testData.data[i])
               --image.save(filename,tensor)
                tensor.libjpeg.save(filename,tensor,1,75)

        end
end

