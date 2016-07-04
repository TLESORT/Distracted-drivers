-- This file assume your data is contained in different folder (one per classe)
-- And that the folders have the name of the classe you gave in the classe list

require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'math'
require 'string'

---------------------------------------------------------------------------------------
-- Function : images_Paths(path)
-- Input : path of a Folder which contained jpg images
-- Output : list of the jpg files path
---------------------------------------------------------------------------------------
function images_Paths(path)
	local list={}
	for file in paths.files(path) do
	   -- We only load files that match the extension
	   if file:find('jpg' .. '$') then
	      -- and insert the ones we care about in our table
	      table.insert(list, paths.concat(path,file))
	   end
	end
	return list
end

---------------------------------------------------------------------------------------
-- Function : GetImageTrainAndTestList(datapath, classes)
-- Input : path of a Folder which contains Folder of the classes
-- Input : List of the Classes
-- Input : Pourcentage of trainsize
-- Output : list of the jpg files path for training and testing
---------------------------------------------------------------------------------------
function GetImageTrainAndTestList(datapath, classes, trainRelativeSize)
	local train={}
	local test={}
	local trainRelativeSize=trainRelativeSize or 80
	
	if trainRelativeSize<1 or trainRelativeSize>100 then
		print("error in GetImageTrainAndTestList(datapath, classes, trainRelativeSize), trainRelativeSize, should be between 1 and 100")
	end
	
	trainList = {data = {},label = {}}
	testList = {data = {},label = {}}
	
    	print("classe = tonumber(string.sub(StrClasse, 2))+1 <-- corrrect?")


	for i = 1, #classes do
		StrClasse=classes[i]
		StrImages=images_Paths(datapath .. StrClasse .. "/")
		for j=1, #StrImages do
			if math.random(1, 100)<= trainRelativeSize then
				table.insert(trainList.data,StrImages[j])
				classe = tonumber(string.sub(StrClasse, 2))+1
				--table.insert(trainList.label,i)
				table.insert(trainList.label,classe)
			else
				table.insert(testList.data,StrImages[j])
				classe = tonumber(string.sub(StrClasse, 2))+1
				table.insert(testList.label,classe)
			end
		end
	end
	return trainList, testList
end

---------------------------------------------------------------------------------------
-- Function : images_Paths(path)
-- Input (lenght): lenght of the batch
-- Input (classes): 
-- Input (classes): 
-- Output : A batch of 'lenght' files with their label ('c0'->1, 'c1'->2 ...) randomly selected inside the 'classes' suitable for training or testing
---------------------------------------------------------------------------------------
function getRandomBatch(im_list, lenght, image_width, image_height, Type)

	local Type=Type or 'TRAIN'
	local image_width= image_width or 100
	local image_height= image_height or 100
	-- create train set structure:
	trainData = {
	   data = torch.Tensor(lenght, 3, image_width, image_height),
	   label = torch.IntTensor(lenght),
	   size = function() return lenght end
		}

	setmetatable(trainData, {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end});

	--create test set structure:
	testData = {
	      data = torch.Tensor(lenght, 3, image_width, image_height),
	      label = torch.IntTensor(lenght),
	      size = function() return lenght end
		}

	if Type=='TRAIN' then 
		struct=trainData
	elseif Type=='VALID' then
		struct=testData
	else
		print("Wrong type in getRandomBatch 'TRAIN' and 'VALID' (upper case) are only possible")
	end

	for i = 1, lenght do
		nb=math.random(1, #im_list.data)
		strData=im_list.data[nb]
		print(strData)
		img = image.load(strData,3,'byte')
		img_rsz=image.scale(img,image_width.."x"..image_height)
		struct.data[i]=img_rsz
		struct.label[i]=im_list.label[nb]
	end

	return struct

end


---------------------------------------------------------------------------------------
-- Function : images_Paths(path)
-- Input : List of the images
-- Input (lenght): lenght of the batch
-- Input (Type) : 'train' or 'valid' (default is train)
-- input : image_width
-- Input : image_height
-- Input (indice): give the indice of the batch inside the list : should be < to #im_indice/lenght 
-- Output : A batch of 'lenght' files which correspond to the indice given with their label ('c0'->1, 'c1'->2 ...) selected inside the list suitable for training or testing
---------------------------------------------------------------------------------------
function Batch(im_list, lenght, image_width, image_height, indice, Type)
	local Type=Type or 'TRAIN'
	local image_width= image_width or 100
	local image_height= image_height or 100
	local indice=indice or 0
	local datapath="/home/lesort/TrainTorch/Kaggle/imgs/train/"

	-- create train set structure:
	trainData = {
	   data = torch.Tensor(lenght, 3, image_width, image_height),
	   label = torch.IntTensor(lenght),
	   size = function() return lenght end
		}

	setmetatable(trainData, {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end});

	--create test set structure:
	testData = {
	      data = torch.Tensor(lenght, 3, image_width, image_height),
	      label = torch.IntTensor(lenght),
	      size = function() return lenght end
		}

	if Type=='TRAIN' then 
		struct=trainData
	elseif Type=='VALID' then
		struct=testData
	else
		print("Wrong type in getRandomBatch 'TRAIN' and 'VALID' are only possible")
	end
	
	if indice*lenght>#im_list.data then
		print("Wrong indice in getRandomBatch, the indice represent the number of the batch, it should be < to #im_list/lenght")
		return nil
	end
	if (indice+1)*lenght>=#im_list.data then	
		print("indice too big : some images will be seen 2 times for this epoch")
		start=#im_list.data-lenght
	else
		start=lenght*indice
	end
	for i = 1, lenght do
		img = image.load(im_list.data[start+i],3,'byte')
		img_rsz=image.scale(img,image_width.."x"..image_height)
		struct.data[i]=img_rsz
		struct.label[i]=im_list.label[start+i]
	end
	return struct
end

function getBatch(im_list, lenght, image_width, image_height, indice, Type,usePreprocessedData)
	local batch=Batch(im_list, lenght, image_width, image_height, indice, Type)
	if not usePreprocessedData then
		if Type=='TRAIN' then 
			batch=DataAugmentation(batch,lenght,image_width,image_height, 10, 0.1, 1)
		end
		batch=PreTraitement(batch,lenght)
	end
	
	return batch
end

function getRandomData(im_list,lenght, im_width, im_height,Type)
	local batch=getRandomBatch(im_list,lenght, im_width, im_height,Type)
	if Type=='TRAIN' then 
		batch=DataAugmentation(train,lenght,im_width,im_height, 20, 0.1, 10)
	end
	return PreTraitement(batch,lenght)
end

---------------------------------------------------------------------------------------
-- Function : shuffleDataList(im_list)
-- Input (im_list): list to shuffle
-- Output : The previous list after shuffling
---------------------------------------------------------------------------------------
function shuffleDataList(im_list)
	local rand = math.random 
	local iterations = #im_list.data
	local j

	for i = iterations, 2, -1 do
		j = rand(i)
		im_list.data[i], im_list.data[j] = im_list.data[j], im_list.data[i]
		im_list.label[i], im_list.label[j] = im_list.label[j], im_list.label[i]
	end
	return im_list
end


---------------------------------------------------------------------------------------
-- Function : PreTraitement(batch,lenght)
-- Input (batch): Batch of images
-- Input (lenght): lenght of the batch
-- Output : A batch with a mean and a 1 variance for each channel
---------------------------------------------------------------------------------------

function PreTraitement(batch,lenght)
	-- Name channels for convenience
	local channels = {'y','u','v'}

	-- Normalize each channel, and store mean/std
	-- per channel. These values are important, as they are part of
	-- the trainable parameters. At test time, test data will be normalized
	-- using these values.
	--print('==> preprocessing data: normalize each feature (channel) globally')
	local mean = {}
	local std = {}
	local sub_mean = torch.Tensor(lenght)
	--image.display{image=batch.data, legend='Avant'}
	for i,channel in ipairs(channels) do
	   -- normalize each channel globally:
	   
	   mean[i] = batch.data[{ {},i,{},{} }]:mean()
	   std[i] = batch.data[{ {},i,{},{} }]:std()
	   batch.data[{ {},i,{},{} }]:add(-mean[i])
	   batch.data[{ {},i,{},{} }]:div(std[i])
	end
	--image.display{image=batch.data, legend='Après'}

	--preprocessing data: normalize all three channels locally----------------

	-- Define the normalization neighborhood:
	local neighborhood = image.gaussian1D(5) -- 5 for face detector training

	-- Define our local normalization operator (It is an actual nn module, 
	-- which could be inserted into a trainable model):
	local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1e-4)

	-- Normalize all channels locally:
	for c in ipairs(channels) do
	   for i = 1,lenght do
	      batch.data[{ i,{c},{},{} }] = normalization:forward(batch.data[{ i,{c},{},{} }])
	   end
	end
	return batch
end
---------------------------------------------------------------------------------------
-- Function : DataAugmentation(batch,lenght,height,width, angle, shift, chance)
-- Input (batch) : Batch of images
-- Input (lenght): lenght of the batch
-- Input (width): width of images
-- Input (height) : height of images
-- Input (shift) : max shift applied for the random translatation
-- Input (chance): Chance for each pixel for being reamplaced by a random other
-- Output : A batch with new images slightly modified for regularization
---------------------------------------------------------------------------------------
function DataAugmentation(batch,lenght,width,height, angle, shift, chance)
	for i = 1,lenght do
		-- random rotation
		angle=(math.random(0,2*angle)-angle)*math.pi/180
		batch.data[{ i,{},{},{} }] =  image.rotate(batch.data[{ i,{},{},{} }], angle)
		-- random translation
		dx=math.random(0,2*shift)-shift
		dy=math.random(0,2*shift)-shift
		batch.data[{ i,{},{},{} }] = image.translate(batch.data[{ i,{},{},{} }], dx, dy)
		-- add random pixel kill
		batch.data[{ i,{},{},{} }] = randomPixelKillAndNoise(batch.data[{ i,{},{},{} }],width,height, chance)
	end
	return batch
end

---------------------------------------------------------------------------------------
-- Function : randomPixelKill(im,width,height, chance)
-- Input (im) : image
-- Input (height) : height of images
-- Input (width): width of images
-- Input (chance): Chance for each pixel for being reamplaced by a random other (1 for 1% chance to be killed)
-- Output : A batch with random pixels killed
---------------------------------------------------------------------------------------
function randomPixelKillAndNoise(im,width,height, chance)
	local channels = {'y','u','v'}

	for c in ipairs(channels) do
		for i=1,height do
			for j=1, width do
				
				if math.random(0, 100)<chance then
					im[{ {c},i,j}]=math.random(0,255)
				end
				Noise=math.random(-5, 5)
				im[{ {c},i,j}]=im[{ {c},i,j}]+Noise
				if im[{ {c},i,j}][1]>255 then im[{ {c},i,j}]=255 end
				if im[{ {c},i,j}][1]<0 then im[{ {c},i,j}]=0 end
			end
		end
	end
	return im

end

