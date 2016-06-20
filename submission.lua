require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'cutorch'
require 'cunn'
require 'math'
require 'csvigo'

require 'GetBatchData'

function getfile()
	path="/home/lesort/TrainTorch/Kaggle/imgs/test/"
	local pathlist={}
	local filelist={}
	for file in paths.files(path) do
	   -- We only load files that match the extension
	   if file:find('jpg' .. '$') then
	      -- and insert the ones we care about in our table
	      table.insert(pathlist, paths.concat(path,file))
	      table.insert(filelist, file)
	   end
	end
	return pathlist, filelist
end 

function PreTraining(im)
	-- Name channels for convenience
	local channels = {'y','u','v'}

	-- Normalize each channel, and store mean/std
	-- per channel. These values are important, as they are part of
	-- the trainable parameters. At test time, test data will be normalized
	-- using these values.
	--print('==> preprocessing data: normalize each feature (channel) globally')
	local mean = {}
	local std = {}
	--image.display{image=batch.data, legend='Avant'}
	for i,channel in ipairs(channels) do
	   -- normalize each channel globally:
	   mean[i] = im.data[{ {},i,{},{} }]:mean()
	   std[i] = im.data[{ {},i,{},{} }]:std()
	   im.data[{ {},i,{},{} }]:add(-mean[i])
	   im.data[{ {},i,{},{} }]:div(std[i])
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
	      im.data[{ 1,{c},{},{} }] = normalization:forward(im.data[{ 1,{c},{},{} }])
	end
	return im

end


function Image(im_list, image_width, image_height, indice)
	local image_width= image_width or 100
	local image_height= image_height or 100
	local indice=indice or 1
	local datapath="/home/lesort/TrainTorch/Kaggle/imgs/train/"

	-- create train set structure:
	Data = {
	   data = torch.Tensor(1, 3, image_width, image_height),
	   size = function() return 1 end
		}

	setmetatable(Data, {__index = function(t, i) 
                    return {t.data[i]} 
                end});
	img = image.load(im_list[indice],3,'byte')
	img_rsz=image.scale(img,image_width.."x"..image_height)
	Data.data[1]=img_rsz
	return Data
end
pathlist,filelist=getfile()
Timnet=torch.load('model.t7'):double()
Timnet=Timnet:cuda()

--csv=csvigo.load({path = "./submission.csv"})
local file = io.open("submission.csv", "a+")
file:write("img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n")
for i=1,#pathlist do
	im=Image(pathlist,200,200,i)
	im.data=PreTraining(im).data:cuda()
	--image.display{im.data, zoom=4}
	prediction=Timnet:forward(im.data)
	pred={}
	table.insert(pred,filelist[i])
	for i=1, prediction:size(1) do
		value=math.exp(prediction[i])
		strValue= string.gsub(value, ",", ".") 
		table.insert(pred,strValue)
	end
	file:write(table.concat(pred,",")..'\n')
end
file:close()


