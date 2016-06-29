
require 'nn'

-- network-------------------------------------------------------
function getNet(image_width,image_height)
	nbFilter=32
	local Drop_Out=nn.SpatialDropout(0.25)

	Timnet = nn.Sequential()

	Timnet:add(nn.SpatialConvolution(3, nbFilter, 3, 3))
	Timnet:add(nn.ReLU())
	Timnet:add(nn.SpatialConvolution(nbFilter, nbFilter, 3, 3))
	Timnet:add(nn.SpatialBatchNormalization(nbFilter))
	Timnet:add(nn.ReLU())
	Timnet:add(nn.SpatialMaxPooling(2,2,2,2))
 	--Timnet:add(Drop_Out)

	width=math.floor((image_width-2)/2)-1
	height=math.floor((image_height-2)/2)-1
	print(height.." : height")
	image_size=width*height 
	print(image_size.." : image size")

	Timnet:add(nn.SpatialConvolution(nbFilter, 2*nbFilter, 3, 3))
	Timnet:add(nn.ReLU()) 
	Timnet:add(nn.SpatialConvolution(2*nbFilter, 2*nbFilter, 3, 3))
	Timnet:add(nn.SpatialBatchNormalization(2*nbFilter))    
	Timnet:add(nn.ReLU())                
	Timnet:add(nn.SpatialMaxPooling(2,2,2,2)) -- 128*23*23
	Timnet:add(Drop_Out)

	width=math.floor((width-2)/2)-1
	height=math.floor((height-2)/2)-1
	print(height.." : height")
	image_size=width*height
	print(image_size.." : image size")

	Timnet:add(nn.SpatialConvolution(2*nbFilter, 4*nbFilter, 3, 3))
	Timnet:add(nn.ReLU()) 
	Timnet:add(nn.SpatialConvolution(4*nbFilter, 4*nbFilter, 3, 3))
	Timnet:add(nn.SpatialBatchNormalization(4*nbFilter))   
	Timnet:add(nn.ReLU())                  
	Timnet:add(nn.SpatialMaxPooling(10,10,10,10))
	Timnet:add(Drop_Out)

	width=math.floor((width-2)/10)
	height=math.floor((height-2)/10)
	print(height.." : height")
	image_size=width*height
	print(image_size.." : image size")

	Timnet:add(nn.View(4*nbFilter*width*height))                    -- reshapes  3D tensor into 1D tensor 
	Timnet:add(nn.Linear(4*nbFilter*width*height, 500))             -- fully connected layer 
	Timnet:add(nn.ReLU())                       
	Timnet:add(nn.Linear(500, 500))
	Timnet:add(nn.ReLU())                       
	Timnet:add(nn.Linear(500, 10))                   -- 10 is the number of outputs of the network 
	Timnet:add(nn.LogSoftMax())                     -- converts the output to a log-probability. 
	Timnet=Timnet:cuda()

	-- Initiallisation : "Understanding the difficulty of training deep feedforward neural networks"
	local method = 'xavier'
	local Timnet = require('weight-init')(Timnet, method)
	print('Timnet\n' .. Timnet:__tostring());
	--image.display{image=Timnet.modules[1].weight[{ {},{3},{},{} }], zoom=16, legend="apres-3"}
	return Timnet
end
