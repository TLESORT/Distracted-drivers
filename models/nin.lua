require 'nn'

-- Network-in-Network
-- achieves 92% with BN and 88% without

local model = nn.Sequential()

local function Block(...)
  local arg = {...}
  model:add(nn.SpatialConvolution(...))
  model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
  model:add(nn.ReLU(true))
  return model
end

function GetModel()
	nbF=96--192
	nbF2=80--160
	nbF3=48--96
	Block(3,nbF,5,5,1,1,2,2)
	Block(nbF,nbF2,1,1)
	Block(nbF2,nbF3,1,1)
	model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
	model:add(nn.Dropout())
	Block(nbF3,nbF,5,5,1,1,2,2)
	Block(nbF,nbF,1,1)
	Block(nbF,nbF,1,1)
	model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
	model:add(nn.Dropout())
	Block(nbF,nbF,3,3,1,1,1,1)
	Block(nbF,nbF,1,1)
	Block(nbF,10,1,1)
	model:add(nn.SpatialAveragePooling(8,8,1,1):ceil())
	--Timnet:add(nn.SpatialMaxPooling(10,10,10,10))
	model:add(nn.Dropout())
	model:add(nn.View(10*1849))
	model:add(nn.Linear(10*1849, 10))
	model:add(nn.LogSoftMax())   

	for k,v in pairs(model:findModules(('%s.SpatialConvolution'):format(backend_name))) do
	  v.weight:normal(0,0.05)
	  v.bias:zero()
	end
--print(#model:cuda():forward(torch.CudaTensor(1,3,32,32)))
	print('nin\n' .. model:__tostring());
	return model
end
