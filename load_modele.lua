
require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'cutorch'

require 'GetBatchData'
require 'GetDataFromCsv'


--net = torch.load('model-test.t7'):double()
net = torch.load('Save/Savemodele18_07_best.t7'):double()

datapath=paths.home.."/Kaggle/imgs/train/"
trainList, testList=GetTestAndTrain(csv,datapath, 80)
BatchSize=2

im_width=200
im_height=200
lenght=1
testData0=getBatch(train_list, BatchSize,200,200,numBatch-1,'TRAIN', false)

--[[
correct = 0
for i=1,testData.label:size(1) do
    local groundtruth = testData.label[i]
    local prediction = net:forward(testData.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    --print("groundtruth", groundtruth, "indices", indices[1])
    if groundtruth == indices[1] then
	correct = correct + 1
    end
end

print(correct, 100*correct/testData.label:size(1) .. ' % ')
print(testData.label:size(1))

class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,testData.label:size(1) do
    local groundtruth = testData.label[i]
    local prediction = net:forward(testData.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
	class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end

--image.display(net.get(6).weight)

for i=1,#classes do
    print(classes[i], 100*class_performance[i]/testData.label:size(1) .. ' %')
end

print(net:get(6).weight:size())
--]]

image0=PreTraitement(testData0,1)
image.display{image=image0.data[1],  zoom=4, legend="image0"}
local prediction=net:forward(image0.data)
maxs, indices = torch.max(prediction,1)
print("truth = " .. testData0.label[1])
print("classe = " .. indices[1]-1)
output0=net:get(15).output[1]
image.display{image=image0.data,  zoom=4, legend="image0"}
image.display{image=output0, nrow=16,  zoom=2, legend="image0"}


image1=PreTraitement(testData1,1)
local prediction=net:forward(image1.data)
maxs, indices = torch.max(prediction,1)
print("truth = " .. testData1.label[1])
print("classe = " .. indices[1]-1)
output1=net:get(15).output[1]
image.display{image=image1.data,  zoom=4, legend="image1"}
image.display{image=output1, nrow=16,  zoom=2, legend="image1"}












