function show_figure(list_error_train,list_error_test,list_loss_train,list_loss_test,name)

	-- log results to files
	accLogger = optim.Logger('./Log/'..name..'accuracy.log')
	LossLogger = optim.Logger('.Log/'..name..'Loss.log')

	for i=1, #list_error_train do
	-- update logger
		accLogger:add{['% train accuracy'] = list_error_train[i], ['% test accuracy'] = 					list_error_test[i]}
		LossLogger:add{['% train loss (x100)']    = list_loss_train[i], ['% test loss (x100)']    = 					list_loss_test[i]}
	end
	-- plot logger
	accLogger:style{['% train accuracy'] = '-', ['% test accuracy'] = '-'}
	LossLogger:style{['% train loss']    = '-', ['% test loss']    = '-'}
	accLogger:plot()
	LossLogger:plot()
end

-- truth: integers between 1 and 10
function count_error(truth, prediction,class_Accuracy)
	local errors=0
	maxs, indices = torch.max(prediction,2)
	for i=1,truth:size(1) do
	    if truth[i] ~= indices[i][1] then
		errors = errors + 1
	    else
		class_Accuracy[truth[i]] = class_Accuracy[truth[i]] + 1
	    end
	end
	return errors, class_Accuracy
end


function print_performance(im_list, Batchsize, net, criterion, classes, image_width, image_height,Type, usePreprocessedData, nbBatch)
	local correct = 0
	local nbBatch=nbBatch or (math.floor(#im_list.label/Batchsize)+1)
	local class_Accuracy = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	local class_tot = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	local errors_tot=0
	local loss_tot=0

	print("Performance : "..nbBatch.. " Test Batchs - time : "..os.date("%X"))
	for i=1, nbBatch do
	    local Data=getBatch(im_list, Batchsize, image_width, image_height, i-1, Type, usePreprocessedData)
	    Data.data= Data.data:cuda()
	    Data.label= Data.label:cuda()
	    local groundtruth = Data.label
	    --image.display(Data.data)
	    local prediction = net:forward(Data.data)--one image per batch
	    local loss=criterion:forward(prediction, groundtruth)
	    loss_tot=loss_tot+loss
	    --!--local confidences, indices = torch.sort(prediction, true)  --true -> sort in descending order
	    errors, class_Accuracy=count_error(groundtruth, prediction,class_Accuracy)
	    for j=1, Batchsize do
	    	class_tot[groundtruth[j]]=class_tot[groundtruth[j]]+1
	    end
	    errors_tot=errors_tot+errors
	    xlua.progress(i, nbBatch)
	end
	print("----------------"..Type.."-----------------")
	print('loss_tot '.. loss_tot/nbBatch) 
	print("nb errors : ".. errors_tot)
	print("errors : "..100*errors_tot/(nbBatch*Batchsize).. ' % ')
	print("Classe Accuracy: ")
	for i=1,#classes do
	    print(classes[i], 100*class_Accuracy[i]/class_tot[i] .. ' %	   sur : '..class_tot[i])
	end
	print("-------------------------------------")

	return 100*errors_tot/(nbBatch*Batchsize), loss_tot/nbBatch
end
