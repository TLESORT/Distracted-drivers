
---------------------------------------------------------------------------------------
-- Function : ParseCSVLine (line,sep)
-- Input (line): line to parse
-- Input (sep) : seprator between two data "," for "example
-- Output : list of the data contained in the csv line
---------------------------------------------------------------------------------------
function ParseCSVLine (line,sep) 
	local res = {}
	local pos = 1
	sep = sep or ','
	while true do 
		local c = string.sub(line,pos,pos)
		if (c == "") then break end
		if (c == '"') then
			-- quoted value (ignore separator within)
			local txt = ""
			repeat
				local startp,endp = string.find(line,'^%b""',pos)
				txt = txt..string.sub(line,startp+1,endp-1)
				pos = endp + 1
				c = string.sub(line,pos,pos) 
				if (c == '"') then txt = txt..'"' end 
				-- check first char AFTER quoted string, if it is another
				-- quoted string without separator, then append it
				-- this is the way to "escape" the quote char in a quote. example:
				--   value1,"blub""blip""boing",value3  will result in blub"blip"boing  for the middle
			until (c ~= '"')
			table.insert(res,txt)
			assert(c == sep or c == "")
			pos = pos + 1
		else	
			-- no quotes used, just look for the first separator
			local startp,endp = string.find(line,sep,pos)
			if (startp) then 
				table.insert(res,string.sub(line,pos,startp-1))
				pos = endp + 1
			else
				-- no separator found -> use rest of string and terminate
				table.insert(res,string.sub(line,pos))
				break
			end 
		end
	end
	return res
end


---------------------------------------------------------------------------------------
-- Function : list_contains(list,objet)
-- Input (list): list of objects
-- Input (objet) : object to find in the list
-- Output : true if the object is in the list false otherwise
---------------------------------------------------------------------------------------
function list_contains(list,objet)
	for i=1,#list do
		if list[i]==objet then
			 return true
		 end
	end
	return false
end 

---------------------------------------------------------------------------------------
-- Function : file_exists(name)
-- Input (name): complete path of a file
-- Output : true if the file exist, false otherwise
---------------------------------------------------------------------------------------

function file_exists(name)
	return os.rename(name, name)
end

---------------------------------------------------------------------------------------
-- Function : GetTestAndTrain(csv,Trainpath, RelativeSize, Testpath)
-- Input (csv): file path where the data paths is
-- Input (Trainpath) : folder where the train data is
-- Input (RelativeSize): relative size between train and test (ex : 80 for #train/(#train+#test)=0.8)
-- Input (Testpath) : folder where the test data is 
-- Output : return a list of data and label for train and test for distracterd driver kaggle competition
---------------------------------------------------------------------------------------
function GetTestAndTrain(csv,Trainpath, RelativeSize, Testpath)
	local RelativeSize=RelativeSize or 80
	local Testpath=Testpath or Trainpath
	local fp = assert(io.open (csv))
	local line=fp:read()
	local headers=ParseCSVLine(line,",")

	-- now read the next line from the file and store in a hash

	list={sujet={},file={},label={}}
	local line=fp:read()
	while line do
		local cols=ParseCSVLine(line,",")
		table.insert(list.sujet,cols[1])
		table.insert(list.label,cols[2])
		table.insert(list.file,cols[3])
		line=fp:read()
	end
	fp:close()


	local test_list={data={},label={}}
	local train_list={data={},label={}}
	local subject_train={}
	local subject_test={}
	if Trainpath == Testpath then
		for i=1, #list.sujet do
			if not (list_contains(subject_train,list.sujet[i]) or list_contains(subject_test,list.sujet[i])) then

				-- we choose which Subject go in train and which in test
				if math.random(1, 100)<RelativeSize then
					table.insert(subject_train,list.sujet[i])
				else
					table.insert(subject_test,list.sujet[i])
				end
			else
				if list_contains(subject_train,list.sujet[i]) then
					path_file=Trainpath..list.label[i].."/"..list.file[i]
					table.insert(train_list.data,path_file)
					classe = tonumber(string.sub(list.label[i], 2))+1
					table.insert(train_list.label,classe)
				else
					path_file=Testpath..list.label[i].."/"..list.file[i]
					table.insert(test_list.data,path_file)
					classe = tonumber(string.sub(list.label[i], 2))+1
					table.insert(test_list.label,classe)
				end
			end
		end
	else
		-- this case is when we use preprocessed data
		-- in this case Train and test are already separated because they don't have the same preprocessing	
	for i=1, #list.file do
			name=Trainpath..list.label[i].."/"..list.file[i]
			--if the file exist in the training directory then it is a training example
			if file_exists(name) then 
				table.insert(train_list.data,name)
				classe = tonumber(string.sub(list.label[i], 2))+1
				table.insert(train_list.label,classe)
			elseif file_exists(Testpath..list.label[i].."/"..list.file[i]) then 
				-- if the ile isn't in the train directory then it is a test example
				name=Testpath..list.label[i].."/"..list.file[i]
				table.insert(test_list.data,name)
				classe = tonumber(string.sub(list.label[i], 2))+1
				table.insert(test_list.label,classe)
			else
				print(list.file[i].." don't exist either in train or test path")
			end
		end
	end
	return train_list, test_list
end





