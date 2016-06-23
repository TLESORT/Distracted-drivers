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

function list_contains(list,objet)
	for i=1,#list do
		if list[i]==objet then return true end
	end
	return false
end 

function GetTestAndTrain(csv,path, RelativeSize)
	local csv=csv or "/home/lesort/TrainTorch/Kaggle/driver_imgs_list.csv"
	local RelativeSize=RelativeSize or 80
	local path= path or "/home/lesort/TrainTorch/Kaggle/imgs/train/"
	
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
				path_file=path..list.label[i].."/"..list.file[i]
				table.insert(train_list.data,path_file)
				classe = tonumber(string.sub(list.label[i], 2))+1
				table.insert(train_list.label,classe)
			else
				path_file=path..list.label[i].."/"..list.file[i]
				table.insert(test_list.data,path_file)
				classe = tonumber(string.sub(list.label[i], 2))+1
				table.insert(test_list.label,classe)
			end
		end
	end

	return train_list, test_list
end





