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


local fp = assert(io.open ("/home/lesort/TrainTorch/Kaggle/driver_imgs_list.csv"))
local line=fp:read()
local headers=ParseCSVLine(line,",")
for i,v in ipairs(headers) do print(i,v) end

-- now read the next line from the file and store in a hash

list={sujet={},file={},label={}}
line=fp:read()
while line do
	line=fp:read()
	local cols=ParseCSVLine(line,",")
	table.insert(list.sujet,cols[1])
	table.insert(list.file,cols[2])
	table.insert(list.label,cols[3])

end
fp:close()

