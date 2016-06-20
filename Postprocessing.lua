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

local file = io.open("understanding.csv", "a+")
file:write("img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n")


local fp = assert(io.open ("submission.csv"))
local line=fp:read()
local headers=ParseCSVLine(line,",")
for i,v in ipairs(headers) do print(i,v) end

-- now read the next line from the file and store in a hash

while line do
	local line=fp:read()
	--print(line)
	if line then
		local cols=ParseCSVLine(line,",")
		local values={}
	
		max=0
		for i,v in ipairs(cols) do
		   if tonumber(cols[i]) then
			result=tonumber(cols[i])
			if result>max then
			    max=result
			end
		   end
		end
		for i,v in ipairs(headers) do
		   if tonumber(cols[i]) then
			if tonumber(cols[i])==max then
				cols[i]=0.5275
			else
				cols[i]=0.0525
			end
		   end
		end


		pred={}
		table.insert(pred,cols[1])
		for i=2, #cols do
			value=cols[i]
			strValue= string.gsub(value, ",", ".") 
			table.insert(pred,strValue)
		end
		newline=table.concat(pred,",")..'\n'
		file:write(newline)
	else
	   print(newline)
	   fp:close()
	   file:close()
	   return nil
	end
end
fp:close()
file:close()

