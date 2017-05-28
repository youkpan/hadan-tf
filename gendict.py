zh_dict_index = {}
zh_dict_str = []
zh_dict_str_index = []
cnt = 0
def hexShow(argv):  
    result = ''  
    hLen = len(argv)  
    for i in xrange(hLen):  
        hvol = ord(argv[i])  
        hhex = '%02x'%hvol  
        result += hhex+' '  
    return result  
  

with open("/home/pan/download/utf/allbook.txt", "rb") as f:
  #if(f.readline() == ""):
	print("geting data")
	bookdata = f.read(190000000).decode('UTF-8')
	print("geting data  OK ")
	lineu = bookdata

	cnt += 1
	#print(lineu)
	for k in range(0,len(lineu)):
		try:
			#t = str(lineu[k]).encode('gb2312')
			t = str(lineu[k]).encode('utf-8')
			#print(lineu[k],t , int(t[0]),int(t[1]),"%02x"%int(t[1]), len(t))
			#print(lineu[k],t2 , int(t2[0]),int(t2[1]),"%02x"%int(t2[2]), len(t2))
			#print(lineu[k],t ,int(t))
			#if(len(t) >2):
			#	print(lineu[k],t , len(t))
			index = 0
			for j in  range(0,len(t)):
				index = index << 8
				index |= int(t[j])
			#index = int(t[0])<<16 | int(t[1]) << 8 |  int(t[2])
			#index2 = int(t[0])*65536 + int(t[1]) *256 +  int(t[2])
			#print( lineu[k] , index )
			if index not in zh_dict_index.keys() :
				print ("insert(index)",index , lineu[k])
				zh_dict_index[index] = 1
				zh_dict_str.append(lineu[k])
				zh_dict_str_index.append(index)
		except Exception as e:
			pass

f2 = open("/home/pan/fairseq/dict_string2.txt","w") #opens file with name of "test.txt"
for k in range(0,len(zh_dict_index)):
	print(k,zh_dict_str[k] , zh_dict_str_index[k])
	f2.write(zh_dict_str[k]+" ")

f2.close()
print(len(zh_dict_index))
