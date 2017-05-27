dict_index = {}
dict_vector = []
with open("/home/pan/fairseq/dict_string", "rb") as f:
  dict_string = str(f.read(100000).decode('utf-8'))
  dict2=dict_string.split(" ")
  for k in range(0,len(dict2)):
    dict_index[dict2[k]] = k

with open("/home/pan/fairseq/train_file/train.tags.zh-en.zh", "rb") as f:
  for line in f.readlines():
    lineu=line.decode('utf-8')
    for k in range(0,len(lineu)):
    	try:
    		print( lineu[k],dict_index[lineu[k]] )
    	except Exception as e:
    		pass
      

exit




print("dict_index",dict_index)
print(len(dict_index))
print(dict_index['你'])
print(dict_index['咱']) #2709,	

def check_contain_chinese(check_str):
    for ch in check_str.decode('utf-8'):
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

a = "你好  123 你好 你好"

print(a.encode('utf-8'))

b={"你":1}
print(b['你'.encode('gb2312').decode('gb2312	')])

for k in range(0,len(a.encode('gb2312'))):
	print(a[k])
	print( (a[k].encode('gb2312')))
	t = a[k].encode('gb2312')
	print( int(t[0])*256+int(t[1]))

for k in a.encode('utf-8'):
	print(v)
	r = check_contain_chinese(v)
	print (r)
