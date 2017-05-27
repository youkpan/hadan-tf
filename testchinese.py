def check_contain_chinese(check_str):
    for ch in check_str.decode('utf-8'):
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

a = "你好  123 你好 你好"

print(a.encode('utf-8'))

for k in range(0,len(a.encode('gb2312'))):
	print(a[k])
	print( (a[k].encode('gb2312')))

for k in a.encode('utf-8'):
	print(v)
	r = check_contain_chinese(v)
	print (r)
