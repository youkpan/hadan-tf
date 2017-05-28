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
			index = int(t[0])<<16 | int(t[1]) << 8 |  int(t[2])
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



end






import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

Vector_Word_i=[  [[1,2],[3,4] ,[1,2]],
				 [[3,4],[5,6],[7,8]] ,
				 [[9,10],[11,12],[13,14]],
				 [[15,16],[17,18],[19,20]] 
			  ] #np.ones((2,3,2), dtype=float)
Word_mark_i=[[1,2,3],[1,2,3],[1,2,3],[5,6,7]] #np.ones((2,3), dtype=float)

x_digit = tf.placeholder(tf.float32, shape=[None,3,2])
Vector_Word =tf.reshape(x_digit, shape=[-1,3,2])

Word_mark = tf.placeholder(tf.float32, shape=[None,3])

Word_mark_w = 1
Word_mark_b = 1
Word_mark_t = tf.reshape(Word_mark,shape=[-1,1])
#print(Vector_Word.shape)

S1 = tf.gather(Vector_Word, 0,)

def word_loop(idx,Wc,S1): 
  W1 = tf.gather(Vector_Word, idx)
  W2 = tf.gather(Vector_Word, idx+1)

  W12 =  tf.reduce_sum( tf.abs(tf.subtract(W2 , W1)))/3072
  mark_i = Word_mark_w*tf.gather(Word_mark_t, idx+1)+Word_mark_b

  S2 =  mark_i*(S1 + W1*W12)*  Wc   + W2
  return S2

S1=tf.gather(Vector_Word, 0)
S2 = word_loop(0,0.07,S1)

print( sess.run(S2,feed_dict={x_digit:Vector_Word_i,Word_mark:Word_mark_i} ))

end



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
