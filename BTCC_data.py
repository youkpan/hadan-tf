import scipy.misc
import scipy.ndimage
import random
from numpy  import *
#import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy  import *
from subprocess import call
import tensorflow as tf

num_images=0
vector_w = 168
sentence_len = 17


global BTCC_pro_data
global sentence_line_mark
global BTCC_pro_market_price
global num_train_images
global num_val_images

global dict_vector
global dict_index
global dict_index_str

dict_index = {}
dict_index_str = {}
dict_vector = []
with open("/home/pan/fairseq/dict_string2.txt", "rb") as f:
  dict_string = str(f.read(100000).decode('utf-8'))
  dict2=dict_string.split(" ")
  for k in range(0,len(dict2)):
    dict_index[dict2[k]] = k
    dict_index_str[k] = dict2[k]

  #print("dict_index",dict_index)

with open("/home/pan/fairseq/dict_vector", "rb") as f:
  dict_vector_t = f.read(256)
  while(dict_vector_t):
    word_v = np.zeros(256)
    k=0
    for d in dict_vector_t:
      word_v[k] = (float(d))
      k= k +1
    #print(word_v) 
    dict_vector.append(word_v)
    #print(dict_vector_t)
    dict_vector_t = f.read(256)

#print(len(dict_vector))
#print(dict_vector[1])
bookdata = []
lineu = []
sentence_data = []
with open("/home/pan/download/utf/allbook.txt", "rb") as f:
  #if(f.readline() == ""):
  print("geting data")
  bookdata = f.read(190000000).decode('UTF-8')
  print("geting data  OK ")
  lineu = bookdata

def load_next_banch(start_index,banch_size,predict_time):

  global num_images

  xs = []
  ys = []

  #points to the end of the last batch
  train_batch_pointer = 0
  val_batch_pointer = 0

  #start_index = 68000

  filename = []
  global BTCC_pro_data 
  BTCC_pro_data = []
  global sentence_line_mark
  sentence_line_mark = []
  global BTCC_pro_market_price
  BTCC_pro_market_price = []
  global BTCC_pro_market_price_last
  BTCC_pro_market_price_last =[]
  global BTCC_pro_market_ys
  BTCC_pro_market_ys =[]
  global dict_index
  global dict_vector
  global lineu
  #print("dict_index",dict_index)

  signal_2 =0;
  signal_1 =0;
  signal_0 =0;
  signal_n1 =0;
  signal_n2 =0;

  imgblank = zeros((5,120), dtype=float)

  fn=0
  last_price = 0
  fileidx = start_index
  data_buf1 = []
  data_buf2 = []
  data_buf3 = []
  data_buf4 = []

  i=0

  input_dialog = []

  if 1:
    print (len(bookdata))
    text_words = len(bookdata)
    #for line in f.readlines():
    position = 0

    position = start_index
    while(position+500 < text_words):

      if (fn >banch_size ):
        break

      while(position +100 < text_words ):
        position +=1
        k = position
        if( lineu[k] =='：' or lineu[k] ==':' or lineu[k] =='“' or lineu[k] =='"'):
          #or lineu[k] =='，' or lineu[k] =='，' or  lineu[k] =='。'or lineu[k] ==';' or lineu[k] =='！'or lineu[k] =='？' or lineu[k] =='”' 
          #  or lineu[k] ==':'  or lineu[k] =='.' or lineu[k] ==';' or lineu[k] =='!' or lineu[k] =='?' or lineu[k] =='"' ):
            break;

      position +=1
      
      if lineu[k] =="“" :
        position +=1
      #else:
        #position +=1

      #lineu=line.decode('utf-8')
      line_vector = []
      line_mark = np.ones(sentence_len)
      #print(line)
      position_t =  position
      print("----------")
      sentence = ''
      for k in range(position,position+30):
        try:
          #print( lineu[k],dict_index[lineu[k]] )
          sentence += lineu[k]
          #word_v = dict_vector[dict_index[lineu[k]]] 
          word_v = np.zeros([168])
          word_v[int(dict_index[lineu[k]]/84)+128] = 1
          word_v[int(dict_index[lineu[k]]%84)] = 1
          #，。;！？”“
          if(lineu[k] ==','  or lineu[k] ==','
            or lineu[k] =='，'  or lineu[k] =='“'):
          #print(line_v)
              line_mark[k] = 0

             #print (lineu[k]) lineu[k] =='：'  or
          if((( lineu[k] =='，') and (position_t-position >7)) or 
            lineu[k] =='。'or lineu[k] ==';' or lineu[k] =='！'or lineu[k] =='？' or lineu[k] =='”' 
            or lineu[k] =='.' or lineu[k] ==';' or lineu[k] =='!' or lineu[k] =='?' or lineu[k] =='"' ):
              break

          position_t +=1
          line_vector.append(word_v)
        except Exception as e:
          pass
      
      position = position_t

      if len(line_vector)< 3 :
        continue
      print (sentence)
      sentence_data.append(sentence)

      if(len(line_vector) < sentence_len):
        for k in range(0,sentence_len-len(line_vector)):
          line_vector.append(dict_vector[0])
          sentence_line_mark.append(np.ones(sentence_len))

      '''
      line_vector2 = np.zeros([sentence_len,vector_w])

      for ii in range(0, sentence_len):
        for jj in range(0, vector_w):
          line_vector2[ii][jj] = line_vector[ii][jj]
      #x_digit3 = line_vector.reshape([3*16,16*4])
      plt.imshow(line_vector2)
      plt.show()
      '''

      sentence_line_mark.append(line_mark)
      BTCC_pro_data.append(line_vector)
      BTCC_pro_market_price.append(0)
      BTCC_pro_market_price_last.append(0)
       
      rand = random.random()

      #plt.imshow(sentence2.reshape(4*16,3*16))
      #plt.show()

      xs.append(fn)
      ys.append(fn)
      fn +=1

      
      
  #print(xs)
  print("finish read data")
  #get number of images
  num_images = len(xs)
  BTCC_pro_market_ys = ys
  #shuffle list of images
  c = list(zip(xs, ys))
  random.shuffle(c)
  xs, ys = zip(*c)

  global train_xs
  train_xs = xs[:int(len(xs) * 0.95)]
  global train_ys
  train_ys = ys[:int(len(xs) * 0.95)]

  global val_xs
  val_xs = xs[-int(len(xs) * 0.05):]
  global val_ys
  val_ys = ys[-int(len(xs) * 0.05):]

  global num_train_images
  num_train_images = len(train_xs)
  global num_val_images
  num_val_images = len(val_xs)

  print("num_train_images %d" %(num_train_images))

#imgblank = ones((5,200,3), dtype=float)

def get_data_img(i):
  return BTCC_pro_market_price_last[i]

def get_data_img0(i):
  return sentence_line_mark[i]

def get_price(i):
  return BTCC_pro_market_price[i]

def  get_price_list():
  return BTCC_pro_market_price

def get_xdigit(i):
  return BTCC_pro_data[i]

def get_price_last(i):
  return BTCC_pro_market_price_last[i]

def get_data_size():
  return len(BTCC_pro_data)

def LoadBatchData(xss,num_train_images,train_batch_pointer,batch_size,predict_time):

    x_out = []
    x_digit = []
    y_out = []
    y_last_out =[]
    Word_mark = []
    diff_s = []
    y_out_test2 = []
    sentence_data_t = []
    for ii in range(0, batch_size):
        img_index = xss[(train_batch_pointer + ii) % num_train_images]
        #if img_index > len(sentence_line_mark)-predict_time:
        #    img_index = len(sentence_line_mark)-predict_time
        #print(ii)
        #print(num_train_images)
        if(img_index < 3):
          img_index = 3
        #print(img_index)
        y_out_test = []
        if random.random() < 1.1 :
          img_index1 = int(random.random()*num_train_images)
          img_index2 = int(random.random()*num_train_images)
          img_index3 = int(random.random()*num_train_images)
          y_out_test = np.array([0,1])
        else :
          img_index1 = img_index - 3
          img_index2 = img_index - 2
          img_index3 = img_index - 1
          y_out_test = np.array([1,0])

        y_out_test2.append(y_out_test)
        data_buf1 = get_xdigit(img_index1)
        data_buf2 = get_xdigit(img_index2)
        data_buf3 = get_xdigit(img_index3)
        sentence_data_t.append([sentence_data[img_index1],sentence_data[img_index2]
          ,sentence_data[img_index3]])

        input_dialog_t = np.zeros([sentence_len*3,vector_w],dtype=float)
        for j in range(0,sentence_len):
          input_dialog_t[j] = np.zeros(vector_w)
          try:
            input_dialog_t[j] = data_buf1[j].copy()
          except Exception as e:
            pass

        for j in range(sentence_len,sentence_len*2):
          input_dialog_t[j] = np.zeros(vector_w)
          try:
            input_dialog_t[j] = data_buf2[j].copy()
          except Exception as e:
            pass

        for j in range(sentence_len*2,sentence_len*3):
          input_dialog_t[j] = np.zeros(vector_w)
          try:
            input_dialog_t[j] = data_buf3[j].copy()
          except Exception as e:
            pass  

        input_dialog = input_dialog_t.reshape([sentence_len*3*vector_w])

        Word_mark_d1 = sentence_line_mark[img_index1]
        Word_mark_d2 = sentence_line_mark[img_index2]
        Word_mark_d3 = sentence_line_mark[img_index3]

        Word_mark_d = np.concatenate((Word_mark_d1,Word_mark_d2,Word_mark_d3)).reshape([sentence_len*3])
        Word_mark.append(Word_mark_d)
        #print(Word_mark_d.shape)

        #input_dialog = np.concatenate((data_buf1,data_buf2,data_buf3)).reshape([sentence_len*3*vector_w])
        #print(input_dialog.shape)
        #print(random.random(1))


        y = get_xdigit(img_index)

        diff = zeros(sentence_len*3, dtype=float)
        for i in range(0,sentence_len*3-1):
          diff2 = 0
          for j in range(0,vector_w):
            diff2 += input_dialog[(i+1)*vector_w+j] - input_dialog[i*vector_w+j]
          diff[i] = diff2/vector_w

        diff_s.append(diff)

        #print(y[1])
        #print(y.shape)
        yy = np.zeros([sentence_len*vector_w], dtype=float)

        for tt in range(0,min(len(y),sentence_len)):
          for j in range(0,vector_w):
            yy[tt*vector_w+j] = float(y[tt][j])

        #print("show,yy")
        #plt.imshow(yy.reshape(3*16,4*16))
        #plt.show()

        '''
        for tt in range(0,10):
          for j in range(0,vector_w):
            ii  = int(tt/2)
            yy[ii*vector_w+j] +=  (yy[tt*vector_w+j]+yy[(tt+1)*vector_w+j])/2

        
        print("SUM:")
        print(np.sum(y))
        print(np.sum(yy))
        '''
        #print(yy.shape)
        y_out.append(yy)
        #print(data_buf1.shape)
        #data_mean = (data_buf1+data_buf2+data_buf3)/3
        #print (data_mean)
        y_last_out.append(yy)
        #print("img_index %d"% img_index )
        xin = np.zeros(vector_w)
        x_out.append(xin)
        x_digit.append(input_dialog)
        
        '''
        print("SUM2:")
        print(np.sum(data_buf1))
        print(np.sum(data_buf2))
        print(np.sum(input_dialog))
        

        plt.imshow(get_xdigit(img_index).reshape(4*16,3*16))
        plt.show()
        plt.imshow(input_dialog.reshape(4*16,9*16))
        plt.show()
        '''
        
        #learn_r = 0.001
    #return x_out, y_out
    #print("train pointer %d ,batch index %d ,img_index %d"% ( train_batch_pointer,((train_batch_pointer ) % num_train_images),img_index))
    '''
    plt.figure(1)
    #plt.figure(2)

    x = range(0,197)
    
    plt.figure(1)  

    plt.plot(x, BTCC_pro_data[img_index][0:197])
    plt.ylim(-20, 60.)
    '''
    #print("y:%f"%(train_ys[img_index-batch_size-1]))
    #plt.figure(2)  
    #plt.plot(x, BTCC_pro_data[img_index])
    #plt.ylim(-5, 10.)
    #plt.show()

    #print(BTCC_pro_data[img_index])

    train_batch_pointer += batch_size
    #while cv2.waitKey(100) != ord('q'):
	#   continue
    x_digit2 = np.zeros((sentence_len*3,batch_size,vector_w))
    diff_s2 = np.zeros((sentence_len*3,batch_size))
    Word_mark2 = np.zeros((sentence_len*3,batch_size))
    #print(len(x_digit[0]))
    for ii in range(0, sentence_len*3):
      for jj in range(0, batch_size):
        
        x_digit2[ii][jj] = x_digit[jj][ii].copy()
        #x_digit2[ii][jj] = x_digit[jj][ii]
        diff_s2[ii][jj] = diff_s[jj][ii]
        Word_mark2[ii][jj] = Word_mark[jj][ii]

    y_out2 = np.zeros((batch_size,3072))

    #print(x_digit2[4])
    
    '''
    for ii in range(0, sentence_len):
      for i2 in range(0, vector_w):
      for jj in range(0, batch_size):
        y_out2[jj][ii] = y_out[jj][ii]
    '''

    return x_out,x_digit, y_out_test2 , y_last_out ,Word_mark2 ,diff_s2 ,sentence_data_t

def LoadTrainBatch(train_batch_pointer,batch_size,predict_time):
  return LoadBatchData(train_xs,num_train_images,train_batch_pointer,batch_size,predict_time)
  
def LoadValBatch(val_batch_pointer ,batch_size,predict_time):
    #global val_batch_pointer
    global num_val_images
    return LoadBatchData(val_xs,num_val_images,val_batch_pointer,batch_size,predict_time)
 