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

global BTCC_pro_data
global BTCC_pro_market_img
global BTCC_pro_market_price
global num_train_images
global num_val_images

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
  global BTCC_pro_market_img
  BTCC_pro_market_img = []
  global BTCC_pro_market_price
  BTCC_pro_market_price = []
  global BTCC_pro_market_price_last
  BTCC_pro_market_price_last =[]
  global BTCC_pro_market_ys
  BTCC_pro_market_ys =[]

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

  for fileidx in range(1,banch_size):
    ll = 0
    
    input_dialog = []
    #read data.txt

    #fileidx+=1 #10+int(random.random()*20)
    #print(fileidx%100)
    if(fileidx%1000==1):
      #call("clear")
      print(" reading data: %d %2.2f %% \r"%(fileidx,float(float(fileidx)*100/banch_size)))

    #print(fileidx)
    with open("/home/pan/fairseq/sentence/s_"+str(fileidx), "rb") as f:

      #head = f.read(257)
      #print("head2 "+ str(head))
      head = f.read(13)
      #print("head "+ str(head))
      if(str(head)[-2]!="n"):
        head = f.read(1)
        #print(str(head))

      try:
        if(len(str(head))==0):
          print('break')
          break
      except Exception as e:
        print('break')
        break
      finally:
        #print('read')

        sentence = f.read(256*12)
        sentence2 = np.zeros(256*12)
        lens = len(sentence)
        for i in range(0,lens) :
          char=sentence[i:i+1]
          sentence2[i] = ord(char)/255
        #print(np.sum(sentence2))
        #print(np.sum(data_buf4))
        #print(sentence2.shape)
        #input_dialog=vstack((data_buf1,data_buf2,data_buf3))
        #print(sentence2)
        #return
        #data_buf1 = data_buf2
        #data_buf2 = data_buf3
        #data_buf3 = data_buf4
        data_buf4 = sentence2


        BTCC_pro_market_img.append(0)
        BTCC_pro_market_price.append(0)
        BTCC_pro_market_price_last.append(0)
         
        rand = random.random()

        #plt.imshow(sentence2.reshape(4*16,3*16))
        #plt.show()

        xs.append(fn)
        ys.append(fn)
        fn +=1

        BTCC_pro_data.append(sentence2)
        

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
  return BTCC_pro_market_img[i]

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

def LoadTrainBatch(train_batch_pointer,batch_size,predict_time):
    #global train_batch_pointer
    global num_train_images
    x_out = []
    x_digit = []
    y_out = []
    y_last_out =[]

    for ii in range(0, batch_size-1):
        img_index = train_xs[(train_batch_pointer + ii) % num_train_images]
        #if img_index > len(BTCC_pro_market_img)-predict_time:
        #    img_index = len(BTCC_pro_market_img)-predict_time
        #print(ii)
        #print(num_train_images)
        if(img_index < 3):
          img_index = 3
        #print(img_index)
        data_buf1 = get_xdigit(img_index- 3)
        data_buf2 = get_xdigit(img_index- 2)
        data_buf3 = get_xdigit(img_index- 1)

        input_dialog = np.concatenate((data_buf1,data_buf2,data_buf3))
        #print(input_dialog.shape)
        y = get_xdigit(img_index)
        yy = np.zeros(256)
        for tt in range(0,12):
          for j in range(0,256):
            yy[tt] +=  y[j]
        '''
        print("SUM:")
        print(np.sum(y))
        print(np.sum(yy))
        '''
        y_out.append(y)
        data_mean = (data_buf1+data_buf2+data_buf3)/3
        #print (data_mean)
        y_last_out.append(data_mean)
        #print("img_index %d"% img_index )
        xin = np.zeros(256)
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
        
        learn_r = 0.001
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


    return x_out,x_digit, y_out,learn_r , y_last_out

def LoadValBatch(val_batch_pointer ,batch_size,predict_time):
    #global val_batch_pointer
    global num_val_images
    x_out = []
    y_out = []
    x_digit = []
    y_last_out = []
    for ii in range(0, batch_size):
        img_index = val_xs[(val_batch_pointer + ii) % num_val_images]
        if(img_index < 3):
          img_index = 3
        data_buf1 = get_xdigit(img_index- 3)
        data_buf2 = get_xdigit(img_index- 2)
        data_buf3 = get_xdigit(img_index- 1)

        input_dialog = np.concatenate((data_buf1,data_buf2,data_buf3))
        
        y = get_xdigit(img_index)
        yy = np.zeros(256)
        for tt in range(0,12):
          for j in range(0,256):
            yy[tt] +=  y[j]


        y_out.append(y)
        data_mean = (data_buf1+data_buf2+data_buf3)/3
        y_last_out.append(data_mean)
        #print("img_index %d"% img_index )
        xin = np.zeros(256)
        x_out.append(xin)
        x_digit.append(input_dialog)
        
    val_batch_pointer += batch_size
    return x_out,x_digit, y_out,y_last_out
