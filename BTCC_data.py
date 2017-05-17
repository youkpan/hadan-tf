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

def load_next_banch(start_index,banch_size,predict_time):

  global num_images

  xs = []
  ys = []

  #points to the end of the last batch
  train_batch_pointer = 0
  val_batch_pointer = 0

  i=0

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

  ll=0
  last_price = 0
  #read data.txt
  with open("../BTCC_pro_train_data.txt") as f:
    print(" reading data:")
    for line in f:
      if(ll< start_index):
        ll+=1
        continue
      data = line.split('|') 

      try:
        if(len(data[-2])==0):
          continue
      except Exception as e:
        continue
      
      #print(len(data))
      #print(data)
      #print(data[2:268])
      #print(data[-2])
      data_v = []
      data_v2 = []
      j=0
      val2 = 0
      avg_line=0
      for val in data[2:268]:
        val1=float(val)
        avg_line = avg_line*0.9+0.1*val1
        if(j<120):
          val2+=val1
          data_v2.append (val2)
        '''
        if(j<120):
          if(avg_line >=0):
            val1 = pow(avg_line,0.46)
          else:
            val1 = -pow(-avg_line,0.46)
        if((j>=120 and j<240) or j== 265):
          if(val1 >=0):
            val1 = pow(val1,0.46)
          else:
            val1 = -pow(-val1,0.46)
        '''
        j+=1
        data_v.append (val1)

      price = float(data[-4])
      price_img = []
      '''
      vol_bar = zeros((5,120), dtype=float)

      for x in range(0,120,10):
        vol_sum = 0
        for k in range(0,10):
          vol_sum += float(data[122+x+k])
        vol_bar[0][x] = vol_sum
        #print("vol_sum x:%d sum:%d"%(x,vol_sum))

     
      if(last_price==0):
        last_price = price
      amount = data[-3]

      data_bar = zeros((5,120), dtype=float)
      data_bar[0][0] = price
      data_bar[0][1] = amount

      hour_bar = zeros((5,120), dtype=float)
      for x in range(0,24):
        if(int(data[242+x])) == 1:
          hour_bar[0][int(x/2)*10]=1
          #print("find hour %d pix: %d"%(x,int(x/2)*10))
          break

      price_img = zeros((120,120), dtype=float)
      last_pix_y = 0
      minn = 0

      for x in range(0,120):
        if data_v2[x] <= minn or x == 0:
          minn = data_v2[x]
      for x in range(0,120):
        pix_y = int(round(((data_v2[x]-data_v2[119])/5)+60))

        if pix_y >119 :
          pix_y = 119
        if pix_y < 0:
          pix_y = 0
        pix_y = 119-pix_y
        if(last_pix_y==0):
          last_pix_y = pix_y

        if(pix_y==last_pix_y):
          price_img[pix_y][x] = 1
        elif pix_y>last_pix_y:
          for y in range(last_pix_y,pix_y+1):
            price_img[y][x] = 1
        else:
          for y in range(pix_y,last_pix_y+1):
            price_img[y][x] = 1

        last_pix_y = pix_y

      '''

      #input_img = vstack((vol_bar,imgblank,data_bar,imgblank,hour_bar,
      #    imgblank,price_img))

      #input_img = tf.reshape(input_img,[90,120,1])

      BTCC_pro_market_img.append(price_img)
      BTCC_pro_market_price.append(price)
      BTCC_pro_market_price_last.append(data_v2)

      xs.append(i)
      
      data_predict = float(data[-2])
      diff = (price-last_price)*1000/price
      if i>=predict_time:
        last_price = BTCC_pro_market_price[i-predict_time]
      else:
        last_price = price;
      #if(i%predict_time == predict_time-1):
      #    last_price = price
      #print(data_predict)
      '''
      if diff>5:
        ys.append(np.array([1.0,0.0]) )
        signal_2 +=1
        data_v.append (1)
        data_v.append (0)
      elif data_predict>2:
        ys.append(np.array([1.0,0.0]))
        #print(1)
        signal_1 +=1

        data_v.append (1)
        data_v.append (0)
      el
      '''
      rand = random.random()
      #win = 1.0
      loss = 0.0
      #if(rand<0.6):
      win = 0.02*abs(price-last_price)/price +0.5
      #if(rand<0.1):
      #  win = 0.03*abs(data_predict)+0.5
      #if(rand>0.6):
      #  win = 0.37 + 0.02*abs(data_predict)
      #if(rand>0.9):
      #  win = 0.27 + 0.02*abs(data_predict)

      if(win>1):
        win = 1
      loss = 1-win

      if diff>0:
        
        ys.append(np.array([win,loss]))
        signal_0 +=1
        data_v.append (1)
        data_v.append (0)

      else:
        ys.append(np.array([loss,win]))
        signal_n2 +=1
        data_v.append (0)
        data_v.append (1)

      BTCC_pro_data.append(data_v)
      
      i+=1 #10+int(random.random()*20)
      if(i%2000==1):
        #call("clear")
        print(" reading data: %d %2.2f %% \r"%(start_index+i,float(float(i)*100/115001)))
        print(" signal_2:  %2.2f %% \r"%(float(float(signal_2)*100/i)))
        print(" signal_1:  %2.2f %% \r"%(float(float(signal_1)*100/i)))
        print(" signal_0:  %2.2f %% \r"%(float(float(signal_0)*100/i)))
        print(" signal_n1:  %2.2f %% \r"%(float(float(signal_n1)*100/i)))
        print(" signal_n2:  %2.2f %% \r"%(float(float(signal_n2)*100/i)))

      if(i>banch_size):
        break

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
    x_out = []
    x_digit = []
    y_out = []

    for i in range(0, batch_size-1):
        img_index = train_xs[(train_batch_pointer + i) % num_train_images]
        #if img_index > len(BTCC_pro_market_img)-predict_time:
        #    img_index = len(BTCC_pro_market_img)-predict_time

        try:
          y_out.append(BTCC_pro_market_ys[img_index ])
        except Exception as e:
          continue
        #print("img_index %d"% img_index )
        x_out.append(get_price_last(img_index))
        x_digit.append(get_xdigit(img_index))
        
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


    return x_out,x_digit, y_out,learn_r

def LoadValBatch(val_batch_pointer ,batch_size,predict_time):
    #global val_batch_pointer
    x_out = []
    y_out = []
    x_digit = []
    for i in range(0, batch_size):
        img_index = val_xs[(val_batch_pointer + i) % num_val_images]
        try:
          y_out.append(BTCC_pro_market_ys[img_index ])
        except Exception as e:
          continue

        x_out.append(get_price_last(img_index))#,imgblank,img_5,imgblank,img_10,imgblank,img_20)))
        x_digit.append(get_xdigit(img_index))
        #y_out.append(val_ys[(val_batch_pointer + i) % num_val_images])
        
    val_batch_pointer += batch_size
    return x_out,x_digit, y_out
