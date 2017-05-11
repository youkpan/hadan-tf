import tensorflow as tf
import scipy.misc
import scipy.ndimage
import model
import cv2
from subprocess import call
from numpy  import *
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = scipy.misc.imread('steering_wheel_image.png', mode="RGB")
cv2.imshow("steering wheel", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


smoothed_angle = 0

i = 0
start_index = 71050
differ = 0.0
differ_s = 0.0
imgblank = zeros((5,120), dtype=float)


uda_data = 1
game_data = 2
default_data = 0
use_data = game_data

filename = []
BTCC_pro_data =[]
BTCC_pro_market_data =[]
BTCC_pro_market_img= []

def get_data(i):
    if i > len(BTCC_pro_market_img)-1:
        i=len(BTCC_pro_market_img)-1
    return BTCC_pro_market_img[i]

 
orgi_start_time = 0

xs=[]
ys=[]

signal_2 =0;
signal_1 =0;
signal_0 =0;
signal_n1 =0;
signal_n2 =0;


with open("BTCC_pro_train_data.txt") as f:
  for line in f:  
      if(i< start_index):
        i+=1
        continue
      data = line.split('|')
      if(len(data[-2])==0):
        continue
      
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
        j+=1
        data_v.append (val1)
      xx='''   
      vol_bar = zeros((5,120), dtype=float)

      for x in range(0,120,10):
        vol_sum = 0
        for k in range(0,10):
          vol_sum += float(data[122+x+k])
        vol_bar[0][x] = vol_sum
        #print("vol_sum x:%d sum:%d"%(x,vol_sum))

      price = data[-4]
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
      #minn = min(data_v2)
      '''
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

      #input_img = vstack((vol_bar,imgblank,data_bar,imgblank,hour_bar,
       # imgblank,price_img))
      BTCC_pro_market_img.append(price_img)
      #cv2.imshow("input_img", input_img)

      xs.append(i)
      BTCC_pro_data.append(data_v)
      BTCC_pro_market_data.append(data_v2)
      
      data_predict = float(data[-2])
      if data_predict>20*2.72:
        ys.append(np.array([1.0,0.0,0.0]) )
        signal_2 +=1
      elif data_predict>20:
        ys.append(np.array([1.0,0.0,0.0]))
        signal_1 +=1
      elif data_predict>-20:
        ys.append(np.array([0.0,1.0,0.0]))
        signal_0 +=1
      elif data_predict>-20*2.72:
        ys.append(np.array([0.0,0.0,1.0]))
        signal_n1 +=1
      else:
        ys.append(np.array([0.0,0.0,1.0]))
        signal_n2 +=1
      
      #y.append(float(data[-2])/100)
      #print(float(data[-2])/100)
      i+=1
      if(i%5000==1):
        print(" reading: %d %2.2f %% \r"%(i,float(float(i)*100/115001)))
      if(i>75000):
        break

def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

ii=1
pass_cnt2=1
pass_cnt1=0
refuse_cnt =0
err_cnt =0
while (ii < len(BTCC_pro_data)):

    data = get_data(ii)
    #print(data)
    out = model.y.eval(feed_dict={model.x: [data], model.keep_prob: 1.0})[0]
    

    orgi_i = 1- np.argmax(ys[ii], 0)
    out_i = 1- np.argmax(out, 0)

    if( orgi_i == 1 or orgi_i== -1):
      pass_cnt2 +=1

    if(( orgi_i == 1) and out_i ==1) or (( orgi_i == -1) and out_i ==-1):
      pass_cnt1 +=1

    if( ( out_i != orgi_i) and (out_i !=0)):
      err_cnt +=1
    
    if ( out_i ==0):
      refuse_cnt +=1

    if ii %20 ==0 :
      call('clear')
      print("now_index: " + str(ii)+"\n")
      
      print(" orgi: "+str(orgi_i) +"\n")
      print(out)
      print(" output: "+str(out_i) +"\n")
      #print(" diff: %f %%\n"%diff )

      print(" chance in all: %f %%\n"%(float(pass_cnt2)*100/ii) )
      print(" pass 1 in all: %f %%\n"%(float(pass_cnt1)*100/ii) )
      print(" pass 1 catch chance: %f %%\n"%(float(pass_cnt1)*100/pass_cnt2) )
      print(" err_cnt: %f %%\n"%(float(err_cnt)*100/ii) )
      print(" refuse: %f %%\n"%(float(refuse_cnt)*100/ii) )
      
      image_data1 = 255 - BTCC_pro_market_img[ii]*255
      image_data = BTCC_pro_market_img[ii]
      #print(image_data)
      #img = np.array([image_data,image_data,image_data])
      bigimg = cv2.resize(image_data1, (0,0), fx=2, fy=2) 
      cv2.imshow("input_img", image_data1)
      #cv2.resizeWindow("input_img", 120*2+20, 90*2+20)
      cv2.imwrite("input_img.jpg", image_data1)
      bbb='''
      plt.figure(2) 
      image_show = zeros((90,120,3), dtype=float)
      for x in range(0,120):
        for y in range(0,90):
          pix = 1-image_data[y][x]
          if pix >1:
            pix =1
          if pix <0:
            pix =0
          image_show[y:x:1] = pix
          image_show[y:x:2] = pix
          image_show[y:x:3] = pix
      plt.imshow(image_show)'''
      axx='''
      x = range(0,264)
      x1 = range(0,120)
      plt.figure(1) 
      #plt.subplot(111)
      showdata = BTCC_pro_data[ii][2:266]
      showdata[120] = -100
      showdata1 = BTCC_pro_market_data[ii][0:120]
      plt.plot(x, showdata,x1,showdata1,'r')
      plt.ylim(-20, 60.)
      plt.show()
      
      while cv2.waitKey(5) != ord('n'):
       pass

    axx='' '
    #(use_data == uda_data  ) and
    diff = abs(out-y[ii])*100/y[ii]

    if (ys[ii][0] > 0.5):
      pass_cnt2 +=1

    if (ys[ii][0] > 0.5 and out>0.5):
      pass_cnt1 +=1

    if (ys[ii] < 0.5 and out>0.5):
      err_cnt +=1

    if (ys[ii] < 0.5 and out<0.5):
      refuse_cnt +=1

    if ii %100 ==0 :
      call('clear')
      print("now_index: " + str(ii)+"\n")
      
      print(" orgi: "+str(ys[ii]*100) +"\n")
      print(" output: "+str(out*100) +"\n")
      print(" diff: %f %%\n"%diff )

      print(" orgi in all: %f %%\n"%(float(pass_cnt2)*100/ii) )
      print(" pass 1 in all: %f %%\n"%(float(pass_cnt1)*100/ii) )
      print(" pass 1 catch chance: %f %%\n"%(float(pass_cnt1)*100/pass_cnt2) )
      print(" err_cnt: %f %%\n"%(float(err_cnt)*100/ii) )
      print(" refuse: %f %%\n"%(float(refuse_cnt)*100/ii) )
'''
    ii +=1




    


