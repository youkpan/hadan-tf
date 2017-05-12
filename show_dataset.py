import tensorflow as tf
import scipy.misc
import scipy.ndimage
import model
#import cv2
from subprocess import call
from numpy  import *
import numpy as np
import matplotlib.pyplot as plt
import BTCC_data
import os

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model_1D.ckpt")

#img = scipy.misc.imread('steering_wheel_image.png', mode="RGB")
#cv2.imshow("steering wheel", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


smoothed_angle = 0

i = 0
start_index = 0
differ = 0.0
differ_s = 0.0
imgblank = zeros((5,120), dtype=float)


uda_data = 1
game_data = 2
default_data = 0
use_data = game_data

filename = []

def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

def get_showdata(data,idx,num):
  t=zeros((num), dtype=float)
  for i in range(0,num):
      t[i]=(data[i][idx:idx+1])[0]

  return t

def buy_BTC(money,price):
  if money >0:
    return money/price
  else:
    return 0


learning_rate = tf.placeholder(tf.float32, shape=[])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits= model.y,  labels= model.y_))

#loss = tf.reduce_mean((model.y_[0][0]- model.y[0][0])**2 + (model.y_[0][1]- model.y[0][1])**2)
# Optimizer.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
train_prediction = tf.nn.softmax(model.y)

def calc_profit(buy_lost,sell_lost,money_init,buy_once,sell_once,
  buy_wait,sell_wait, sell_limit_diff
  ,judge_minu,quick_move,slow_move):
  ii=0
  pass_cnt2=1
  pass_cnt1=0
  refuse_cnt =0
  err_cnt =0
  start_cnt = 0
  global bench_size

  #money_init = 10000
  money = money_init

  mmoney=0
  mBTC = 0.0
  bench_size = bench_size
  profit_data=zeros(bench_size, dtype=float)
  own_BTC_data=zeros(bench_size, dtype=float)
  own_money_data=zeros(bench_size, dtype=float)
  
  buy_times = 0
  sell_times=0
  price_sum =0
  amount_sum = 0
  datanum = bench_size
  print("bench_size")
  print(bench_size)
  profit = 0
  min_cnt = 0
  min_profit = 100
  sell_cnt = 0
  p_buy_show = []
  op_action_show = []
  avg_7_2 =zeros((7), dtype=float)
  avg_7_2_i = 0
  DIF= 0
  DEA = 0
  p_buy = 0
  buy = 1
  sell = -1
  action = 0
  last_calc_m = 0
  dataimg = []
  last_price = 100000
  x_digit = []
  dataimg = []
  p_buy = []
  dataimg_list= []
  x_digit_list = []
  action_list = []
  reward_list = []
  loop_cnt = 0
  while (ii < datanum):
      op_action = 0

      minu = ii
      price=BTCC_data.get_price(ii)

      if ii==0:
        sell_limit = int(price)+sell_limit_diff
        last_price = price

      amount=1
      price_sum += price*amount
      amount_sum += amount
      #print(minu - last_calc_m)
      if minu != last_calc_m:
        min_cnt +=1
      last_calc_m = minu

      if(min_cnt == judge_minu ):
        min_cnt = 0
        if amount_sum>0:
          #print ('in')
          price = price_sum/amount_sum
          price_sum =0
          amount_sum= 0
          reward = 1

          reward = price/last_price
          reward_list.append(reward)
          train_size = 20
          if(loop_cnt == train_size and len(x_digit)>0):
            ys_list = []
            loop_cnt = 0
            j = 0;
            
            train_batch = 1
            while( j < train_size-1):
              #if(fabs(reward_list[j]-1)>0.002):
                #learn_r = (fabs(reward_list[j])-1.0)*1000/20/50/6 #* (fabs(p_buy[0]-p_buy[1])*1000)

                if(action_list[j] == buy):
                  x_digit_list[j][266] = 1.0
                  x_digit_list[j][267] = 0.0
                  
                else:
                  x_digit_list[j][266] = 0.0
                  x_digit_list[j][267] = 1.0
                
                if (float(reward_list[j+1]) < 1.0):

                  if(action_list[j] == buy):
                    x_digit_list[j][266] = 0.0
                    x_digit_list[j][267] = 1.0
                  else:
                    x_digit_list[j][266] = 1.0
                    x_digit_list[j][267] = 0.0
                mutitimes = fabs(reward_list[j+1]-1)*1000 /20  #* (fabs(p_buy[0]-p_buy[1])*10)
                if(mutitimes >1): 
                  mutitimes = 1
                
                ys = [x_digit_list[j][266]*mutitimes,x_digit_list[j][267]*mutitimes]
                #ys = [0,1]
                ys_list.append(ys)
                #print(ys)
                #print(ys_list)
                j+=1

            dataimg_list2 = []
            x_digit_list2 = []
            ys_list2 = []
            reward_sum  =  0
            for i in range(0,train_size-1) :
              if(ys_list[i][0]+ys_list[i][1] < 0.1 ):#or action_list[i]==0):
                continue
              dataimg_list2.append(dataimg_list[i])
              x_digit_list2.append(x_digit_list[i])
              ys_list2.append(ys_list[i])
              reward_sum += reward_list[i+1]
              print("train : ",ys_list[i]," action:",action_list[i]," reward ",reward_list[i+1]) 

            if(train_batch == 1 and len(x_digit_list2)>0):
              learn_r = 0.00005 + fabs(reward_sum / len(x_digit_list2)-1)/200
              feed_dict = {model.x: dataimg_list2,model.x_digit:x_digit_list2, model.y_: ys_list2, model.keep_prob: 1.0,learning_rate:learn_r}
              _, l, predictions = sess.run(
                  [optimizer, loss, train_prediction], feed_dict=feed_dict)
              print("training~ learn_r",learn_r)

            dataimg_list= []
            x_digit_list = []
            action_list = []
            reward_list = []
            ys_list = []

          last_price = price
          loop_cnt +=1

          x_digit = BTCC_data.get_xdigit(ii)
          x_digit_list.append(x_digit)
          if(action == buy):
            x_digit[266] = 1.0
            x_digit[267] = 0.0
            action_s = 'buy'
          else:
            x_digit[266] = 0.0
            x_digit[267] = 1.0
            action_s = 'sell'
          #print(x_digit)

          dataimg = BTCC_data.get_data_img(ii)
          dataimg_list.append(dataimg)
          
          #print(dataimg)
          p_buy = model.y.eval(feed_dict={model.x: [dataimg],model.x_digit: [x_digit], model.keep_prob: 1.0})[0]

          if(int(ii/100)%2 ==1):
            print("eval out : ",p_buy," last action:",action_s," reward ",reward) 

          if( p_buy[0] > p_buy[1] ):
            start_cnt +=1
            
          else:
            start_cnt -=1
            sell_cnt+=1
            
          action = 0
          p_buy_show.append(p_buy[0])
          #or start_cnt< 0-sell_wait
          if (start_cnt<0-sell_wait  or (sell_limit_diff>0 and price < sell_limit)):
            if(mBTC>0):
              sBTC = sell_once /price
              if sBTC > mBTC:
                sBTC = mBTC
              money += sBTC*(price-sell_lost)
              mBTC -= sBTC
              sell_times +=1
              op_action = -100
              sell_cnt = 0
              action = sell
            start_cnt =0

          if(start_cnt>buy_wait):
            if(money>0):
              bmoney = buy_once
              if bmoney >money:
                bmoney = money
              mBTC += buy_BTC(bmoney,(price+buy_lost))
              money -= bmoney
              sell_limit = price + sell_limit_diff
              buy_times +=1
              op_action = 100
              action = buy
              sell_cnt = 0
              last_buy_price = price
            start_cnt = 0

          action_list.append(action)

        op_action_show.append(op_action)
      mmoney = money+mBTC*price
      profit =float(mmoney)*100/money_init
      profit_data[ii] = mmoney
      own_BTC_data[ii] = mBTC*price
      own_money_data[ii] = money

      if profit<min_profit:
        min_profit = profit

      if ii % (datanum) ==(datanum-1) :

        #call('clear')
        print("now_index: " + str(ii)+"\n")
        print("buy_times: " + str(buy_times)+"\n")
        print("sell_times: " + str(sell_times)+"\n")
        print("sell_limit: " + str(sell_limit)+"\n")
        print("min_profit: " + str(min_profit)+"\n")
        print("mBTC: " + str(mBTC)+"\n")
        print("money: " + str(money)+"\n")
        print(" profit: %f %%\n"%(profit) )

        shownum= ii-10
        x1 = range(0,shownum)
        x2 = range(0,datanum)
        #plt.figure(1) 
        imgplot = plt.imshow(BTCC_data.get_data_img0(ii))
        fig = plt.figure()
        ax1 = fig.add_subplot(411)
        #plt.subplot(121) 
        #plt.subplot(111)
        start_i =ii-shownum
        sd=BTCC_data.get_price_list()[0:shownum]
        #print(shape(sd))
        #print(shape(sd[0:120,5:6]))
        showdata1 = sd #get_showdata(sd,5,shownum)
        #showdata2 = get_showdata(sd,6,shownum)
        #showdata3 = get_showdata(sd,7,shownum)
        #showdata4 = get_showdata(sd,8,shownum)
        #showdata5 = get_showdata(sd,11,shownum)

        ax1.plot(x1,showdata1)#x1,showdata2,x1,showdata3,
         # x1,showdata4,

        #plt.subplot(122) 
        ax2 = fig.add_subplot(412)
        #print(shape(profit_data[-1200:]))
        ax2.plot(x2,profit_data,x2,own_BTC_data,x2,own_money_data,'r')

        #plt.subplot(123) 
        ax3 = fig.add_subplot(413)
        x4= range(0,len(p_buy_show))
        #plt.figure(2) 
        ax3.plot(x4,p_buy_show,'b')
        
        ax4 = fig.add_subplot(414)
        x5= range(0,len(op_action_show))
        #plt.figure(2) 
        ax4.plot(x5,op_action_show,'r')

        #plt.ylim(-20, 60.)
        plt.show()
     
      ii +=1

  minetes = ii
  #<120 1-0.3*x/120 >120 80*x^-1+0.05
  if(buy_times+sell_times==0):
    buy_times=1
  minetes_per_op = minetes/(buy_times+sell_times)
  if(minetes_per_op <120):
    op_times_score = 1-0.3*minetes_per_op/120
  else:
    op_times_score = 80/minetes_per_op + 0.05
  min_profit_score = (min_profit-90)/100.0
  days = minetes/(24*60)
  profit_score = (profit)/(1.0*days)
  print("op_times_scroe: " + str(op_times_score)+"\n")
  print("min_profit_score: " + str(min_profit_score)+"\n")
  print("profit_score: " + str(profit_score)+",days:"+str(days)+"\n")
  score = profit_score*0.7+min_profit_score*0.2+op_times_score*0.1

  return {'profit':profit,'min_profit':min_profit,'buy_times':buy_times,
  'sell_times':sell_times,'score':score}

judge_minu = 1
idx = int(random.random()*judge_minu)
for i in range(1,110):

  bench_size  = 10000
  BTCC_data.load_next_banch(idx,bench_size,judge_minu)
  idx += bench_size
  profit = calc_profit(buy_lost=0,sell_lost=0,money_init=10000,buy_once=10000,
      sell_once=10000,buy_wait=1,sell_wait=1,sell_limit_diff=0,judge_minu=judge_minu,quick_move=10,slow_move=40)

  print(profit)

  if(i%10 ==9):
    print("saving model")
    LOGDIR = './save'
    if not os.path.exists(LOGDIR):
            os.makedirs(LOGDIR)
    checkpoint_path = os.path.join(LOGDIR, "model2.ckpt")
    
    filename = saver.save(sess, checkpoint_path)
    print("Model saved in file: %s" % filename)
