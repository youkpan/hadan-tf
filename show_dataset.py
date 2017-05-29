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

image_show = np.array((1,2))
print(image_show.shape)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model_t_same_dict.ckpt")

train_prediction = tf.nn.softmax(model.y)
 
filename = []

#print(BTCC_data.dict_index_str)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits= model.y,  labels= model.y_))

def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]
 
idx = 0#int(random.random(1)*1000)+ int(random.random()*20000)
max_cnt = 1300000

for q in range(1,110):

  bench_size  = 10000
  
  BTCC_data.load_next_banch(idx,bench_size,1)
   
  idx += bench_size
 
  train_batch_pointer = 0

  predict_time = 1
  for i in range(0,int(100 * 100)):
      banch_size = 100
      train_batch_pointer = i*banch_size + int(random.random()*max_cnt)
      xs,x_digit, ys,yl,Word_mark,diff_w,sentence_data_t  = BTCC_data.LoadTrainBatch(train_batch_pointer,banch_size,predict_time)
      #print(x_digit)
      #print(image_show.shape)
      #x_digit2 = x_digit.reshape([3*12*2,16*8])
      #plt.imshow(x_digit2)
      #plt.show()
      #plt.imshow(ys[0].reshape([3*16,4*16]))

      l,predictions,yy = sess.run(
        [ loss , train_prediction ,model.y], feed_dict={model.x: xs,model.x_digit:x_digit, model.y_: ys,model.yl: yl, 
          model.Word_mark:Word_mark,model.diff_w:diff_w, model.keep_prob: 1.0})
      accuracy2 = accuracy(predictions, ys)
      

      y_get = model.y.eval(feed_dict={model.x: xs,model.x_digit:x_digit, model.y_: ys,model.yl: yl, 
        model.Word_mark:Word_mark,model.diff_w:diff_w, model.keep_prob: 1.0})[0]

      for ii in range (0,banch_size):

        result  = "false"
        if(ys[ii][0] > ys[ii][1]):
          result  = "true"

        if(yy[ii][0] > yy[ii][1] and result  == "true" or yy[ii][0] < yy[ii][1] and result  == "false"):
          print("Minibatch accuracy: %.2f%% , %.6f%%" % (accuracy2 , l))
          print("---------------------")
          print(sentence_data_t[ii][0])
          print(sentence_data_t[ii][1])
          print(sentence_data_t[ii][2])
          print("should: ",result )
          print("   YES    ************")

        #else:
          #print("   NO     !!!!!!")

      continue

      image_show = model.y.eval(feed_dict={model.x: xs,model.x_digit:x_digit, model.y_: ys,model.yl: yl, 
        model.Word_mark:Word_mark,model.diff_w:diff_w, model.keep_prob: 1.0})[0]

      image_show = image_show.reshape([12,256])
      #print(image_show.shape)

      sentence = image_show
      sentence_w =[]
      sentence_str = ""
      sentence_str2 = ""
      for j in range(0,12):
        top = np.dot(BTCC_data.dict_vector,sentence[j].reshape([256,1]))
        max_idx = 0
        max_s = top[0]
        #print(top)
        max_idx2 = 0
        #print(len(BTCC_data.dict_index_str))
        for k in range(0,len(BTCC_data.dict_index_str)):
          if top[k] > max_s :
            max_s = top[k]
            max_idx2 = max_idx
            max_idx = k

        #print(max_idx)
        word = BTCC_data.dict_index_str[max_idx]
        word2 =BTCC_data.dict_index_str[max_idx2]
        sentence_w.append(word)
        sentence_str += word
        sentence_str2 += word2

      print(sentence_str)
      print(sentence_str2)
      image_show = image_show.reshape(12,256)
      print(image_show)
      print(np.sum(image_show))
      image_show = image_show.reshape(3*16,4*16)

      plt.imshow(image_show)

      #plt.ylim(-20, 60.)
      plt.show()


  if(i%10 ==100):
    print("saving model")
    LOGDIR = './save'
    if not os.path.exists(LOGDIR):
            os.makedirs(LOGDIR)
    checkpoint_path = os.path.join(LOGDIR, "model2.ckpt")
    
    filename = saver.save(sess, checkpoint_path)
    print("Model saved in file: %s" % filename)
