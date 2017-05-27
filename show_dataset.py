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
saver.restore(sess, "save/model_1D.ckpt")
 
filename = []

def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]
 
idx = int(random.random(1)*1000)+ int(random.random()*20000)

for i in range(1,110):

  bench_size  = 10000
  
  BTCC_data.load_next_banch(idx,bench_size,1)
   
  idx += bench_size
 
  train_batch_pointer = 0
  banch_size = 1000
  predict_time = 1
  for i in range(0,int(100 * 100)):
      train_batch_pointer = i*banch_size
      xs,x_digit, ys,learn_r2,yl = BTCC_data.LoadTrainBatch(train_batch_pointer,banch_size,predict_time)
      #print(dataimg)
      image_show = model.y.eval(feed_dict={model.x: xs,model.x_digit:x_digit, model.y_: ys,model.yl: yl, model.keep_prob: 1.0})[0]
      image_show = image_show.reshape(12,256)
      print(image_show)
      print(image_show.shape)
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
