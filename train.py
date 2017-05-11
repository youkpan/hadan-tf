import os
import tensorflow as tf
import scipy.misc
import BTCC_data
import model
import cv2

LOGDIR = './save'

sess = tf.InteractiveSession()

loss = tf.reduce_mean(tf.square(tf.sub(model.y_, model.y)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.initialize_all_variables())
#self.init = tf.initialize_variables(tf.all_variables(), name="nInit")

saver = tf.train.Saver()
saver.restore(sess,LOGDIR+"/model.ckpt")
print("Model restore") 

start_it = 0
iteration = 2000
global train_batch_pointer 

img = scipy.misc.imread('steering_wheel_image.png', mode="RGB")
cv2.imshow("steering wheel", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('bitcoin_logs', sess.graph)

#save_pb(saver)

#train over the dataset about 30 times
for i in range(start_it,int(BTCC_data.num_images * 100)):
  train_batch_pointer = i*iteration
  xs, ys = BTCC_data.LoadTrainBatch(train_batch_pointer,iteration)
  #print("training: %d" % i)
  train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.5})
  #print("step run over")
  key = cv2.waitKey(5)
  if (i % 20 == 1) or (key == ord('t')):
    val_batch_pointer = i*iteration
    xs, ys = BTCC_data.LoadValBatch(val_batch_pointer ,iteration)
    mloss = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    print("step %d, val loss %g"%(i, mloss))
    if (mloss < 0.02):
      checkpoint_path = os.path.join(LOGDIR, "%g-model.ckpt"%mloss)
      filename = saver.save(sess, checkpoint_path)
      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
      filename = saver.save(sess, checkpoint_path)
      print("Model saved in file: %s" % filename)
      break

  #if (i % 100 == 10 or (key == ord('l'))):
    #summary_str = sess.run(merged_summary_op)
    #summary_writer.add_summary(summary_str, i)
    #print("log saved in file: bitcoin_logs" )
  if (i % 500 == 200 or (key == ord('s'))):
    print("saving model")
    if not os.path.exists(LOGDIR):
            os.makedirs(LOGDIR)
    checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
    
    filename = saver.save(sess, checkpoint_path)
    print("Model saved in file: %s" % filename)
