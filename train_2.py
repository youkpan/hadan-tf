import os
import tensorflow as tf
import scipy.misc
import BTCC_data
import model
#import cv2
import random
import numpy as np

LOGDIR = './save'

sess = tf.InteractiveSession()

#loss = tf.reduce_mean(tf.square(tf.sub(model.y_, model.y)))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
#hidden = tf.nn.relu(ys)
#logits = tf.matmul(hidden, weights2) + biases2
#print(FLAGS)
learning_rate = tf.placeholder(tf.float32, shape=[])
#loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y)))
#loss = tf.reduce_mean((model.y_[0][0]- model.y[0][0]) **2 + (model.y_[0][1]- model.y[0][1])**2)
loss = tf.reduce_sum( tf.abs(tf.subtract(model.y,   model.y_)))**2
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits= model.y,  labels= model.y_))

#loss_summary = tf.scalar_summary("loss", loss)

# Optimizer.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Predictions for the training, validation, and test data.
train_prediction = tf.nn.softmax(model.y)

sess.run(tf.global_variables_initializer())
#self.init = tf.initialize_variables(tf.all_variables(), name="nInit")

saver = tf.train.Saver()
#saver.restore(sess,LOGDIR+"/model_t_same_dict.ckpt")
print("Model restore") 

#merged_summary_op = tf.merge_all_summaries()


banch_size = 100

loop_cnt = 10000
predict_time = 5
max_cnt = 1300000 # BTCC_data.get_data_size()
start_it = int(random.random()*20)+20
step_times = int(700/banch_size)

#img = scipy.misc.imread('steering_wheel_image.png', mode="RGB")
#cv2.imshow("steering wheel", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def accuracy0(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


#save_pb(saver)
def tf_sgd_relu_nn2(sess1=0):
  global sess
  global max_cnt
  if(sess1!=0):
    sess =sess1
  last_loss=0
  loss_change_cnt = 0
  learn_r = 0.002/banch_size
  banch_i = 10000
  accuracy2 = 0
  banch_i = int(random.random()*max_cnt)
  BTCC_data.load_next_banch(banch_i,loop_cnt,predict_time)
  key = ''
  #summary_writer = tf.train.SummaryWriter('tf_train', sess.graph)
  #tf.scalar_summary("accuracy", accuracy2)

  #train over the dataset about 30 times
  for i in range(start_it,int(BTCC_data.num_images * 10000000000)):
    train_batch_pointer = i*banch_size
    xs,x_digit, ys,yl,Word_mark,diff_w,sentence = BTCC_data.LoadTrainBatch(train_batch_pointer,banch_size,predict_time)
    #print(x_digit[1:2].shape,xs[1:2].shape,ys[1:2].shape)
    #print("training: %d" % i)
    #train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.5})
    feed_dict = {model.x: xs,model.x_digit:x_digit, model.y_: ys,model.yl: yl, 
    model.Word_mark:Word_mark,model.diff_w:diff_w, model.keep_prob: 0.7,learning_rate:learn_r}
    #{tf_train_dataset: batch_data, tf_train_labels: batch_labels}
    _, l, predictions = sess.run(
        [optimizer, loss, train_prediction], feed_dict=feed_dict)
    #print(np.sum(ys[1]))
    if last_loss ==0 and i>5:
      last_loss = l
    #key = cv2.waitKey(5)

    last_loss = last_loss*0.98+0.02*l
    print("Minibatch loss at step %d: %f  loss_avg:%f" % (i, l,last_loss))
    if (i % (20*step_times) == 1) or (key == ord('t')):
      
      accuracy2 = accuracy(predictions, ys)
      print("Minibatch accuracy: %.1f%%" % accuracy2)
      print("last_loss ",last_loss)
      #if (l<last_loss ):
      #  last_loss = l
      if (l>last_loss ):
        loss_change_cnt +=1
        print("---loss not decress")
        if (loss_change_cnt >2):
          loss_change_cnt = 0
          #last_loss = l
          if learn_r > 0.00005:
            learn_r = 0.7*learn_r
            print("learn rate changed:%f"%learn_r)
    #else:
    #  last_loss = last_loss*0.95+0.05*l
      #if loss_change_cnt>1:
       # loss_change_cnt -=1
        
    #print("step run over")
    
    if (i % (10*step_times) == 1) or (key == ord('t')):
      val_batch_pointer = i*banch_size
      xs,x_digit, ys,yl ,Word_mark,diff_w, sentence= BTCC_data.LoadValBatch(val_batch_pointer ,banch_size ,predict_time)
      mloss = loss.eval(feed_dict={model.x:xs,model.x_digit:x_digit, 
        model.y_: ys,model.yl: yl,model.Word_mark:Word_mark,model.diff_w:diff_w,
        model.keep_prob: 1.0})
      print("step %d, val loss %g"%(i, mloss))

      if (mloss < 0.02):
        checkpoint_path = os.path.join(LOGDIR, "%g-model_t_same_dict.ckpt"%mloss)
        filename = saver.save(sess, checkpoint_path)
        print("Model saved in file: %s" % filename)
        break
    '''
    if (i % 80 == 100 or key == ord('l')):
      #accuracy_summary = tf.scalar_summary("accuracy", accuracy)
      print("saving summary")
      feed_dict = {model.x: xs ,model.x_digit:x_digit, model.y_: ys, model.keep_prob: 1.0,learning_rate:0.000001}
      _,l,t,summary_str = sess.run([optimizer, loss, train_prediction,merged_summary_op],feed_dict=feed_dict)
      accuracy2 = accuracy(predictions, ys)
      summary_writer.add_summary(summary_str, i)
      print("log saved in file: tf_train" )
    '''
    if (i % (2*loop_cnt/banch_size) == (2*loop_cnt/banch_size)-1 or key == ord('n')):
      banch_i += loop_cnt
      #banch_i = int(random.random()*max_cnt)
      
      if(banch_i>max_cnt):
        banch_i = 0
      BTCC_data.load_next_banch(banch_i%max_cnt,loop_cnt,predict_time)

    if (i % (200*step_times) == 100*step_times or (key == ord('s'))):
      print("saving model")
      if not os.path.exists(LOGDIR):
              os.makedirs(LOGDIR)
      checkpoint_path = os.path.join(LOGDIR, "model_t_same_dict.ckpt")
      
      filename = saver.save(sess, checkpoint_path)
      print("Model saved in file: %s" % filename)
 

if __name__ == '__main__':
    tf_sgd_relu_nn2()
