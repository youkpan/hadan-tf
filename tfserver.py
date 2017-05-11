import tensorflow as tf
import os
import scipy.misc
import BTCC_data
import model
import cv2
import numpy as np

# cluster specification
parameter_servers = ["http://127.0.0.1:2222"]
workers = [ "http://127.0.0.1:2222", 
      "pc-03:2222",
      "pc-04:2222"]
#cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})


# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "127.0.0.1:2222",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "127.0.0.1:2223",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  
  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    #with tf.device('/cpu:0'):
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):
      learning_rate = tf.placeholder(tf.float32, shape=[])
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(model.y, model.y_))
        #import train_2 as mtrain
            # Build model...
      loss_summary = tf.scalar_summary("loss", loss)

      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(model.y)

      #sess.run(tf.initialize_all_variables())
      #self.init = tf.initialize_variables(tf.all_variables(), name="nInit")

      #saver = tf.train.Saver()
      #saver.restore(sess,LOGDIR+"/model.ckpt")
      print("Model restore") 

      merged_summary_op = tf.merge_all_summaries()

      start_it = 0
      iteration = 100


      img = scipy.misc.imread('steering_wheel_image.png', mode="RGB")
      cv2.imshow("steering wheel", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


      global_step = tf.Variable(0)

      #train_op = tf.train.AdagradOptimizer(0.01).minimize(
       #   loss, global_step=global_step)
      train_op = optimizer

      saver = tf.train.Saver()
      #summary_op = tf.merge_all_summaries()
      init_op = tf.initialize_all_variables()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="./train_logs2",
                             init_op=init_op,
                             summary_op=merged_summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600,
                             allow_soft_placement=True)

    # The supervisor takes care of session initialization and restoring from
    # a checkpoint.
    #config = tf.ConfigProto(allow_soft_placement=True)
    sess = sv.prepare_or_wait_for_session(server.target)

    # Start queue runners for the input pipelines (if any).
    sv.start_queue_runners(sess)

    last_loss=0
    loss_change_cnt = 0
    learn_r = 0.01
    banch_i = 65000
    accuracy2 = 0
    BTCC_data.load_next_banch(banch_i)
    summary_writer = tf.train.SummaryWriter('tf_train', sess.graph)
    tf.scalar_summary("accuracy", accuracy2)
    #train over the dataset about 30 times

    # Loop until the supervisor shuts down (or 1000000 steps have completed).
    step = 0
    i = start_it
    while not sv.should_stop() and step < 1000000:
      # Run a training step asynchronously.
      # See `tf.train.SyncReplicasOptimizer` for additional details on how to
      # perform *synchronous* training.
      #_, step = sess.run([train_op, global_step])
        
      train_batch_pointer = i*iteration
      xs,x_digit, ys = BTCC_data.LoadTrainBatch(train_batch_pointer,iteration)
      
      #print("training: %d" % i)
      #train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.5})
      feed_dict = {model.x: xs,model.x_digit:x_digit, model.y_: ys, model.keep_prob: 0.7,learning_rate:learn_r}
      #{tf_train_dataset: batch_data, tf_train_labels: batch_labels}
      _, l, predictions = sess.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)

      if last_loss ==0:
        last_loss = l
      key = cv2.waitKey(5)
      if (i % 20 == 1) or (key == ord('t')):
        print("Minibatch loss at step %d: %f  loss_avg:%f" % (i, l,last_loss))
        accuracy2 = accuracy(predictions, ys)
        print("Minibatch accuracy: %.1f%%" % accuracy2)
        
        if (l>last_loss ):
          loss_change_cnt +=1
          print("loss not decress")
          if (loss_change_cnt >7):
            loss_change_cnt = 0
            last_loss = l
            if learn_r > 0.00005:
              learn_r = 0.7*learn_r
              print("learn rate changed:%f"%learn_r)
        else:
          last_loss = last_loss*0.95+0.05*l
          if loss_change_cnt>1:
            loss_change_cnt -=1
          
      #print("step run over")
      
      if (i % 100 == 1) or (key == ord('t')):
        val_batch_pointer = i*iteration
        xs,x_digit, ys = BTCC_data.LoadValBatch(val_batch_pointer ,iteration)
        mloss = loss.eval(feed_dict={model.x:xs,model.x_digit:x_digit, model.y_: ys, model.keep_prob: 1.0})
        print("step %d, val loss %g"%(i, mloss))

        if (mloss < 0.02):
          checkpoint_path = os.path.join(LOGDIR, "%g-model.ckpt"%mloss)
          filename = saver.save(sess, checkpoint_path)
          checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
          filename = saver.save(sess, checkpoint_path)
          print("Model saved in file: %s" % filename)
          break

      if (i % 80 == 100 or key == ord('l')):
        #accuracy_summary = tf.scalar_summary("accuracy", accuracy)
        print("saving summary")
        feed_dict = {model.x: xs ,model.x_digit:x_digit, model.y_: ys, model.keep_prob: 1.0,learning_rate:0.000001}
        _,l,t,summary_str = sess.run([optimizer, loss, train_prediction,merged_summary_op],feed_dict=feed_dict)
        accuracy2 = accuracy(predictions, ys)
        summary_writer.add_summary(summary_str, i)
        print("log saved in file: tf_train" )

      if (i % 800 == 799 or key == ord('n')):
        banch_i += 20000
        if(banch_i>115000):
          banch_i = 0
        BTCC_data.load_next_banch(banch_i%115000)

      if (i % 600 == 400 or (key == ord('s'))):
        print("saving model")
        if not os.path.exists(LOGDIR):
                os.makedirs(LOGDIR)
        checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
        
        filename = saver.save(sess, checkpoint_path)
        print("Model saved in file: %s" % filename)
   


    sv.stop()


if __name__ == "__main__":
  tf.app.run()