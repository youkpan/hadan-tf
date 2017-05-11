import tensorflow as tf
import scipy

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

mshape = 266 # (28+24+24+120)*2+24+7+31+12+2
x = tf.placeholder(tf.float32, shape=[None, mshape])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

x_image = x

#FCL 1
W_fc1 = weight_variable([mshape, 1024])
b_fc1 = bias_variable([1024])

h_conv5_flat = tf.reshape(x_image, [-1, mshape])

h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)#

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024, 1024])
b_fc2 = bias_variable([1024])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


W_fc3 = weight_variable([1024, 1024])
b_fc3 = bias_variable([1024])

h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)


W_fc4 = weight_variable([1024, 1024])
b_fc4 = bias_variable([1024])

h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)


 
W_fc7 = weight_variable([1024, 64])
b_fc7 = bias_variable([64])

h_fc7 = tf.nn.relu(tf.matmul(h_fc4_drop, W_fc7) + b_fc7)

h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)


#Output
W_fc8 = weight_variable([64, 3])
b_fc8 = bias_variable([3])

y = tf.matmul(h_fc7_drop, W_fc8) + b_fc8 #scale the atan outputs
