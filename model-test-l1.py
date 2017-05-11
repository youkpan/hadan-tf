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
W_fc1 = weight_variable([mshape, 4096])
b_fc1 = bias_variable([4096])

h_conv5_flat = tf.reshape(x_image, [-1, mshape])

h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)#

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

 
#Output
W_fc8 = weight_variable([4096, 3])
b_fc8 = bias_variable([3])

y = tf.matmul(h_fc1_drop, W_fc8) + b_fc8 #scale the atan outputs
