import tensorflow as tf
import scipy

def weight_variable(shape,name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name=name)

def bias_variable(shape,name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name=name)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride,1], padding='SAME')

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride,1], padding='SAME')
  
def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

mshape = 266 # (28+24+24+120)*2+24+7+31+12+2
x = tf.placeholder(tf.float32, shape=[None, 120,120])
x_digit = tf.placeholder(tf.float32, shape=[None, 266])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

variable_summaries(x,"placeholder_x")
variable_summaries(y_,"placeholder_y")

x_image = tf.reshape(x,[-1,120,120,1])
#x_image1 = tf.matmul(x_image)
#tf.image_summary("x_image", x_image1,max_images=10)

#first convolutional layer
W_conv1 = weight_variable([6, 6, 1, 196],name='W_conv1')
b_conv1 = bias_variable([196],name='b_conv1')

#with tf.name_scope('Layer0'):
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 3) + b_conv1)
#variable_summaries(h_conv1,"h_conv1")

#tf.image_summary("h_conv1", h_conv1)

tf.histogram_summary('W_conv1', W_conv1)
tf.histogram_summary('b_conv1', b_conv1)


W_conv2 = weight_variable([4, 4, 1, 196],name='W_conv2')
b_conv2 = bias_variable([196],name='b_conv2')

#with tf.name_scope('Layer0'):
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)


W_conv3 = weight_variable([3, 3, 1, 196],name='W_conv3')
b_conv3 = bias_variable([196],name='b_conv3')

#with tf.name_scope('Layer0'):
h_conv3 = tf.nn.relu(conv2d(h_conv1, W_conv3, 1) + b_conv3)

#FCL 1
W_fc1 = weight_variable([298116, 512],name="W_fc1")
b_fc1 = bias_variable([512],name="b_fc1")

tf.histogram_summary('W_fc1', W_fc1)
tf.histogram_summary('b_fc1', b_fc1)

h_conv5_flat = tf.reshape(h_conv1, [-1, 130732])#76636

with tf.name_scope('Layer1'):
	h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)#

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


bbbb='''
W_fc2 = weight_variable([1024, 1024])
b_fc2 = bias_variable([1024])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
'''


W_fc7 = weight_variable([512, 3],name='W_fc7')
b_fc7 = bias_variable([3],name='b_fc7')

tf.histogram_summary('W_fc7', W_fc7)
tf.histogram_summary('b_fc7', b_fc7)

h_fc7 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc7) + b_fc7)

h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)


#start digital
W_fc3 = weight_variable([266, 1024],name='W_fc3')
b_fc3 = bias_variable([1024],name='b_fc3')

h_fc3 = tf.nn.relu(tf.matmul(x_digit, W_fc3) + b_fc3)

h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

W_fc4 = weight_variable([1024, 3],name='W_fc4')
b_fc4 = bias_variable([3],name='b_fc4')

h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

#end digital
 


#Output
W_fc8 = weight_variable([360, 3],name='W_fc8')
b_fc8 = bias_variable([3],name='b_fc8')

tf.histogram_summary('W_fc8', W_fc8)
tf.histogram_summary('b_fc8', b_fc8)

h_fc8 = tf.matmul(h_fc7_drop, W_fc8) + b_fc8 #scale the atan outputs

W_add = bias_variable([1],name='W_add')
tf.histogram_summary('W_add', W_add)

with tf.name_scope('Layer2'):
	y=tf.add(tf.matmul(h_fc7_drop,1-W_add),tf.matmul(h_fc8,W_add))
