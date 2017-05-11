import tensorflow as tf
import scipy
import scipy.misc
from numpy  import *

n_input = 120 
n_classes = 2 

mshape = 268 # (28+24+24+120)*2+24+7+31+12+2


_weights = {
    'wc11': tf.Variable(tf.random_normal([5, 1, 64])),
    'wc12': tf.Variable(tf.random_normal([3, 64, 64])),
    'wc13': tf.Variable(tf.random_normal([3, 64, 64])),
    'wc21': tf.Variable(tf.random_normal([3, 64, 64])),
    'wc22': tf.Variable(tf.random_normal([3, 64, 64])),
    'wc23': tf.Variable(tf.random_normal([3, 64, 64])),
    #'wc31': tf.Variable(tf.random_normal([3, 3, 196, 256])),
    #'wc32': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    #'wc33': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'wd1': tf.Variable(tf.random_normal([30*64, 512])),
    'wd2': tf.Variable(tf.random_normal([512, 512])),
    'out': tf.Variable(tf.random_normal([512, 3]))
}
_biases = {
    'bc11': tf.Variable(tf.random_normal([64])),
    'bc12': tf.Variable(tf.random_normal([64])),
    'bc13': tf.Variable(tf.random_normal([64])),
    'bc21': tf.Variable(tf.random_normal([64])),
    'bc22': tf.Variable(tf.random_normal([64])),
    'bc23': tf.Variable(tf.random_normal([64])),
    #'bc31': tf.Variable(tf.random_normal([256])),
    #'bc32': tf.Variable(tf.random_normal([256])),
    #'bc33': tf.Variable(tf.random_normal([256])),

    'bd1': tf.Variable(tf.random_normal([512])),
    'bd2': tf.Variable(tf.random_normal([512])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


x = tf.placeholder(tf.float32, shape=[None, n_input])
x_digit = tf.placeholder(tf.float32, shape=[None, mshape])
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])

keep_prob = tf.placeholder(tf.float32)

def prelu(x,name=0,alphas=0.25):
  if name==0:
    return tf.nn.relu(x) #+ tf.mul(alphas, (x - tf.abs(x))) * 0.5
  else:
    return tf.nn.relu(x,name=name) #+ tf.mul(alphas, (x - tf.abs(x))) * 0.5


def conv2d2(name, l_input, w, b,s=1):
    
      return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, s, s, 1], padding='SAME'),b), name=name)

def conv1d2(name, l_input, w, b,s=1):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv1d(l_input,w,stride=s, padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


def weight_variable(shape,name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name=name)

def bias_variable(shape,name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name=name)


def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

def conv1d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, 1], padding='VALID')

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

#variable_summaries(x,"placeholder_x")
#variable_summaries(y_,"placeholder_y")

_X = tf.reshape(x, shape=[-1, n_input, 1])

#  6x6 s 3
conv1 = conv1d2('conv11', _X, _weights['wc11'], _biases['bc11'],2)
conv1 = conv1d2('conv12', conv1, _weights['wc12'], _biases['bc12'],2)
conv1 = conv1d2('conv13', conv1, _weights['wc13'], _biases['bc13'],1)
#  60x60+
#pool1 = max_pool('pool1', conv1, k=2)
# 
#norm1 = norm('norm1', conv1, lsize=4)
# Dropout
norm1 = tf.nn.dropout(conv1, keep_prob)

# 
conv2 = conv1d2('conv21', norm1, _weights['wc21'], _biases['bc21'],1)
conv2 = conv1d2('conv22', conv2, _weights['wc22'], _biases['bc22'],1)
conv2 = conv1d2('conv23', conv2, _weights['wc23'], _biases['bc23'],1)

#  to 30x30
#pool2 = max_pool('pool2', conv2, k=2)
# 
#norm2 = norm('norm2', pool2, lsize=4)
# 
norm2 = tf.nn.dropout(conv2, keep_prob)

# 
#conv3 = conv2d2('conv31', norm2, _weights['wc31'], _biases['bc31'],1)
#conv3 = conv2d2('conv32', conv3, _weights['wc32'], _biases['bc32'],1)
#conv3 = conv2d2('conv33', conv3, _weights['wc33'], _biases['bc33'],1)
#  to 15x15
#pool3 = max_pool('pool3', conv3, k=2)
# 
#norm3 = norm('norm3', pool3, lsize=4)
# Dropout
#norm3 = tf.nn.dropout(norm3, keep_prob)

# 
dense1 = tf.reshape(norm2, [-1, _weights['wd1'].get_shape().as_list()[0]]) 
dense1 = prelu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') 
# 

# 
#out = tf.matmul(dense2, _weights['out']) + _biases['out']

#start digital
W_fc2 = weight_variable([mshape, 512],name='W_fc2')
b_fc2 = bias_variable([512],name='b_fc2')

h_fc2 = prelu(tf.nn.bias_add(tf.matmul(x_digit, W_fc2) , b_fc2))

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

dense2 = prelu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
#print("dense1:  shape  \n",dense1.get_shape(),tf.shape(dense1))
#print("h_fc2_drop:  shape  \n",h_fc2_drop.get_shape(),tf.shape(h_fc2_drop))
#out = vstack ((dense2,h_fc3_drop))
concat = tf.reshape(tf.concat( [dense1, h_fc2_drop] ,1), shape=[-1,1024])

W_fc3 = weight_variable([1024, 512],name='W_fc3')
b_fc3 = bias_variable([512],name='b_fc3')

h_fc3 = prelu(tf.nn.bias_add(tf.matmul(concat, W_fc3) , b_fc3))

h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

W_fc5 = weight_variable([512, 512],name='W_fc5')
b_fc5 = bias_variable([512],name='b_fc5')

h_fc5 = prelu(tf.nn.bias_add(tf.matmul(h_fc3_drop, W_fc5) , b_fc5))

h_fc5_drop = tf.nn.dropout(h_fc5, keep_prob)

W_fc6 = weight_variable([512, 128],name='W_fc6')
b_fc6 = bias_variable([128],name='b_fc6')

h_fc6 = prelu(tf.nn.bias_add(tf.matmul(h_fc5_drop, W_fc6) , b_fc6))

h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob)

W_fc4 = weight_variable([128, n_classes],name='W_fc4')
b_fc4 = bias_variable([n_classes],name='b_fc4')

h_fc4 = tf.nn.bias_add(tf.matmul(h_fc6_drop, W_fc4) , b_fc4)

#end digital

W_add = bias_variable([1],name='W_add')
#tf.histogram_summary('W_add', W_add)

#with tf.name_scope('Layer2'):
y=h_fc4 #h_fc4*(1-W_add)+(out*W_add)
  #y=tf.add(tf.matmul(h_fc4,1-W_add),tf.matmul(h_fc4,W_add))


# 