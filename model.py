import tensorflow as tf
import scipy


n_input = 120 
n_classes = 3 

mshape = 266 # (28+24+24+120)*2+24+7+31+12+2


_weights = {
    'wc11': tf.Variable(tf.random_normal([4, 4, 1, 96])),
    #'wc12': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    #'wc13': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    #'wc21': tf.Variable(tf.random_normal([3, 3, 128, 196])),
    #'wc22': tf.Variable(tf.random_normal([3, 3, 196, 196])),
    #'wc23': tf.Variable(tf.random_normal([3, 3, 196, 196])),
    #'wc31': tf.Variable(tf.random_normal([3, 3, 196, 256])),
    #'wc32': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    #'wc33': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'wd1': tf.Variable(tf.random_normal([20*20*96, 512])),
    'wd2': tf.Variable(tf.random_normal([512, 512])),
    'out': tf.Variable(tf.random_normal([512, 3]))
}
_biases = {
    'bc11': tf.Variable(tf.random_normal([96])),
    'bc12': tf.Variable(tf.random_normal([256])),
    'bc13': tf.Variable(tf.random_normal([128])),
    'bc21': tf.Variable(tf.random_normal([196])),
    'bc22': tf.Variable(tf.random_normal([196])),
    'bc23': tf.Variable(tf.random_normal([196])),
    #'bc31': tf.Variable(tf.random_normal([256])),
    #'bc32': tf.Variable(tf.random_normal([256])),
    #'bc33': tf.Variable(tf.random_normal([256])),

    'bd1': tf.Variable(tf.random_normal([512])),
    'bd2': tf.Variable(tf.random_normal([512])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


x = tf.placeholder(tf.float32, shape=[None, n_input,n_input])
x_digit = tf.placeholder(tf.float32, shape=[None, mshape])
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])

keep_prob = tf.placeholder(tf.float32)

def prelu(x,name=0,alphas=0.25):
  if name==0:
    return tf.nn.relu(x) #+ tf.mul(alphas, (x - tf.abs(x))) * 0.5
  else:
    return tf.nn.relu(x,name=name) #+ tf.mul(alphas, (x - tf.abs(x))) * 0.5


def conv2d2(name, l_input, w, b,s=1):
    
      return prelu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, s, s, 1], padding='SAME'),b), name=name)


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

variable_summaries(x,"placeholder_x")
variable_summaries(y_,"placeholder_y")

_X = tf.reshape(x, shape=[-1, 120, 120, 1])

#  6x6 s 3
conv1 = conv2d2('conv11', _X, _weights['wc11'], _biases['bc11'],3)
#conv1 = conv2d2('conv12', conv1, _weights['wc12'], _biases['bc12'],1)
#conv1 = conv2d2('conv13', conv1, _weights['wc13'], _biases['bc13'],1)
#  60x60
pool1 = max_pool('pool1', conv1, k=2)
# 
#norm1 = norm('norm1', pool1, lsize=4)
# Dropout
norm1 = tf.nn.dropout(pool1, keep_prob)

# 
#conv2 = conv2d2('conv21', norm1, _weights['wc21'], _biases['bc21'],1)
#conv2 = conv2d2('conv22', conv2, _weights['wc22'], _biases['bc22'],1)
#conv2 = conv2d2('conv23', conv2, _weights['wc23'], _biases['bc23'],1)

#  to 30x30
#pool2 = max_pool('pool2', conv2, k=2)
# 
#norm2 = norm('norm2', pool2, lsize=4)
# 
#norm2 = tf.nn.dropout(norm2, keep_prob)

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
dense1 = tf.reshape(norm1, [-1, _weights['wd1'].get_shape().as_list()[0]]) 
dense1 = prelu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') 
# 
dense2 = prelu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation

# 
out = tf.matmul(dense2, _weights['out']) + _biases['out']

#start digital
W_fc2 = weight_variable([266, 512],name='W_fc2')
b_fc2 = bias_variable([512],name='b_fc2')

h_fc2 = prelu(tf.nn.bias_add(tf.matmul(x_digit, W_fc2) , b_fc2))

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([512, 512],name='W_fc3')
b_fc3 = bias_variable([512],name='b_fc3')

h_fc3 = prelu(tf.nn.bias_add(tf.matmul(h_fc2_drop, W_fc3) , b_fc3))

h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

W_fc4 = weight_variable([512, 3],name='W_fc4')
b_fc4 = bias_variable([3],name='b_fc4')

h_fc4 = tf.nn.bias_add(tf.matmul(h_fc3_drop, W_fc4) , b_fc4)

#end digital

W_add = bias_variable([1],name='W_add')
tf.histogram_summary('W_add', W_add)

with tf.name_scope('Layer2'):
  y=h_fc4*(1-W_add)+(out*W_add)
  #y=tf.add(tf.matmul(h_fc4,1-W_add),tf.matmul(h_fc4,W_add))


# 