import tensorflow as tf
import scipy
import scipy.misc
from numpy  import *

n_input = 256 
n_classes = 256*12 
n_vocab = 6204
mshape = 256*12*3 # (28+24+24+120)*2+24+7+31+12+2
word_len = 20
batch = 100

_weights = {
    'wc11': tf.Variable(tf.random_normal([1, 1, 32])),
    'wc12': tf.Variable(tf.random_normal([3, 1, 32])),
    'wc13': tf.Variable(tf.random_normal([5, 1, 32])),
    'wc21': tf.Variable(tf.random_normal([1, 96, 64])),
    'wc22': tf.Variable(tf.random_normal([3, 96, 64])),
    'wc23': tf.Variable(tf.random_normal([5, 96, 64])),
    'wc31': tf.Variable(tf.random_normal([1, 192, 96])),
    'wc32': tf.Variable(tf.random_normal([3, 192, 96])),
    'wc33': tf.Variable(tf.random_normal([5, 192, 96])),
    'wd1': tf.Variable(tf.random_normal([n_input , 5000])),
    'wd2': tf.Variable(tf.random_normal([5000, n_classes])),
    'out': tf.Variable(tf.random_normal([n_classes, n_classes]))
}
_biases = {
    'bc11': tf.Variable(tf.random_normal([32])),
    'bc12': tf.Variable(tf.random_normal([32])),
    'bc13': tf.Variable(tf.random_normal([32])),
    'bc21': tf.Variable(tf.random_normal([64])),
    'bc22': tf.Variable(tf.random_normal([64])),
    'bc23': tf.Variable(tf.random_normal([64])),
    'bc31': tf.Variable(tf.random_normal([96])),
    'bc32': tf.Variable(tf.random_normal([96])),
    'bc33': tf.Variable(tf.random_normal([96])),

    'bd1': tf.Variable(tf.random_normal([5000])),
    'bd2': tf.Variable(tf.random_normal([n_classes])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

x = tf.placeholder(tf.float32, shape=[None, n_input])
x_digit = tf.placeholder(tf.float32, shape=[36,batch,256])
y_ = tf.placeholder(tf.float32, shape=[batch,3072])
yl = tf.placeholder(tf.float32, shape=[None, n_classes])
#Wab = tf.placeholder(tf.float32, shape=[None, word_len])
#Vector_Word = tf.placeholder(tf.float32, shape=[None, n_vocab])

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
  initial = tf.constant(0.00, shape=shape) #tf.truncated_normal(shape, stddev=0.1) #
  #initial = tf.random_normal(shape)
  return tf.Variable(initial,name=name)

def bias_variable(shape,name):
  #initial = tf.random_normal(shape)
  initial = tf.constant(0.0, shape=shape)
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

#_X = tf.reshape(x, shape=[-1, n_input, 1])

#  6x6 s 3
#conv11 = conv1d2('conv11', _X, _weights['wc11'], _biases['bc11'],1)
#conv12 = conv1d2('conv12', _X, _weights['wc12'], _biases['bc12'],1)
#conv13 = conv1d2('conv13', _X, _weights['wc13'], _biases['bc13'],2)
#  60x60+
#pool1 = max_pool('pool1', conv1, k=2)
# 
#norm1 = norm('norm1', conv1, lsize=4)
# Dropout
#norm1 = tf.nn.dropout(conv1, keep_prob)
#concatv1 = tf.reshape(tf.concat( [conv11, conv12,conv13] ,1), shape=[-1,1,96])
'''
# 
conv21 = conv1d2('conv21', concatv1, _weights['wc21'], _biases['bc21'],1)
conv22 = conv1d2('conv22', concatv1, _weights['wc22'], _biases['bc22'],1)
conv23 = conv1d2('conv23', concatv1, _weights['wc23'], _biases['bc23'],2)

#  to 30x30
#pool2 = max_pool('pool2', conv2, k=2)
# 
#norm2 = norm('norm2', pool2, lsize=4)
# 
#norm2 = tf.nn.dropout(conv2, keep_prob)
concatv2 = tf.reshape(tf.concat( [conv21, conv22,conv23] ,1), shape=[-1,1,192])
# 

conv31 = conv1d2('conv31', concatv2, _weights['wc31'], _biases['bc31'],1)
conv32 = conv1d2('conv32', concatv2, _weights['wc32'], _biases['bc32'],1)
conv33 = conv1d2('conv33', concatv2, _weights['wc33'], _biases['bc33'],2)
#  to 15x15
#pool3 = max_pool('pool3', conv3, k=2)
# 
#norm3 = norm('norm3', pool3, lsize=4)
# Dropout
#
concatv3 = tf.reshape(tf.concat( [conv31, conv32,conv33] ,1), shape=[-1,288])
# 
'''
xin = tf.nn.dropout(x, keep_prob)
#word  contrl weight
#WWc = weight_variable([6204],name='W_c')

#WWc = weight_variable([6204],name='W_c')

#ids = tf.constant([0,1,2,3])
#tf.nn.embedding_lookup(WWc,ids)

#W12=tf.reshape(Wab, shape=[-1,12])
#print(x_digit.shape)
Vector_Word =tf.reshape(x_digit, shape=[36,batch,256])
Word_mark = tf.placeholder(tf.float32, shape=[36,batch])
diff_w = tf.placeholder(tf.float32, shape=[36,batch])

Word_mark_w = weight_variable([1],name='W_m')
Word_mark_b = bias_variable([1],name='W_b')
Word_mark_t = tf.reshape(Word_mark,shape=[-1,1])
#print(Vector_Word.shape)

S1 = tf.gather(Vector_Word, 0)

def word_loop(idx,Wc,S1): 
  W1 = tf.gather(Vector_Word, idx)
  W2 = tf.gather(Vector_Word, idx+1)
  #print(W1.shape)
  
  W12 =  tf.reshape(tf.gather(diff_w, idx),shape=[batch,1]) #tf.reduce_sum( tf.abs(tf.subtract(W2 , W1)))/3072
  #print(W12.shape)
  #W12 = tf.reshape(W12,shape=[256,1])
  #W12 =  tf.matmul(W1,W12)
  mark_i = Word_mark_w*tf.gather(Word_mark_t, idx+1)+Word_mark_b

  S2 =  mark_i*(S1 + W1*W12) + W2
  return S2

S1=tf.gather(Vector_Word, 0)
S2 = word_loop(0,0.07,S1)
S3 = word_loop(1,0.07,S2)
S4 = word_loop(2,0.07,S3)
S5 = word_loop(3,0.07,S4)
S6 = word_loop(4,0.07,S5)
S7 = word_loop(5,0.07,S6)
S8 = word_loop(6,0.07,S7)
S9 = word_loop(7,0.07,S8)
S10= word_loop(8,0.07,S9)
S11= word_loop(9,0.07,S10)
S12= word_loop(10,0.07,S11)


SS1=tf.gather(Vector_Word, 11)
SS2 = word_loop(12,0.07,SS1)
SS3 = word_loop(13,0.07,SS2)
SS4 = word_loop(14,0.07,SS3)
SS5 = word_loop(15,0.07,SS4)
SS6 = word_loop(16,0.07,SS5)
SS7 = word_loop(17,0.07,SS6)
SS8 = word_loop(18,0.07,SS7)
SS9 = word_loop(19,0.07,SS8)
SS10= word_loop(20,0.07,SS9)
SS11= word_loop(21,0.07,SS10)
SS12= word_loop(22,0.07,SS11)


SSS1=tf.gather(Vector_Word, 23)
SSS2 = word_loop(24,0.07,SSS1)
SSS3 = word_loop(25,0.07,SSS2)
SSS4 = word_loop(26,0.07,SSS3)
SSS5 = word_loop(27,0.07,SSS4)
SSS6 = word_loop(28,0.07,SSS5)
SSS7 = word_loop(29,0.07,SSS6)
SSS8 = word_loop(30,0.07,SSS7)
SSS9 = word_loop(31,0.07,SSS8)
SSS10= word_loop(32,0.07,SSS9)
SSS11= word_loop(33,0.07,SSS10)
SSS12= word_loop(34,0.07,SSS11)

#print("S1.shape",S1.shape)
#concat = tf.concat( [S1, S2] ,0)
#print("concat.shape", concat.shape)

concat = tf.reshape( tf.concat([S1, S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,
  SS1, SS2,SS3,SS4,SS5,SS6,SS7,SS8,SS9,SS10,SS11,SS12,SSS1, 
    SSS2,SSS3,SSS4,SSS5,SSS6,SSS7,SSS8,SSS9,SSS10,SSS11,SSS12 ] , 0) , shape=[-1,n_input*36])

#print(concat.shape)

W_fc2 = weight_variable([n_input*36, 5000],name='W_fc2')
b_fc2 = bias_variable([5000],name='b_fc2')

h_fc2 = prelu(tf.nn.bias_add(tf.matmul(concat, W_fc2) , b_fc2))

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
  


W_fc3 = weight_variable([5000, n_classes],name='W_fc3')
b_fc3 = bias_variable([ n_classes],name='b_fc3')

h_fc3 = prelu(tf.nn.bias_add(tf.matmul(h_fc2_drop, W_fc3) , b_fc3))

h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

W_fc4 = weight_variable([n_classes, 5000],name='W_fc4')
b_fc4 = bias_variable([ 5000],name='b_fc4')

h_fc4 = prelu(tf.nn.bias_add(tf.matmul(h_fc3_drop, W_fc4) , b_fc4))

h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

W_fc5 = weight_variable([5000, n_classes],name='W_fc5')
b_fc5 = bias_variable([ n_classes],name='b_fc5')

h_fc5 = prelu(tf.nn.bias_add(tf.matmul(h_fc4_drop, W_fc5) , b_fc5))

h_fc5_drop = tf.nn.dropout(h_fc5, keep_prob)

W_fc6 = weight_variable([n_classes, 5000],name='W_fc6')
b_fc6 = bias_variable([ 5000],name='b_fc6')

h_fc6 = prelu(tf.nn.bias_add(tf.matmul(h_fc5_drop, W_fc6) , b_fc6))

h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob)


W_fc7 = weight_variable([ 5000,  n_classes],name='W_fc7')
b_fc7 = bias_variable([ n_classes],name='b_fc7')

y = tf.nn.bias_add(tf.matmul(h_fc6_drop, W_fc7) , b_fc7)

#y= tf.add(y1,yl)

#W_fc6 = weight_variable([512, 128],name='W_fc6')
#b_fc6 = bias_variable([128],name='b_fc6')

#h_fc6 = prelu(tf.nn.bias_add(tf.matmul(h_fc5_drop, W_fc6) , b_fc6))

#h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob)

#W_fc4 = weight_variable([512, n_classes],name='W_fc4')
#b_fc4 = bias_variable([n_classes],name='b_fc4')

#h_fc4 = tf.nn.bias_add(tf.matmul(h_fc3_drop, W_fc4) , b_fc4)

#end digital

#W_add = bias_variable([1],name='W_add')
#tf.histogram_summary('W_add', W_add)

#with tf.name_scope('Layer2'):
#y=h_fc4 #h_fc4*(1-W_add)+(out*W_add)
  #y=tf.add(tf.matmul(h_fc4,1-W_add),tf.matmul(h_fc4,W_add))


# 