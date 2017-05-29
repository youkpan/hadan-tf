import tensorflow as tf
import scipy
import scipy.misc
from numpy  import *
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid
import numpy as np

vector_w = 1
EMBEDDING_DIM = 256*2
WORDS_NUM = 7000
sentence_len = 12
n_input = vector_w 
n_classes = 4096
mshape = vector_w*sentence_len*3 # (28+24+24+120)*2+24+7+31+12+2
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
x_digit = tf.placeholder(tf.int32, shape=[None,batch,1])
y_ = tf.placeholder(tf.float32, shape=[None,1])
yl = tf.placeholder(tf.float32, shape=[None, n_input*sentence_len])
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


embeddings = tf.Variable(
        tf.random_uniform([WORDS_NUM, EMBEDDING_DIM], -1.0, 1.0))


Word_mark = tf.placeholder(tf.float32, shape=[sentence_len*3,batch])
diff_w = tf.placeholder(tf.float32, shape=[sentence_len*3,batch])

Vector_Word =tf.reshape(x_digit, shape=[36,batch])

Word_vector = weight_variable([vector_w,256],name='W_m')

Word_mark_w = weight_variable([1],name='W_m')
Word_mark_b = bias_variable([1],name='W_b')
Word_mark_t = tf.reshape(Word_mark,shape=[-1,1])

Wr = weight_variable([2*256*2,1],name='W_m')
Wz = weight_variable([2*256*2,1],name='W_m')
Wh = weight_variable([2*256*2,256*2],name='W_m')
#print(Vector_Word.shape)

S1 = tf.gather(Vector_Word, 0)

def word_loop(idx,S1): 
  #100x256
  #print(idx)
  W2 = tf.nn.embedding_lookup(embeddings, tf.gather(Vector_Word, idx+1))
  #1 256 , 256 256
  #[100,1]
  Rt = tf.reshape(sigmoid(tf.matmul(tf.concat([S1,W2],1),Wr)),shape=[batch])
  Zt = tf.reshape(sigmoid(tf.matmul(tf.concat([S1,W2],1),Wz)),shape=[batch])
  Rt = tf.expand_dims(Rt,1)
  Zt = tf.expand_dims(Zt,1)
  #print("Rt.shape",Rt.shape)
  #[100,256]
  #result = bias_variable([batch,256],name="res")
  #for i in range(batch):
  #    result[i,:] = tf.matmul(Rt[i], S1[i,:])
  #S1_ = tf.expand_dims(S1,2)
  #print("S1_.shape",S1_.shape)
  #result = tf.reshape(tf.matmul( S1_ ,Rt),shape=[batch,256])
  #print("result.shape",result.shape)
  result = Rt * S1
  h_t= tanh(tf.matmul(tf.concat([result,W2],1),Wh))
  #print("h_t.shape",h_t.shape)
  S2 =  (1-Zt)*S1 + Zt*h_t
  #print("S2.shape",S2.shape)
  return S2

S1=  word_loop(-1,np.zeros([batch,256*2]))
S2 = word_loop(0,S1)
S3 = word_loop(1,S2)
S4 = word_loop(2,S3)
S5 = word_loop(3,S4)
S6 = word_loop(4,S5)
S7 = word_loop(5,S6)
S8 = word_loop(6,S7)
S9 = word_loop(7,S8)
S10= word_loop(8,S9)
S11= word_loop(9,S10)
S12= word_loop(10,S11)


SS1= word_loop(11,np.zeros([batch,256*2]))
SS2 = word_loop(12,SS1)
SS3 = word_loop(13,SS2)
SS4 = word_loop(14,SS3)
SS5 = word_loop(15,SS4)
SS6 = word_loop(16,SS5)
SS7 = word_loop(17,SS6)
SS8 = word_loop(18,SS7)
SS9 = word_loop(19,SS8)
SS10= word_loop(20,SS9)
SS11= word_loop(21,SS10)
SS12= word_loop(22,SS11)


SSS1= word_loop(23,np.zeros([batch,256*2]))
SSS2 = word_loop(24,SSS1)
SSS3 = word_loop(25,SSS2)
SSS4 = word_loop(26,SSS3)
SSS5 = word_loop(27,SSS4)
SSS6 = word_loop(28,SSS5)
SSS7 = word_loop(29,SSS6)
SSS8 = word_loop(30,SSS7)
SSS9 = word_loop(31,SSS8)
SSS10= word_loop(32,SSS9)
SSS11= word_loop(33,SSS10)
SSS12= word_loop(34,SSS11)

#print("S1.shape",S1.shape)
#concat = tf.concat( [S1, S2] ,0)
#print("concat.shape", concat.shape)

concat = tf.reshape( tf.concat([S12,SS12,SSS12] , 1) , shape=[-1,256*2*3])

#print(concat.shape)

W_fc2 = weight_variable([256*2*3, 6000],name='W_fc2')
b_fc2 = bias_variable([6000],name='b_fc2')
#x_digit2 = tf.reshape(x_digit, shape=[-1,vector_w*sentence_len*3])

h_fc2 = prelu(tf.nn.bias_add(tf.matmul(concat, W_fc2) , b_fc2))

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
  

W_fc3 = weight_variable([6000, n_classes],name='W_fc3')
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


W_fc7 = weight_variable([ 5000,  2],name='W_fc7')
b_fc7 = bias_variable([ 2],name='b_fc7')

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