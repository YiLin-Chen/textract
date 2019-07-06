'''
classifier.py module is used for defining the model of text/non-text classifier in layout analysis
currently the model is CNN(VGG16) with pretrained weight
'''

import tensorflow as tf
import numpy as np 
import cv2
from enum import Enum

class ClassifierType(Enum):
    CNN_VGG16 = 0


class BlockClassifier(object):
    def __init__(self, model_type = ClassifierType.CNN_VGG16):
        if model_type == ClassifierType.CNN_VGG16:
            self._model = VGG16('./textract/model/vgg/model.ckpt-19999')

        # elif
        # other model in the future...

        else:
            print('model type is not correct')

    def classify(self, block):
        return self._model.predict(block)

    def get_type(self):
        pass


class VGG16(object):
    def __init__(self, model_path):
        self._categories = {0:'Text', 1:'Image'}
        
        # pretrained model
        self._data_dict = np.load('./textract/model/vgg16.npy', encoding='latin1').item()

        self._x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
        self._keep_prob = tf.placeholder(tf.float32)
        output = self._build(self._keep_prob, 2)
        self._score = tf.nn.softmax(output)
        self._f_cls = tf.argmax(self._score, 1)

        self._sess = tf.InteractiveSession()
        self._sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        saver.restore(self._sess, model_path)

    def predict(self, src):
        src = cv2.resize(src, (224 , 224))
        src = np.expand_dims(src, axis=0)

        pred, _score = self._sess.run([self._f_cls, self._score], feed_dict={self._x:src, self._keep_prob:1.0})
        prob = round(np.max(_score), 4)

        return self._categories[int(pred)], prob

    def _build(self, _dropout, n_cls):


        conv1_1 = self._conv(self._x, 64, 'conv1_1', fineturn=True)
        conv1_2 = self._conv(conv1_1, 64, 'conv1_2', fineturn=True)
        pool1   = self._maxpool(conv1_2, 'pool1')

        conv2_1 = self._conv(pool1, 128, 'conv2_1', fineturn=True)
        conv2_2 = self._conv(conv2_1, 128, 'conv2_2', fineturn=True)
        pool2   = self._maxpool(conv2_2, 'pool2')

        conv3_1 = self._conv(pool2, 256, 'conv3_1', fineturn=True)
        conv3_2 = self._conv(conv3_1, 256, 'conv3_2', fineturn=True)
        conv3_3 = self._conv(conv3_2, 256, 'conv3_3', fineturn=True)
        pool3   = self._maxpool(conv3_3, 'pool3')

        conv4_1 = self._conv(pool3, 512, 'conv4_1', fineturn=True)
        conv4_2 = self._conv(conv4_1, 512, 'conv4_2', fineturn=True)
        conv4_3 = self._conv(conv4_2, 512, 'conv4_3', fineturn=True)
        pool4   = self._maxpool(conv4_3, 'pool4')


        conv5_1 = self._conv(pool4, 512, 'conv5_1', fineturn=True)
        conv5_2 = self._conv(conv5_1, 512, 'conv5_2', fineturn=True)
        conv5_3 = self._conv(conv5_2, 512, 'conv5_3', fineturn=True)
        pool5   = self._maxpool(conv5_3, 'pool5')

        flatten  = tf.reshape(pool5, [-1, 7*7*512])
        fc6      = self._fc(flatten, 4096, 'fc6', fineturn=False, xavier=False)
        dropout1 = tf.nn.dropout(fc6, _dropout)

        fc7      = self._fc(dropout1, 4096, 'fc7', fineturn=False, xavier=False)
        dropout2 = tf.nn.dropout(fc7, _dropout)

        fc8      = self._fc8(dropout2, n_cls, 'fc8', xavier=True)

        return fc8
    
    # print info for each layer
    def _print_layer(self, t):
        print(t.op.name, ' ', t.get_shape().as_list(), '\n')

    def _conv(self, x, d_out, name, fineturn=False, xavier=False):
        d_in = x.get_shape()[-1].value
        with tf.name_scope(name) as scope:
            # Fine-tuning 
            if fineturn:
                kernel = tf.constant(self._data_dict[name][0], name="weights")
                bias = tf.constant(self._data_dict[name][1], name="bias")
                print("fineturn")

            elif not xavier:
                kernel = tf.Variable(tf.truncated_normal([3, 3, d_in, d_out], stddev=0.1), name='weights')
                bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[d_out]), trainable=True, name='bias')
                print("truncated_normal")

            else:
                kernel = tf.get_variable(scope+'weights', shape=[3, 3, d_in, d_out], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
                bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[d_out]), trainable=True, name='bias')
                print("xavier")

            
            conv = tf.nn.conv2d(x, kernel,[1, 1, 1, 1], padding='SAME')
            activation = tf.nn.relu(conv + bias, name=scope)

            self._print_layer(activation)

            return activation

    def _maxpool(self, x, name):
        activation = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name=name) 
        self._print_layer(activation)

        return activation

    def _fc(self, x, n_out, name, fineturn=False, xavier=False):
        n_in = x.get_shape()[-1].value

        with tf.name_scope(name) as scope:
            if fineturn:
                weight = tf.constant(self._data_dict[name][0], name="weights")
                bias = tf.constant(self._data_dict[name][1], name="bias")
                print("fineturn")

            elif not xavier:
                weight = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.01), name='weights')
                bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]), trainable=True, name='bias')
                print("truncated_normal")

            else:
                weight = tf.get_variable(scope+'weights', shape=[n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
                bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]), trainable=True, name='bias')
                print("xavier")

            activation = tf.nn.relu_layer(x, weight, bias, name=name)
            self._print_layer(activation)

            return activation

    def _fc8(self, x, n_out, name, fineturn=False, xavier=False):
        n_in = x.get_shape()[-1].value

        with tf.name_scope(name) as scope:
            if fineturn:
                weight = tf.constant(self._data_dict[name][0], name="weights")
                bias = tf.constant(self._data_dict[name][1], name="bias")
                print("fineturn")

            elif not xavier:
                weight = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.01), name='weights')
                bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]), trainable=True, name='bias')
                print("truncated_normal")

            else:
                weight = tf.get_variable(scope+'weights', shape=[n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
                bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]), trainable=True, name='bias')
                print("xavier")
            
            activation = tf.nn.bias_add(tf.matmul(x,weight),bias)
            self._print_layer(activation)

            return activation

