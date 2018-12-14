import tensorflow as tf
import numpy as np 


class gated_pixel_cnn():
    def __init__(self, X, p):
        #X [batch_size, height, width, channel]
        WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()
        self.X = X
        self.output = None
        self.horizontal_stack, self.horizontal_stack_1 = X, X
        self.vertical_stack = X 
        self.tmp = None

        with tf.variable_scope('layer0'):
            self.gated_layer(7, p, 1, 0)
        
        for i in range(1,p.depth+1):
            with tf.variable_scope('layer'+str(i)):
                self.gated_layer(3, p, p.fmaps, i)
        
        with tf.variable_scope("fc1"):
            weight = tf.get_variable('fc1_w', [1,1,p.fmaps,p.fmaps], tf.float32, WEIGHT_INITIALIZER)
            self.output = tf.nn.relu(tf.nn.conv2d(self.horizontal_stack, weight, strides=[1,1,1,1], padding='SAME'))
        with tf.variable_scope("fc2"):
            weight = tf.get_variable('fc2_w', [1,1,p.fmaps,1], tf.float32, WEIGHT_INITIALIZER)
            self.output = tf.nn.relu(tf.nn.conv2d(self.output, weight, strides=[1,1,1,1], padding='SAME'))

        #loss and predict image
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.X))
        self.pred = tf.nn.sigmoid(self.output)


    #middle layer with gated conv        
                 #kernel size , args, previous channel, current layer   
    def gated_layer(self, wsize, p, channel, layer):
        WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()
        #vertical stack
        #n*n conv in vertical stack, the mask here need to be discussed
        weightnn = tf.get_variable('v_w_v', [wsize,wsize,channel,2*p.fmaps], tf.float32, WEIGHT_INITIALIZER)         
        mask = np.ones([wsize,wsize,channel,2*p.fmaps], dtype=np.float32)
        if layer == 0:
            mask[wsize//2:,:,:,:] = 0.0
        else:
            mask[wsize//2+1,:,:,:] = 0.0

        weightnn = tf.multiply(weightnn, mask)
        self.vertical_stack = tf.nn.conv2d(self.vertical_stack, weightnn, strides=[1,1,1,1], padding='SAME')
        #prepare the 1*1 conv in vertial stack to add to h_stack
        weightv11 = tf.get_variable('v_w_h', [1,1,2*p.fmaps,2*p.fmaps], tf.float32, WEIGHT_INITIALIZER)
        biasv11 = tf.get_variable('v_w_hb', [2*p.fmaps], tf.float32, tf.zeros_initializer)
        self.tmp = tf.nn.relu(tf.add(tf.nn.conv2d(self.vertical_stack, weightv11, strides=[1,1,1,1], padding='SAME'),biasv11))
        #split the feature maps for gate, and mutiply
        vtanhpart = tf.tanh(self.vertical_stack[:,:,:,0:p.fmaps], name='vtanh')
        vsigmoidpart = tf.sigmoid(self.vertical_stack[:,:,:,p.fmaps:2*p.fmaps], name='vsig')
        self.vertical_stack = tf.multiply(vtanhpart, vsigmoidpart)

        #horizontal stack
        #1*n conv in horizontal stack, and plus the 1*1 conv from vertical stack
        weight1n = tf.get_variable('h_1', [wsize,wsize,channel,2*p.fmaps], tf.float32, WEIGHT_INITIALIZER)
        mask = np.ones([wsize,wsize,channel,2*p.fmaps], dtype=np.float32)
        if layer == 0:
            mask[wsize//2+1:,:,:,:] = 0.0
            mask[wsize//2, wsize//2:,:,:] = 0.0
        else:
            mask[wsize//2+1:,:,:,:] = 0.0
            mask[wsize//2,wsize//2+1,:,:] = 0.0

        weight1n = tf.multiply(mask, weight1n)
        self.horizontal_stack = tf.nn.conv2d(self.horizontal_stack, weight1n, strides=[1,1,1,1], padding='SAME')
        self.horizontal_stack += self.tmp
        #split the feature maps for gate and mutiply
        htanhpart = tf.tanh(self.horizontal_stack[:,:,:,0:p.fmaps], name='htanh')
        hsigmoidpart = tf.sigmoid(self.horizontal_stack[:,:,:,p.fmaps:2*p.fmaps], name='hsig')
        self.horizontal_stack = tf.multiply(htanhpart, hsigmoidpart)
        #hstack 1*1 conv and the residual part
        weighth11 = tf.get_variable('h_2', [1,1,p.fmaps,p.fmaps], tf.float32, WEIGHT_INITIALIZER)
        biash11 = tf.get_variable('h_2b', [p.fmaps], tf.float32, tf.zeros_initializer)
        self.horizontal_stack = tf.nn.relu(tf.add(tf.nn.conv2d(self.horizontal_stack, weighth11, strides=[1,1,1,1], padding='SAME'), biash11))
        
        if layer == 0:
            self.horizontal_stack_1 = self.horizontal_stack
        else:
            self.horizontal_stack += self.horizontal_stack_1
            self.horizontal_stack_1 = self.horizontal_stack
