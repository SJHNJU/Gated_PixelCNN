import tensorflow as tf
import numpy as np 


class gated_pixel_cnn():
    def __init__(self, X, p):
        #X [batch_size, height, width, channel]
        WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()
        self.X = X
        self.horizontal_stack, self.vertical_stack = self.X, self.X
        self.saved_horizontal_stack = None
        #self.v2h = None

        #kernel size , args, previous channel, current layer   
        with tf.variable_scope('layer0'):
            self.gated_layer(7, p, 1, 0)
        
        for i in range(1,p.depth+1):
            with tf.variable_scope('layer'+str(i)):
                self.gated_layer(3, p, p.fmaps, i)
        
        with tf.variable_scope("fc1"):
            weight = tf.get_variable('fc1_w', [1,1,p.fmaps,p.fmaps], tf.float32, WEIGHT_INITIALIZER)
            bias = tf.get_variable('fc1_b', [p.fmaps], tf.float32, tf.zeros_initializer)
            self.output = tf.nn.relu(tf.add(tf.nn.conv2d(self.horizontal_stack, weight, strides=[1,1,1,1], padding='SAME'), bias))

        with tf.variable_scope("fc2"):
            weight2 = tf.get_variable('fc2_w', [1,1,p.fmaps,1], tf.float32, WEIGHT_INITIALIZER)
            #bias2 = tf.get_variable('fc2_b', [1], tf.float32, tf.zeros_initializer)
            self.output = tf.nn.relu((tf.nn.conv2d(self.output, weight2, strides=[1,1,1,1], padding='SAME')))


        #great build the network! compute the loss and the predict image now !!!!cheers!
        #loss and predict image
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.X))

        self.pred = tf.nn.sigmoid(self.output)






#================================middle layer with gated conv==================================================#   
                 #kernel size , args, previous channel, current layer   
    def gated_layer(self, wsize, p, channel, layer):
        WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()
        #vertical stack
        #two conv in vertical stack
        v_w_f = tf.get_variable('v_w_f', [wsize, wsize, channel, p.fmaps], tf.float32, WEIGHT_INITIALIZER)         
        v_w_g = tf.get_variable('v_w_g', [wsize, wsize, channel, p.fmaps], tf.float32, WEIGHT_INITIALIZER)
        #get mask the layer0 has different mask
        maskv = np.ones([wsize,wsize,channel,p.fmaps], dtype=np.float32)
        if layer == 0:
            maskv[wsize//2:,:,:,:] = 0.0
        else:
            maskv[wsize//2+1,:,:,:] = 0.0

        v_w_f = tf.multiply(v_w_f, maskv)
        v_w_g = tf.multiply(v_w_g, maskv)

        #two convlution here
        self.v_f = tf.nn.conv2d(self.vertical_stack, v_w_f, strides=[1,1,1,1], padding='SAME')
        self.v_g = tf.nn.conv2d(self.vertical_stack, v_w_g, strides=[1,1,1,1], padding='SAME')

        #cheers! get vertical stack need to pepare 1*1conv for layers other than layer 0
        self.vertical_stack = tf.multiply(tf.tanh(self.v_f), tf.sigmoid(self.v_g))
        if layer != 0:
            v_w11 = tf.get_variable('v_w11', [1, 1, p.fmaps, p.fmaps], tf.float32, WEIGHT_INITIALIZER)
            v_b = tf.get_variable('v_b', [p.fmaps], tf.float32, tf.zeros_initializer)
            self.v2h = tf.add(tf.nn.conv2d(self.vertical_stack , v_w11, [1,1,1,1], padding='SAME'), v_b)
            self.v2h = tf.nn.relu(self.v2h)
        #bravo! get the vertical_stack here and the v2h waited to be add in horizontal stack

#==========================================================================================================================#        

        #horizontal stack
        h_w_f = tf.get_variable('h_w_f', [wsize, wsize, channel, p.fmaps], tf.float32, WEIGHT_INITIALIZER)
        h_w_g = tf.get_variable('h_w_g', [wsize, wsize, channel, p.fmaps], tf.float32, WEIGHT_INITIALIZER)

        #get the mask for horizontal stack
        maskh = np.ones([wsize, wsize, channel, p.fmaps], dtype=np.float32)
        if layer == 0:
            maskh[(wsize//2)+1:, :, :, :] = 0.0
            maskh[wsize//2, wsize//2:, :, :] = 0.0
        else:
            maskh[(wsize//2)+1:, :, :, :] = 0.0
            maskh[wsize//2, (wsize//2)+1:, :, :] = 0.0
        
        
        h_w_f = tf.multiply(h_w_f, maskh)
        h_w_g = tf.multiply(h_w_g, maskh)
        #here get the two masked weight ,need to make two convolution here
        self.h_f = tf.nn.conv2d(self.horizontal_stack, h_w_f, strides=[1,1,1,1], padding='SAME')
        self.h_g = tf.nn.conv2d(self.horizontal_stack, h_w_g, strides=[1,1,1,1], padding='SAME')

        #two conv result need to add the infomation from vertical stack for layer other than 0, then pass the gate
        if layer != 0:
            self.h_f += self.v2h
            self.h_g += self.v2h

        self.horizontal_stack = tf.multiply(tf.tanh(self.h_f), tf.sigmoid(self.h_g))
        #now my horizontal stack pass the gate, prepare the 1*1 conv
        h_w11 = tf.get_variable('h_w11', [1, 1, p.fmaps, p.fmaps], tf.float32, WEIGHT_INITIALIZER)
        h_b = tf.get_variable('h_b', [p.fmaps], tf.float32, tf.zeros_initializer)

        self.horizontal_stack = tf.add(tf.nn.conv2d(self.horizontal_stack, h_w11, [1,1,1,1], padding='SAME'), h_b)
        self.horizontal_stack = tf.nn.relu(self.horizontal_stack)
        #great! get the horizontal stack, need to save the current result for the next layer with residual block
        self.saved_horizontal_stack = self.horizontal_stack

        #here get the horizontal stack, need residual for layer other than 0
        if layer != 0:
            self.horizontal_stack += self.saved_horizontal_stack
            self.saved_horizontal_stack = self.horizontal_stack
        #bravo!! get the horizontal_stack and update the saved_horizontal_stack 
