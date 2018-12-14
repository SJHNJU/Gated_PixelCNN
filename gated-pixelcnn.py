import os, argparse
import tensorflow as tf 
import numpy as np 

import input_data

from model import *
from utility import *


'''
author:SJH
date:2018/12/10
description:generate mnist image using gated pixel cnn
'''

def train(p, mnist):
    X = tf.placeholder(tf.float32,[None,28,28,1],'input_data')

    net = gated_pixel_cnn(X,p)

    trainer = tf.train.AdamOptimizer(p.learning_rate)
    
    gradients = trainer.compute_gradients(net.loss)
 
    clipped_gradients = [(tf.clip_by_value(_[0],-1,1), _[1]) for _ in gradients]
    optimizer = trainer.apply_gradients(clipped_gradients)
    

    saver = tf.train.Saver(tf.trainable_variables())
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if p.model_path is not None:
            saver.restore(sess,p.model_path)
            print('MODEL RESTORED')
            get_sample(sess=sess, X=X, data=mnist, epoch=0, pred=net.pred)    


        if p.MAX_EPOCH > 0:
            print('Start training...')
        
        for i in range(0, p.MAX_EPOCH):
            for j in range(0, p.BATCH_NUM):
                I,_ = mnist.train.next_batch(p.BATCH_SIZE)
                I = binarize(I.reshape([p.BATCH_SIZE,28,28,1]))
                data_dict = {X: I}

                _, l , img= sess.run([optimizer, net.loss, net.pred], feed_dict = data_dict)
                #print(c_grad)

                print('[epoch %d batch %d] loss = %.4f' %(i, j, l)) 
                
                

            os.makedirs('./saved_models/epoch'+str(i))
            saver.save(sess, './saved_models/epoch%d/model.ckpt'%i)   
            get_sample(sess=sess, X=X, data=mnist, epoch=i, pred=net.pred)    
            
        

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None) 
    #parameter used in training 
    parser.add_argument('--MAX_EPOCH',default=100)
    parser.add_argument('--BATCH_SIZE',default=50)
    parser.add_argument('--learning_rate',default=0.0008)
    #parameter used in network
    parser.add_argument('--fmaps',default=32)
    parser.add_argument('--depth',default=12)
    args = parser.parse_args()

    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    args.BATCH_NUM = mnist.train.num_examples // args.BATCH_SIZE

    train(args, mnist)
    

