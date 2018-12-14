import os, random
import numpy as np 
import tensorflow as tf 
import scipy.misc

def binarize(IMG):
    return (IMG > 0.5).astype(np.float32)

#input size [batch_size,H,W,C]
def save_img(img,img_type,epoch):
    I = img.reshape((10,10,28,28))
    I = I.transpose(1, 2, 0, 3)
    I = I.reshape((280,280))
    scipy.misc.toimage(I, cmin=0.0, cmax=1.0).save('./samples/EPOCH'+str(epoch)+'/'+img_type+'.png')

#get 100 imgs from test/train batch
#create three images under dir './sample/epoch%d'
#            sess, placeholder, mnist, epoch, model.predict
def get_sample(sess, X, data, epoch, pred):
    print('[EPOCH %d] Get samples'%epoch)
    img,_ = data.test.next_batch(100)
    img = binarize(img.reshape((100,28,28,1)))
    
    os.makedirs('./samples/EPOCH%d'%epoch)
    #save ground truth
    save_img(img,'Ground_Truth',epoch)
    #save masked img
    occluded = np.copy(img)
    occluded[:, 18:, :, :] = 0.5
    save_img(occluded,'Masked_Img',epoch)
    #get predict img
    for i in range(14, 28):
        for j in range(0, 28):
            data_dict = {X : occluded}
            predict_img = sess.run(pred, feed_dict=data_dict)
            predict_img = binarize(predict_img)
            occluded[:, i, j, :] = predict_img[:, i, j, :]

    save_img(occluded,'Predict',epoch)