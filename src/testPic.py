# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model
from processData import get_files
from PIL import Image   
import os
import cv2
import argparse

#-------------------------------------------------------------------
# get dirctory
def get_dir():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='name of test image')
    args = vars(ap.parse_args())
    #train_dir = 'D:/SECRET/trainTF/data/test/' + str(args['image'])
    train_dir = str(args['image'])
    return train_dir

#-------------------------------------------------------------------
#获取一张图片
def get_one_image(train):
    #输入参数：train,训练图片的路径
    #返回参数：image，从训练图片中随机抽取一张图片
    
    img = Image.open(train)
    
    imag = img.resize([64, 64])  #由于图片在预处理阶段以及resize，因此该命令可略
    image = np.array(imag)
    return image

#--------------------------------------------------------------------
#测试图片
def evaluate_one_image(image_array):
    with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 5

       image = tf.cast(image_array, tf.float32)
       image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, 64, 64, 3])

       logit = model.inference(image, BATCH_SIZE, N_CLASSES)

       logit = tf.nn.softmax(logit)

       x = tf.placeholder(tf.float32, shape=[64, 64, 3])

       # you need to change the directories to yours.
       logs_train_dir = 'D:/SECRET/trainTF/data/train'

       saver = tf.train.Saver()

       with tf.Session() as sess:

           print("\n[INFO] Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('[INFO] Loading model successfully, global_step is %s' % global_step)
               print('[INFO] Classifying...please wait...\n')
           else:
               print('No checkpoint file found')

           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)
           if prediction[:, max_index] <= 0.6:
               print('[INFO] Classification fail!')
               print('[RESULT] Sorry, I have no idea what it is.')
           else:
                print('[INFO] Classification success!')
                if max_index==3:
                    print('[RESULT] This is a husky with possibility %.6f' % prediction[:, 3])
                    return 'husky'
                elif max_index==4:
                    print('[RESULT] This is a teddy with possibility %.6f' % prediction[:, 4])
                    return 'teddy'
                elif max_index==0:
                    print('[RESULT] This is a border with possibility %.6f' % prediction[:, 0])
                    return 'border'
                elif max_index == 1:
                    print('[RESULT] This is a bulldog with possibility %.6f' % prediction[:, 1])
                    return 'bulldog'
                elif max_index == 2:
                    print('[RESULT] This is a golden with possibility %.6f' % prediction[:, 2])
                    return 'golden'

#------------------------------------------------------------------------
               
if __name__ == '__main__':
    
    train_dir = get_dir()
    
    img2 = cv2.imread(train_dir)
    img = get_one_image(train_dir) 
    dog = evaluate_one_image(img)
    cv2.putText(img2, 'I think this is a '+dog+'.', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
    cv2.imshow('img', img2)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()