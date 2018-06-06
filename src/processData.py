# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:09:58 2018

@author: Jasper.Hsu
"""

import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#============================================================================
#-----------------生成图片路径和标签的List------------------------------------

train_dir = 'D:/SECRET/trainTF/data/input/'




#step1：获取train_dir下所有的图片路径名，存放到对应的列表中，同时贴上标签，存放到label列表中。
def get_files(file_dir):
    husky = []
    label_husky = []
    teddy = []
    label_teddy = []
    border = []
    label_border = []
    bulldog = []
    label_bulldog = []
    golden = []
    label_golden = []
    for file in os.listdir(file_dir):
        name = file.split('.')
        if name[0] == '4':
            teddy.append(file_dir+file)
            label_teddy.append(4)
        elif name[0] == '0':
            border.append(file_dir+file)
            label_border.append(0)
        elif name[0] == '3':
            husky.append(file_dir+file)
            label_husky.append(3)
        elif name[0] == '1':
            bulldog.append(file_dir + file)
            label_bulldog.append(1)
        elif name[0] == '2':
            golden.append(file_dir+file)
            label_golden.append(2)

    print('There are {} teddy, {} husky, {} border, {} bulldog and {} golden' .format(len(teddy),len(husky),len(border),len(bulldog),len(golden)))
#step2：对生成的图片路径和标签List做打乱处理把全部合起来组成一个list（img和lab）
    image_list = np.hstack((border, bulldog, golden, husky, teddy))
    label_list = np.hstack((label_border, label_bulldog, label_golden, label_husky, label_teddy))

    #利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    #从打乱的temp中再取出list（img和lab）
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list
    
#---------------------------------------------------------------------------
#--------------------生成Batch----------------------------------------------

#step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue，因为img和lab
#是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像
#   image_W, image_H, ：设置好固定的图像高度和宽度
#   设置batch_size：每个batch要放多少张图片
#   capacity：一个队列最大多少
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    #转换类型
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0]) #read img from a queue  
    
#step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    image = tf.image.decode_jpeg(image_contents, channels=3) 
    
#step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    #如果想看到正常图片，请注释掉98和111行(image_batch = tf.cast...)
    image = tf.image.per_image_standardization(image)

#step4：生成batch
#image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32 
#label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 32, 
                                                capacity = capacity)
    #重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch            

#========================================================================</span>
if __name__ == '__main__':  
    batch_size = 20
    capacity = 256
    image_W = 64
    image_H = 64
    
    img_list, lab_list = get_files(train_dir)
    image_batch, label_batch = get_batch(img_list, lab_list, image_W, image_H, batch_size, capacity)

    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop() and i<1:

                img, label = sess.run([image_batch, label_batch])

               # just test one batch
                for j in np.arange(batch_size):
                    print('label: %d' %label[j])
                    plt.imshow(img[j,:,:,:])
                    plt.show()
                i+=1

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)

    
    
    
    