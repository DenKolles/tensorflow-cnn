import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import sys
import argparse

classes = ['oncology', 'other']

for dir in os.listdir('./data_test/'):
    print('Класс: ' + dir)
    correct = 0
    incorrect = 0
    for filename in os.listdir('./data_test/' + dir):
        image_size = 128
        num_channels = 3
        images = []
        image = cv2.imread('./data_test/%s/%s' % (dir, filename))
        image = cv2.resize(image, (image_size, image_size),
                           0, 0, cv2.INTER_LINEAR)
        images.append(image)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0 / 255.0)

        x_batch = images.reshape(1, image_size, image_size, num_channels)

        sess = tf.Session()
        saver = tf.train.import_meta_graph('model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        graph = tf.get_default_graph()

        y_pred = graph.get_tensor_by_name("y_pred:0")

        x = graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y_true:0")
        y_test_images = np.zeros((1, 2))

        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result = sess.run(y_pred, feed_dict=feed_dict_testing)
        result_list = result[0].tolist()

        if classes[result_list.index(max(result_list))] == dir:
            correct += 1
        else:
            incorrect += 1

    print('Точность: %s%%' % (correct / (correct + incorrect) * 100))

# g = tf.Graph()
#
# with g.as_default() as g:
#     tf.train.import_meta_graph('model.meta')
#
# with tf.Session(graph=g) as sess:
#     file_writer = tf.summary.FileWriter(logdir='logs/my-model', graph=g)
