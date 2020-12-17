import tensorflow as tf
import numpy as np
import cv2

from app.imagePreprocess import preprocess

classes = ['oncology', 'other']
image_size = 128
num_channels = 3


def predict_image(file_str):
    image_size = 128
    num_channels = 3
    images = []
    image = cv2.imread(file_str)
    image = cv2.resize(image, (image_size, image_size),
                       0, 0, cv2.INTER_LINEAR)
    image = preprocess(image)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)

    x_batch = images.reshape(1, image_size, image_size, num_channels)

    sess = tf.Session()
    saver = tf.train.import_meta_graph('app/model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./app/'))

    graph = tf.get_default_graph()

    y_pred = graph.get_tensor_by_name("y_pred:0")

    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 2))

    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    result_list = result[0].tolist()

    class_id = result_list.index(max(result_list))
    return {
        'class_id': class_id,
        'class_name': classes[class_id],
        'probability': "{:.0f}%".format(max(result_list) * 100),
    }