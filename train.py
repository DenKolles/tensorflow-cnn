import dataset
import tensorflow as tf

from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

tf.logging.set_verbosity(tf.logging.INFO)

batch_size = 8

classes = ['oncology', 'other']
num_classes = len(classes)

validation_size = 0.2
img_size = 128
num_channels = 3
train_path = './data/'

data = dataset.read_train_sets(
    train_path, img_size, classes, validation_size=validation_size)

print('Чтение данных завершено')
print('Количество данных в тренировочной выборке:{}'.format(len(data.train.labels)))
print('Количество данных в валидационной выборке:{}'.format(len(data.valid.labels)))

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[
                   None, img_size, img_size, num_channels], name='x')

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

filter_size_conv1 = 3
filtres_numbers_conv1 = 64

filter_size_conv2 = 3
filtres_numbers_conv2 = 64

filter_size_conv3 = 3
filtres_numbers_conv3 = 32

filter_size_conv4 = 3
filtres_numbers_conv4 = 32


fc_layer_size = 128


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_parametres(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,
                               input_channels_number,
                               conv_filter_size,
                               filtres_numbers,
                               is_pooling,
                               ):
    # Определяем параметры филтра
    weights = create_weights(
        shape=[conv_filter_size, conv_filter_size, input_channels_number, filtres_numbers])
    # Определяем параметров сдвига
    parametres = create_parametres(filtres_numbers)
    # Создаем сверточный слой
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += parametres
    # Создаем слой пуллинга
    if is_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    # Активационная функция
    layer = tf.nn.relu(layer)
    return layer


def create_flatten_layer(layer):
    # Получение параметров входящего слоя
    layer_shape = layer.get_shape()

    # Получение длины плоского слоя (длина * ширина * кол-во фильтров)
    num_features = layer_shape[1:4].num_elements()

    # Преобразование слоя в плоский
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    # Определение весов нейронов и сдвигов
    weights = create_weights(shape=[num_inputs, num_outputs])
    parametres = create_parametres(num_outputs)

    # Подсчет поступающего сигнала на нейрон (w * x + b)
    layer = tf.matmul(input, weights) + parametres
    if use_relu:
        layer = tf.nn.relu(layer)  # Применение RELU-функции активации

    return layer


# Сверточные слои
layer_conv1 = create_convolutional_layer(input=x,
                                         input_channels_number=num_channels,
                                         conv_filter_size=filter_size_conv1,
                                         filtres_numbers=filtres_numbers_conv1,
                                         is_pooling=True)

layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                         input_channels_number=filtres_numbers_conv1,
                                         conv_filter_size=filter_size_conv2,
                                         filtres_numbers=filtres_numbers_conv2,
                                         is_pooling=True)

layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                         input_channels_number=filtres_numbers_conv2,
                                         conv_filter_size=filter_size_conv3,
                                         filtres_numbers=filtres_numbers_conv3,
                                         is_pooling=True)

layer_conv4 = create_convolutional_layer(input=layer_conv3,
                                         input_channels_number=filtres_numbers_conv3,
                                         conv_filter_size=filter_size_conv4,
                                         filtres_numbers=filtres_numbers_conv4,
                                         is_pooling=True)

# Плоский слой
layer_flat = create_flatten_layer(layer_conv4)

# Полносвязные слои
layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[
                                1:4].num_elements(),
                            num_outputs=fc_layer_size,
                            use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                            num_inputs=fc_layer_size,
                            num_outputs=fc_layer_size,
                            use_relu=True)

layer_fc3 = create_fc_layer(input=layer_fc2,
                            num_inputs=fc_layer_size,
                            num_outputs=num_classes,
                            use_relu=False)

y_pred = tf.nn.softmax(layer_fc3, name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())


def show_teminal(epoch, feed_dict_train, feed_dict_validate, validation_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Эпоха {0} --- Точность на тренировочной выборке: {1:>6.1%}, Точность на валидационной выборке: {2:>6.1%},  Потери валидации: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, validation_loss))


iterations_number_total = 0

saver = tf.train.Saver()


def train(iteration_number):
    # Количество итераций обучения
    global iterations_number_total

    for i in range(iterations_number_total,
                   iterations_number_total + iteration_number):
        # Получение данных из выборок (обучающей и валидационной) для каждой итерации
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(
            batch_size)

        # Подготовка наблоров данных (обучающего и валидационного) для передачи в сессию
        feed_dict_tr = {x: x_batch, y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

        # Запуск сессии
        session.run(optimizer, feed_dict=feed_dict_tr)


        if i % int(data.train.num_examples / batch_size) == 0:
            # Вычисление валидационных потерь
            validation_loss = session.run(cost, feed_dict=feed_dict_val)

            # Вычисление эпохи обучения
            epoch = int(i / int(data.train.num_examples / batch_size))

            show_teminal(epoch, feed_dict_tr, feed_dict_val, validation_loss)

            # Сохранение состояние модели с оптимизированными коэфициентами
            saver.save(session, './model')

    # Счетчик итераций
    iterations_number_total += iteration_number


train(iteration_number=30000)
