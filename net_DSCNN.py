import tensorflow as tf
import tflearn


def network(imga,scope='netflow', reuse=False):

    with tf.variable_scope(scope, reuse=reuse):

        c1_w = tf.get_variable("c1_w", shape=[9, 9, 1, 128],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c1_b = tf.get_variable("c1_b", shape=[128],
                               initializer=tf.constant_initializer(0.0))
        c11_w = tf.get_variable("c11_w", shape=[9, 9, 1, 128],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c11_b = tf.get_variable("c11_b", shape=[128],
                               initializer=tf.constant_initializer(0.0))

        c2_w = tf.get_variable("c2_w", shape=[7, 7, 128, 64],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c2_b = tf.get_variable("c2_b", shape=[64],
                               initializer=tf.constant_initializer(0.0))
        c22_w = tf.get_variable("c22_w", shape=[7, 7, 256, 64],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c22_b = tf.get_variable("c22_b", shape=[64],
                                initializer=tf.constant_initializer(0.0))

        c3_w = tf.get_variable("c3_w", shape=[3, 3, 64, 64],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c3_b = tf.get_variable("c3_b", shape=[64],
                               initializer=tf.constant_initializer(0.0))
        c33_w = tf.get_variable("c33_w", shape=[3, 3, 128, 64],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c33_b = tf.get_variable("c33_b", shape=[64],
                                initializer=tf.constant_initializer(0.0))

        c4_w = tf.get_variable("c4_w", shape=[1, 1, 64, 32],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c4_b = tf.get_variable("c4_b", shape=[32],
                               initializer=tf.constant_initializer(0.0))
        c44_w = tf.get_variable("c44_w", shape=[1, 1, 128, 32],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c44_b = tf.get_variable("c44_b", shape=[32],
                                initializer=tf.constant_initializer(0.0))

        c55_w = tf.get_variable("c55_w", shape=[5, 5, 64, 1],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c55_b = tf.get_variable("c55_b", shape=[1],
                                initializer=tf.constant_initializer(0.0))


        c1 = tf.nn.conv2d(imga, c1_w, strides=[1, 1, 1, 1], padding='SAME')
        c1 = tf.nn.bias_add(c1, c1_b)
        c1 = tflearn.activations.prelu(c1)

        c11 = tf.nn.conv2d(imga, c11_w, strides=[1, 1, 1, 1], padding='SAME')
        c11 = tf.nn.bias_add(c11, c11_b)
        c11 = tflearn.activations.prelu(c11)

        concat1 = tf.concat([c1, c11],3, name='concat1')

        c2 = tf.nn.conv2d(c1, c2_w, strides=[1, 1, 1, 1], padding='SAME')
        c2 = tf.nn.bias_add(c2, c2_b)
        c2 = tflearn.activations.prelu(c2)

        c22 = tf.nn.conv2d(concat1, c22_w, strides=[1, 1, 1, 1], padding='SAME')
        c22 = tf.nn.bias_add(c22, c22_b)
        c22 = tflearn.activations.prelu(c22)

        concat2 = tf.concat([c2, c22],3, name='concat2')

        c3 = tf.nn.conv2d(c2, c3_w, strides=[1, 1, 1, 1], padding='SAME')
        c3 = tf.nn.bias_add(c3, c3_b)
        c3 = tflearn.activations.prelu(c3)

        c33 = tf.nn.conv2d(concat2, c33_w, strides=[1, 1, 1, 1], padding='SAME')
        c33 = tf.nn.bias_add(c33, c33_b)
        c33 = tflearn.activations.prelu(c33)

        concat3 = tf.concat([c3, c33], 3, name='concat3')

        c4 = tf.nn.conv2d(c3, c4_w, strides=[1, 1, 1, 1], padding='SAME')
        c4 = tf.nn.bias_add(c4, c4_b)
        c4 = tflearn.activations.prelu(c4)

        c44 = tf.nn.conv2d(concat3, c44_w, strides=[1, 1, 1, 1], padding='SAME')
        c44 = tf.nn.bias_add(c44, c44_b)
        c44 = tflearn.activations.prelu(c44)

        concat4 = tf.concat([c4, c44],3, name='concat4')

        c55 = tf.nn.conv2d(concat4, c55_w, strides=[1, 1, 1, 1], padding='SAME')
        c55 = tf.nn.bias_add(c55, c55_b)

        output = c55 + imga

        tf.summary.image('input', imga)
        tf.summary.image('output', output)

    return output
