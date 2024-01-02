import tensorflow as tf

from tensorflow import keras

@keras.saving.register_keras_serializable(name='custom_relu')
def relu(x):
    x = tf.maximum(0.0, x)
    return x

@keras.saving.register_keras_serializable(name='custom_tanh')
def tanh(x):
    # x = e^x - e^-x
    x = tf.subtract(tf.exp(x), tf.exp(-x))
    # x = x / e^x + e^-x
    x = tf.divide(x, tf.add(tf.exp(x), tf.exp(-x)))

    return x

@keras.saving.register_keras_serializable(name='custom_sigmoid')
def sigmoid(x):
    # x = 1/1+e^-x
    x = tf.divide(1, tf.add(1, tf.exp(-x)))

    return x

