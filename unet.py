import tensorflow as tf

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1


#Build the model
inputs = tf.keras.layers.Input(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

c1 = tf.keras.layers.Conv2D(16, (3,3),  activation='relu', kernel_initializer = 'he_normal')