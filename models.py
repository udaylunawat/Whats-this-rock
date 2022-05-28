import tensorflow as tf
from config import img_height, img_width, learning_rate, batch_size, num_classes

def model1():
  inputs = tf.keras.layers.Input(shape=(img_height, img_width, 3))

  x = tf.keras.layers.Conv2D(filters=32,
                          kernel_size=(3,3),
                          strides=(1,1),
                          padding='valid',
                          activation='relu')(inputs)

  x = tf.keras.layers.Conv2D(filters=32,
                          kernel_size=(3,3),
                          strides=(1,1),
                          padding='valid',
                          activation='relu')(x)

  x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

  x = tf.keras.layers.GlobalAveragePooling2D()(x)

  x = tf.keras.layers.Dense(128, activation='relu')(x)
  x = tf.keras.layers.Dense(32, activation='relu')(x)

  outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

  return tf.keras.models.Model(inputs=inputs, outputs=outputs)

def model2(img_height, img_width):
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, kernel_size = (3,3), input_shape = (img_height , img_width ,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2D(32, kernel_size = (3,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPooling2D(5,5),

            tf.keras.layers.Conv2D(64, kernel_size = (3,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2D(128, kernel_size = (3,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPooling2D(5,5),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(64),
            tf.keras.layers.Dropout(rate = 0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Dense(32),
            tf.keras.layers.Dropout(rate = 0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Dense(16),
            tf.keras.layers.Dropout(rate = 0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(1),

            tf.keras.layers.Dense(4, activation = 'softmax'),
            tf.keras.layers.Dense(num_classes)
            ])