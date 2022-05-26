import numpy as np
import tensorflow as tf
from datetime import datetime
from pytz import all_timezones

normalization_layer = tf.keras.layers.Rescaling(1./255)
AUTOTUNE = tf.data.AUTOTUNE

num_classes = 4

img_height, img_width = (200,200)
batch_size = 256

train_ds = tf.keras.utils.image_dataset_from_directory(
  "data/2_processed",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=10)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "data/2_processed",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, kernel_size = (3,3), input_shape = (200,200,3)),
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

def train_model():
  optimizer = tf.keras.optimizers.Adam()
  model.compile(optimizer=optimizer, loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['f1'])
  model.fit(train_ds,validation_data=val_ds,epochs=50,batch_size=10)
  timestr=datetime.now()
  timestr1=timestr.astimezone(timezone('Asia/Kolkata'))
  timestr1=str(timestr1)
  model.save("classifier_"+timestr1+".h5")