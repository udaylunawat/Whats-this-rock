import numpy as np
import tensorflow as tf
from datetime import datetime
from pytz import all_timezones

from config import img_height, img_width, learning_rate, batch_size, num_classes
from models import *

normalization_layer = tf.keras.layers.Rescaling(1./255)
AUTOTUNE = tf.data.AUTOTUNE


train_ds = tf.keras.utils.image_dataset_from_directory(
  "data/2_processed",
  validation_split=0.2,
  subset="training",
  seed=123,
  color_mode='rgb',
  image_size=(img_height, img_width),
  batch_size=10)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "data/2_processed",
  validation_split=0.2,
  subset="validation",
  seed=123,
  color_mode='rgb',
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


model = model1()

def train_model():
  optimizer = tf.keras.optimizers.Adam()
  model.compile(optimizer=optimizer, loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['f1'])
  model.fit(train_ds,validation_data=val_ds,epochs=50,batch_size=10)
  timestr=datetime.now()
  timestr1=timestr.astimezone(timezone('Asia/Kolkata'))
  timestr1=str(timestr1)
  model.save("classifier_"+timestr1+".h5")