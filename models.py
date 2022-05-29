from tensorflow.keras.applications import MobileNetV2, EfficientNetV2B0
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

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
    return model

def model3():

  mobilenet_pretrained = MobileNetV2(
      input_shape=(160, 160, 3), weights="imagenet", include_top=False
  )

  # Freeze layers
  mobilenet_pretrained.trainable = True

  # Add untrained final layers
  model = tf.keras.Sequential(
      [
          mobilenet_pretrained,
          tf.keras.layers.GlobalAveragePooling2D(),
          tf.keras.layers.Dense(1024),
          tf.keras.layers.Dense(num_classes, activation="softmax"),
      ]
  )


def baseline_model(IMG_SIZE, CHANNELS, OUTPUT):
    input = tf.keras.layers.Input(shape=(IMG_SIZE,IMG_SIZE,CHANNELS))
    flt_1 = tf.keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS))(input)
    Dense_1 = tf.keras.layers.Dense(128, activation="relu")(flt_1)
    D_out_1 = tf.keras.layers.Dropout(0.1)(Dense_1)
    Dense_2 = tf.keras.layers.Dense(64, activation="relu")(D_out_1)
    output =  tf.keras.layers.Dense(OUTPUT, activation="softmax")(Dense_2)
    model = tf.keras.models.Model(input, output)

    return model