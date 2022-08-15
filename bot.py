import numpy as np
import json
from telegram.ext import Updater, CommandHandler, Filters, MessageHandler
from io import BytesIO
import os
import cv2
from tensorflow.data import AUTOTUNE
from tensorflow.keras import layers, models
import tensorflow_addons as tfa


def get_keys(path):
    with open(path) as f:
        return json.load(f)


def start(update, context):
    update.message.reply_text("""
    Welcome!\nI am a rock classification bot.\n
Send me a photo of a rock and I will tell you what kind of rock it is.\n
I can classify rocks from basalt, granite, quartz, sandstone, marble, coal, and granite.""")


def help(update, context):
    update.message.reply_text("""
    /start - Starts conversation
    /help - Shows this message
    /train - Trains neural networks
    """)


def train(update, context):
    pass
    # update.message.reply_text("Model is being trained...")
    # # train logic
    # update.message.reply_text("Done! You can now send a photo!")


def handle_message(update, context):
    update.message.reply_text("Please send a picture of a rock!")


def handle_photo(update, context):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (config['image_size'], config['image_size']), interpolation=cv2.INTER_AREA)
    # prediction=model.predict(img)[0]
    # print(prediction)
    prediction = model.predict(np.array([img / 255]))
    update.message.reply_text(f"In this image I see {class_names[np.argmax(prediction)]}!")

if __name__ == "__main__":

    # read config file
    with open('config.json') as config_file:
        config = json.load(config_file)

    print("Bot started!")
    print("Downloading model...")
    os.system('wget -O model.hdf5 https://www.dropbox.com/s/msqkqjabo807stl/charmed-sweep-1-efficientnet-epoch-16_val_accuracy-0.52.hdf5')
    TOKEN = get_keys("secrets.json")['TOKEN']
    normalization_layer = layers.Rescaling(1. / 255)
    AUTOTUNE = AUTOTUNE

    num_classes = config['num_classes']

    img_height, img_width = (config['image_size'], config['image_size'])
    batch_size = config['batch_size']

    class_names = ['Basalt', 'Coal', 'Granite', 'Limestone', 'Marble', 'Quartz', 'Sandstone']

    model = models.load_model('model.hdf5')
    optimizer = tf.keras.optimizers.Adam()
    f1_score = tfa.metrics.F1Score(
        num_classes=config['num_classes'],
        average='macro',
        threshold=0.5)
    model.compile(optimizer=optimizer, loss=tf.losses.CategoricalCrossentropy(), metrics=['accuracy', f1_score])

    print("Model loaded!")
    print("Please visit {} to start using me!".format("t.me/test7385_bot"))


    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    # dp.add_handler(CommandHandler("train", train))
    dp.add_handler(MessageHandler(Filters.text, handle_message))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))

    print("Telegram Bot Deployed!")

    # chatbot code
    updater.start_polling()
    updater.idle()
