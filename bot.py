import numpy as np
import json
from telegram.ext import Updater, CommandHandler, Filters, MessageHandler
from io import BytesIO
import cv2
import tensorflow as tf
from tensorflow import keras

from train import train_model

def get_keys(path):
    with open(path) as f:
        return json.load(f)

TOKEN = get_keys("secrets.json")['TOKEN']
normalization_layer = tf.keras.layers.Rescaling(1./255)
AUTOTUNE = tf.data.AUTOTUNE

num_classes = 7

img_height, img_width = (200,200)
batch_size = 256

class_names = ['igneous rocks', 'metamorphic rocks', 'minerals', 'sedimentary rocks']

model = keras.models.load_model('/content/classifier_61_2022-05-26 00_26_27.439409+05_30.h5')
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['f1'])

#chatbot code
def start(update, context):
    update.message.reply_text("Welcome!")

#chatbot code
def help(update, context):
    update.message.reply_text("""
    /start - Starts conversation
    /help - Shows this message
    /train - Trains neural networks
    """)

#chatbot code
def train(update, context):
    update.message.reply_text("Model is being trained...")
    train_model()
    update.message.reply_text("Done! You can now send a photo!")

#chatbot code
def handle_message(update, context):
    update.message.reply_text("Please train the model and send a picture!")

#chatbot code
def handle_photo(update, context):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (200,200), interpolation=cv2.INTER_AREA)
    #prediction=model.predict(img)[0]
    #print(prediction)
    prediction = model.predict(np.array([img / 255]))
    update.message.reply_text(f"In this image I see a {class_names[np.argmax(prediction)]}")

#chatbot code
updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher
dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("help", help))
dp.add_handler(CommandHandler("train", train))
dp.add_handler(MessageHandler(Filters.text, handle_message))
dp.add_handler(MessageHandler(Filters.photo, handle_photo))

print("Telegram Bot Deployed!")

#chatbot code
updater.start_polling()
updater.idle()