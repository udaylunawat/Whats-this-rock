import os
import cv2
import json
import telegram
from telegram.ext import Updater, CommandHandler, Filters, MessageHandler
from predict import get_prediction

# read config file
with open("config.json") as config_file:
    config = json.load(config_file)


def start(update, context):
    update.message.reply_text(
        """
    Welcome!\nI am a rock classification bot.
Send me a photo of a rock and I will tell you what kind of rock it is.\n
I can classify rocks from in these categories Basalt, Granite, Quartz, Sandstone, Marble, Coal, and Granite.\n
You can visit [here](https://github.com/udaylunawat/Whats-this-rock) to check my source code!"""
    )


def help(update, context):
    update.message.reply_text(
        """
    /start - Starts conversation
/help - Shows this message
/model - Show model details
"""
    )


def model_details(update, context):
    update.message.reply_text(
        f"""Model details can be found at https://wandb.ai/{config["pretrained_model_link"]}/
"""
    )
    for img in os.listdir('media/images'):
        if img.endswith('png'):
            cr = img

    cr_path = os.path.join('media', 'images', cr)
    update.message.reply_text("Confusion report for model.")
    print(cr)
    img = cv2.imread(cr)
    bot.send_photo(photo=open(cr_path, 'rb'))


def handle_message(update, context):
    update.message.reply_text(
        """Please send a picture of a rock!\n
Or type /help to learn more.
"""
    )


def handle_photo(update, context):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    update.message.reply_text(get_prediction(file))


if __name__ == "__main__":

    # read config file
    with open("config.json") as config_file:
        config = json.load(config_file)

    print("Bot started!")
    print("Please visit {} to start using me!".format("t.me/test7385_bot"))

    TOKEN = os.environ["TOKEN"]
    bot = telegram.Bot(token=TOKEN)
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(CommandHandler("model", model_details))
    dp.add_handler(MessageHandler(Filters.text, handle_message))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))

    print("Telegram Bot Deployed!")

    updater.start_polling()
    updater.idle()
