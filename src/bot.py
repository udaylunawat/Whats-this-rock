import os
import json
from telegram.ext import Updater, CommandHandler, Filters, MessageHandler
from predict import get_prediction


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
/help - Shows this message\n
"""
    )


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
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(MessageHandler(Filters.text, handle_message))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))

    print("Telegram Bot Deployed!")

    updater.start_polling()
    updater.idle()
