import os

from telegram.ext import CommandHandler, Filters, MessageHandler, Updater

from src.models.predict import get_prediction, get_run_data, load_model


def start(update, context):
    user = update.effective_user
    name = user["first_name"]
    update.message.reply_text(
        f"""
Hi {name}! Welcome!\n
I am a rock classification bot.
Send me a photo of a rock and I will tell you what kind of rock it is.\n
I can classify rocks from in these categories Basalt, Granite, Quartz, Sandstone, Marble, Coal, and Granite.\n
You can visit \n(https://github.com/udaylunawat/Whats-this-rock)\n to check my source code!"""
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
    #     update.message.reply_text(
    #         f"""Model details can be found at \nhttps://wandb.ai/{config["pretrained_model_link"]}/
    # """
    #     )

    if os.path.exists("media/images"):
        dir = os.listdir("media/images")
        model_image = os.path.join("media", "images", dir[0])
        update.message.reply_text("Here's the Confusion matrix heatmap for the model.")
    else:
        model_image = os.path.join("imgs", "result.png")
        update.message.reply_text("Here are the plots for the model.")

    user = update.effective_user
    chat_id = user["id"]
    bot.send_photo(chat_id, photo=open(model_image, "rb"))
    print(f"{model_image} sent to user {chat_id}.")


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

    TOKEN = os.environ["TOKEN"]
    updater = Updater(TOKEN, use_context=True)
    bot = updater.bot
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(CommandHandler("model", model_details))
    dp.add_handler(MessageHandler(Filters.text, handle_message))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))

    get_run_data()  # Download model and report image
    model = load_model()
    updater.start_polling()
    updater.idle()
    print("Bot started!")
    print("Please visit {} to start using me!".format("t.me/test7385_bot"))
    print("Telegram Bot Deployed!")
