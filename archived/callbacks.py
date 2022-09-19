import os
import tensorflow as tf


class custom_callback(tf.keras.callbacks):
    """log lr and clear checkpoints."""

    def on_epoch_end(self, epoch, logs=None):
        lr = float(
            tf.keras.backend.get_value(self.model.optimizer.lr)
        )  # get the current learning rate
        max = 0
        for file_name in os.listdir("checkpoints"):
            val_acc = int(os.path.basename(file_name).split(".")[-2])
            if val_acc > max:
                max = val_acc
            if val_acc < max:
                os.remove(os.path.join("checkpoints", file_name))
