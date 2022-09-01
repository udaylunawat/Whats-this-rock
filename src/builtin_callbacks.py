import os
import wandb
import tensorflow as tf


def get_earlystopper(args):
    args = args.callback_config

    earlystopper = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=args.early_patience, verbose=0, mode='auto',
        restore_best_weights=True
    )

    return earlystopper

def get_reduce_lr_on_plateau(args):
    args = args.callback_config

    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=args.rlrp_factor,
        patience=args.rlrp_patience
    )

    return reduce_lr_on_plateau
