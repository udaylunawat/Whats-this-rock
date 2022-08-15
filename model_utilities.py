import os
import time
import numpy as np

from keras.callbacks import Callback
from sklearn.utils import class_weight

from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow import image

import models

def get_optimizer(config):
    if config.optimizer == 'adam':
        opt = optimizers.Adam(learning_rate=config["lr"])
    elif config.optimizer == 'rms':
        opt = optimizers.RMSprop(learning_rate=config["lr"],
                                 rho=0.9, epsilon=1e-08, decay=0.0)
    elif config.optimizer == 'sgd':
        opt = optimizers.SGD(learning_rate=config["lr"])
    elif config.optimizer == 'adamax':
        opt = optimizers.Adamax(learning_rate=config["lr"])

    return opt


def get_model(config):
    models_dict = {
        'efficientnet': models.get_efficientnet,
        'resnet': models.get_resnet,
        'small_cnn': models.get_small_cnn,
        'large_cnn': models.get_large_cnn,
        'baseline': models.get_baseline,
        'mobilenet': models.get_mobilenet,
        'mobilenetv2': models.get_mobilenetv2,
        'inceptionresnetv2': models.get_inceptionresnetv2,
        'vgg16': models.get_vgg16
    }

    return models_dict[config.model_name](config)


def get_best_checkpoint():
    max = 0
    best_model = None
    for file_name in os.listdir('checkpoints'):
        val_acc = int(os.path.basename(file_name).split('.')[-2])
        if val_acc > max:
            max = val_acc
            best_model = file_name
    return best_model


# https://datamonje.com/image-data-augmentation/#cutout
# https://colab.research.google.com/drive/1on9rvQdr0s8CfqYeZBTbgk0K5I_Dha-X#scrollTo=py9wqsM05kio
def custom_augmentation(np_tensor):

    def random_contrast(np_tensor):
        return np.array(image.random_contrast(np_tensor, 0.5, 2))

    def random_hue(np_tensor):
        return np.array(image.random_hue(np_tensor, 0.5))

    def random_saturation(np_tensor):
        return np.array(image.random_saturation(np_tensor, 0.2, 3))

    def random_crop(np_tensor):
        # cropped height between 70% to 130% of an original height
        new_height = int(np.random.uniform(0.7, 1.30) * np_tensor.shape[0])
        # cropped width between 70% to 130% of an original width
        new_width = int(np.random.uniform(0.7, 1.30) * np_tensor.shape[1])
        # resize to new height and width
        cropped = image.resize_with_crop_or_pad(
            np_tensor, new_height, new_width)
        return np.array(image.resize(cropped, np_tensor.shape[:2]))

    def gaussian_noise(np_tensor):
        mean = 0
        # variance: randomly between 1 to 25
        var = np.random.randint(1, 26)
        # sigma is square root of the variance value
        noise = np.random.normal(mean, var**0.5, np_tensor.shape)
        return np.clip(np_tensor + noise, 0, 255).astype('int')

    def cutout(np_tensor):
        cutout_height = int(np.random.uniform(0.1, 0.2) * np_tensor.shape[0])
        cutout_width = int(np.random.uniform(0.1, 0.2) * np_tensor.shape[1])
        cutout_height_point = np.random.randint(
            np_tensor.shape[0] - cutout_height)
        cutout_width_point = np.random.randint(
            np_tensor.shape[1] - cutout_width)
        np_tensor[cutout_height_point:cutout_height_point + cutout_height,
                  cutout_width_point:cutout_width_point + cutout_width, :] = 127
        return np_tensor

    if (np.random.uniform() < 0.1):
        np_tensor = random_contrast(np_tensor)
    if (np.random.uniform() < 0.1):
        np_tensor = random_hue(np_tensor)
    if (np.random.uniform() < 0.1):
        np_tensor = random_saturation(np_tensor)
    if (np.random.uniform() < 0.2):
        np_tensor = random_crop(np_tensor)

    # Gaussian noise giving error hence removed
    # if (np.random.uniform() < 0.2):
    #     np_tensor = gaussian_noise(np_tensor)
    if (np.random.uniform() < 0.3):
        np_tensor = cutout(np_tensor)
    return np.array(np_tensor)


def get_model_weights(train_generator):
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes)

    train_class_weights = dict(enumerate(class_weights))
    return train_class_weights

############################## Training #####################################


def print_in_color(txt_msg,fore_tupple,back_tupple,):
    # prints the text_msg in the foreground color specified by fore_tupple with the background specified by back_tupple
    # text_msg is the text, fore_tupple is foregroud color tupple (r,g,b), back_tupple is background tupple (r,g,b)
    rf, gf, bf = fore_tupple
    rb, gb, bb = back_tupple
    msg = '{0}' + txt_msg
    mat = '\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' +str(gb) + ';' + str(bb) +'m'
    print(msg .format(mat), flush=True)
    print('\33[0m', flush=True)  # returns default print color to back to black
    return


class LRA(Callback):
    reset=False
    count=0
    stop_count=0
    tepochs=0
    def __init__(self,model, patience,stop_patience, threshold, factor, dwell, model_name, freeze, initial_epoch):
        super(LRA, self).__init__()
        self.model=model
        self.patience=patience # specifies how many epochs without improvement before learning rate is adjusted
        self.stop_patience=stop_patience
        self.threshold=threshold # specifies training accuracy threshold when lr will be adjusted based on validation loss
        self.factor=factor # factor by which to reduce the learning rate
        self.dwell=dwell
        self.lr=float(K.get_value(model.optimizer.lr)) # get the initiallearning rate and save it in self.lr
        self.highest_tracc=0.0 # set highest training accuracy to 0
        self.lowest_vloss=np.inf # set lowest validation loss to infinity
        #self.count=0 # initialize counter that counts epochs with no improvement
        #self.stop_count=0 # initialize counter that counts how manytimes lr has been adjustd with no improvement
        self.initial_epoch=initial_epoch
        #self.epochs=epochs
        best_weights=self.model.get_weights() # set a class vaiable so weights can be loaded after training is completed
        msg=' '
        if freeze==True:
            msgs=f' Starting training using  base model { model_name} with weights frozen to imagenet weights initializing LRA callback'
        else:
            msgs=f' Starting training using base model { model_name} training all layers '
        print_in_color (msgs, (244, 252, 3), (55,65,80))

    def on_epoch_begin(self,epoch, logs=None):
        self.now= time.time()

    def on_epoch_end(self, epoch, logs=None):  # method runs on the end of each epoch
        later=time.time()
        duration=later-self.now
        if epoch== self.initial_epoch or LRA.reset==True:
            LRA.reset=False
            msg='{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^11s}{8:^8s}'.format('Epoch', 'Loss', 'Accuracy','V_loss','V_acc', 'LR', 'Next LR', 'Monitor', 'Duration')
            print_in_color(msg, (244,252,3), (55,65,80))

        lr=float(K.get_value(self.model.optimizer.lr)) # get the current learning rate
        current_lr=lr
        v_loss=logs.get('val_loss')  # get the validation loss for this epoch
        acc=logs.get('accuracy')  # get training accuracy
        v_acc=logs.get('val_accuracy')
        loss=logs.get('loss')
        #print ( '\n',v_loss, self.lowest_vloss, acc, self.highest_tracc)
        if acc < self.threshold: # if training accuracy is below threshold adjust lr based on training accuracy
            monitor='accuracy'
            if acc>self.highest_tracc: # training accuracy improved in the epoch
                self.highest_tracc=acc # set new highest training accuracy
                LRA.best_weights=self.model.get_weights() # traing accuracy improved so save the weights
                self.count=0 # set count to 0 since training accuracy improved
                self.stop_count=0 # set stop counter to 0
                if v_loss<self.lowest_vloss:
                    self.lowest_vloss=v_loss
                color= (0,255,0)
                self.lr=lr
            else:
                # training accuracy did not improve check if this has happened for patience number of epochs
                # if so adjust learning rate
                if self.count>=self.patience -1:
                    color=(255,0,0)
                    self.lr= lr* self.factor # adjust the learning by factor
                    K.set_value(self.model.optimizer.lr, self.lr) # set the learning rate in the optimizer
                    self.count=0 # reset the count to 0
                    self.stop_count=self.stop_count + 1
                    if self.dwell:
                        self.model.set_weights(LRA.best_weights) # return to better point in N space
                    else:
                        if v_loss<self.lowest_vloss:
                            self.lowest_vloss=v_loss
                else:
                    self.count=self.count +1 # increment patience counter
        else: # training accuracy is above threshold so adjust learning rate based on validation loss
            monitor='val_loss'
            if v_loss< self.lowest_vloss: # check if the validation loss improved
                self.lowest_vloss=v_loss # replace lowest validation loss with new validation loss
                LRA.best_weights=self.model.get_weights() # validation loss improved so save the weights
                self.count=0 # reset count since validation loss improved
                self.stop_count=0
                color=(0,255,0)
                self.lr=lr
            else: # validation loss did not improve
                if self.count>=self.patience-1:
                    color=(255,0,0)
                    self.lr=self.lr * self.factor # adjust the learning rate
                    self.stop_count=self.stop_count + 1 # increment stop counter because lr was adjusted
                    self.count=0 # reset counter
                    K.set_value(self.model.optimizer.lr, self.lr) # set the learning rate in the optimizer
                    if self.dwell:
                        self.model.set_weights(LRA.best_weights) # return to better point in N space
                else:
                    self.count =self.count +1 # increment the patience counter
                if acc>self.highest_tracc:
                    self.highest_tracc= acc
        msg=f'{str(epoch+1):^3s}/{str(LRA.tepochs):4s} {loss:^9.3f}{acc*100:^9.3f}{v_loss:^9.5f}{v_acc*100:^9.3f}{current_lr:^9.5f}{self.lr:^9.5f}{monitor:^11s}{duration:^8.2f}'
        print_in_color (msg,color, (55,65,80))
        if self.stop_count> self.stop_patience - 1: # check if learning rate has been adjusted stop_count times with no improvement
            msg=f' training has been halted at epoch {epoch + 1} after {self.stop_patience} adjustments of learning rate with no improvement'
            print_in_color(msg, (0,255,0), (55,65,80))
            self.model.stop_training = True # stop training

class delete_checkpoints(Callback):
    def on_epoch_end(self, epoch, logs=None):
        max = 0
        for file_name in os.listdir('checkpoints'):
            val_acc = int(os.path.basename(file_name).split('.')[-2])
            if val_acc > max:
                max = val_acc
            if val_acc < max:
                os.remove(os.path.join('checkpoints', file_name))
