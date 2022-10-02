
from __future__ import print_function
from sklearn.model_selection import train_test_split
import os
import numpy as np
from utils import ADAMLearningRateTracker
import cloud_net_model
from losses import jacc_coef
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from generators import mybatch_generator_train, mybatch_generator_validation
import pandas as pd
from utils import get_input_image_names


def train():
    model = cloud_net_model.model_arch(input_rows=in_rows,
                                       input_cols=in_cols,
                                       num_of_channels=num_of_channels,
                                       num_of_classes=num_of_classes)
    model.compile(optimizer=Adam(lr=starting_learning_rate), loss=jacc_coef, metrics=[jacc_coef])
    # model.summary()

    model_checkpoint = ModelCheckpoint(newweights_path, monitor='val_loss', save_best_only=False,period=1)
    lr_reducer = ReduceLROnPlateau(factor=decay_factor, cooldown=0, patience=patience, min_lr=end_learning_rate, verbose=1)
    csv_logger = CSVLogger(experiment_name + '_log_1.log')

    train_img_split, val_img_split, train_msk_split, val_msk_split = train_test_split(train_img, train_msk,
                                                                                      test_size=val_ratio,
                                                                                      random_state=42, shuffle=True)

    if train_resume:
        model.load_weights(preweights_path)
        print("\nTraining resumed...")
    else:
        print("\nTraining started from scratch... ")

    print("Experiment name: ", experiment_name)
    print("Input image size: ", (in_rows, in_cols))
    print("Number of input spectral bands: ", num_of_channels)
    print("Learning rate: ", starting_learning_rate)
    print("Batch size: ", batch_sz, "\n")

    model.fit_generator(
        generator=mybatch_generator_train(list(zip(train_img_split, train_msk_split)), in_rows, in_cols, batch_sz, max_bit),
        steps_per_epoch=np.ceil(len(train_img_split) / batch_sz), epochs=max_num_epochs, verbose=1,
        validation_data=mybatch_generator_validation(list(zip(val_img_split, val_msk_split)), in_rows, in_cols, batch_sz, max_bit),
        validation_steps=np.ceil(len(val_img_split) / batch_sz),
        callbacks=[model_checkpoint, lr_reducer, ADAMLearningRateTracker(end_learning_rate), csv_logger])


GLOBAL_PATH = '/home/wxw/.local/share/cloud-net/Cloud-Net/38-Cloud dataset'
TRAIN_FOLDER = os.path.join(GLOBAL_PATH, '38-Cloud_training')
TEST_FOLDER = os.path.join(GLOBAL_PATH, '38-Cloud_test')
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
in_rows = 192
in_cols = 192
num_of_channels = 4
num_of_classes = 1
starting_learning_rate = 1e-5
end_learning_rate = 1e-8
max_num_epochs = 2  # just a huge number. The actual training should not be limited by this value
val_ratio = 0.2
patience = 15
decay_factor = 0.7
batch_sz = 16
max_bit = 255  # maximum gray level in landsat 8 images
experiment_name = "Cloud-Net"
preweights_path = os.path.join(GLOBAL_PATH, 'Cloud-Net_trained_on_38-Cloud_training_patches.h5')
newweights_path = os.path.join(GLOBAL_PATH, 'new_trained.h5')
train_resume = True

# getting input images names
train_patches_csv_name = 'test.csv'
df_train_img = pd.read_csv(os.path.join(TRAIN_FOLDER, train_patches_csv_name))
train_img, train_msk = get_input_image_names(df_train_img, TRAIN_FOLDER, if_train=True)

train()
