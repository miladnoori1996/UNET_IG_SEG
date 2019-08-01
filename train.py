import os
import tensorflow as tf 
import numpy as np
from data_gen import DataGen
from model import unet


NO_OF_EPOCHS = 1000
BATCH_SIZE = 2
SAMPLES = 500
# 2975
VAL_SAMPLES = 4
# 500

UNET_MODEL = unet()


# checkpoint_path = "check/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# latest = tf.train.latest_checkpoint(checkpoint_dir)

# UNET_MODEL.load_weights(latest)

train_gen = DataGen('train', batch_size=BATCH_SIZE, image_height=256, image_width=512,  split=True, amount=SAMPLES)
val_gen = DataGen('val', batch_size=BATCH_SIZE, image_height=256, image_width=512, split=True, amount=VAL_SAMPLES)

# setting checkpoints
checkpoint_path = os.path.join(os.getcwd(), "check/cp-{epoch:04d}.ckpt")
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, keep_only_last_file=True, verbose=1, save_weights_only=True,
    period=10)
UNET_MODEL.save_weights(checkpoint_path.format(epoch=0))

history = UNET_MODEL.fit_generator(train_gen, 
										epochs=NO_OF_EPOCHS,
                                        steps_per_epoch =SAMPLES//BATCH_SIZE,
                                        validation_data=val_gen,
                                        validation_steps=VAL_SAMPLES//BATCH_SIZE,
                                        verbose=1,
                                        callbacks=[cp_callback])