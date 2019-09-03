import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as keras

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)



def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def unet(input_layer):
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_layer)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    drop3 = tf.keras.layers.Dropout(0.3)(conv3)

    up4 = tf.keras.layers.UpSampling2D(size=(2, 2))(drop3)
    up4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up4)
    merge4 = tf.keras.layers.concatenate([conv2, up4], axis=3)
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge4)

    up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up5)
    merge5 = tf.keras.layers.concatenate([conv1, up5], axis=3)
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)

    return conv5






def milad_net(input_shape=(1024, 2048, 3), load_weights=None):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    LDS1 = tf.keras.layers.Conv2D(32, (3, 3), (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(input_layer) # output size (512, 1024, 32)
    LDS1 = tf.keras.layers.BatchNormalization()(LDS1)

    LDS2 = tf.keras.layers.Conv2D(48, (3, 3), (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(LDS1) # output size (256, 512, 48)
    LDS2 = tf.keras.layers.BatchNormalization()(LDS2)

    # SMALL U-NET
    small_unet = unet(LDS2) # output size (256, 512, 64)

    # EXTRACTING GLOBAL FEATURES
    EGF1 = tf.keras.layers.SeparableConv2D(32, (3, 3), (2, 2), activation='relu', padding='same')(LDS2) # output size (128, 256, 32)
    EGF2 = tf.keras.layers.SeparableConv2D(64, (3, 3), (2, 2), activation='relu', padding='same')(LDS2) # output size (128, 256, 64)
    EGF3 = tf.keras.layers.SeparableConv2D(128, (3, 3), (2, 2), activation='relu', padding='same')(LDS2) # output size (128, 256, 128)

    ADD1 = tf.keras.layers.concatenate([EGF1, EGF2, EGF3], axis=3) # output size (128, 256, 224)
    UP1 = tf.keras.layers.UpSampling2D(size=(2, 2))(ADD1) # output size (256, 512, 224)

    ADD2 = tf.keras.layers.concatenate([UP1, small_unet], axis=3) # output size (256, 512, 288)

    FF = tf.keras.layers.UpSampling2D((2, 2))(ADD2)
    
    FF = tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(FF)

    classifier_layer = tf.keras.layers.Dropout(0.3)(FF)
    classifier_layer = tf.keras.layers.UpSampling2D((2, 2))(classifier_layer)
    classifier_layer = tf.keras.layers.Conv2D(10, (1, 1), padding='same', kernel_initializer='he_normal')(classifier_layer)
    classifier_layer = tf.keras.activations.softmax(classifier_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=classifier_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=dice_coef, metrics=['accuracy'])
    model.summary()
    return model