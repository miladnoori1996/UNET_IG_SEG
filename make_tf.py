import tensorflow as tf
from tensorflow.keras.models import load_model
from model import unet

tf.keras.backend.set_learning_phase(0)

MODEL_PATH = 'tf_model/model'

# MODEL_LOAD = 'UNET_MODEL.h5'
# model = load_model(MODEL_LOAD)

UNET_MODEL = unet()
latest = "check/cp-0370.ckpt"
UNET_MODEL.load_weights(latest)
UNET_MODEL.summary()

x = [out.op.name for out in UNET_MODEL.outputs]
print(x)
x = [out.op.name for out in UNET_MODEL.inputs]
print(x)

saver = tf.train.Saver()
sess = tf.keras.backend.get_session()
save_path = saver.save(sess, MODEL_PATH)


