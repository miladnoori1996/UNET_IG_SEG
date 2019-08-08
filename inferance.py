import os
import time
import tensorflow as tf
import numpy as np 
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.platform import gfile

IMG_HEIGHT = 256
IMG_WIDTH = 512

IMG_1_PATH = "cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
IMG_2_PATH = "cityscapes/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png"

image1 = image.img_to_array(image.load_img(IMG_1_PATH, target_size=(IMG_HEIGHT, IMG_WIDTH)))
image2 = image.img_to_array(image.load_img(IMG_2_PATH, target_size=(IMG_HEIGHT, IMG_WIDTH)))

img1 = np.asarray(image1)
img2 = np.asarray(image2)

input_img = np.concatenate(
	(img1.reshape((1, IMG_HEIGHT, IMG_WIDTH, 3)), img2.reshape((1, IMG_HEIGHT, IMG_WIDTH, 3))),
	axis=0)

def read_pb_graph(model):
	with gfile.FastGFile(model,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	return graph_def

TENSORRT_MODEL_PATH = 'trt/TensorRT_model.pb'
graph = tf.Graph()
with graph.as_default():
	with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50))) as sess:
		trt_graph = read_pb_graph(TENSORRT_MODEL_PATH)
		tf.import_graph_def(trt_graph, name='')
		layers = [n.name for n in tf.get_default_graph().as_graph_def().node]
		print(layers)
		input = sess.graph.get_tensor_by_name('input_2:0')
		output = sess.graph.get_tensor_by_name('out_put_1/Sigmoid:0')
		total_time = 0; n_time_inference = 50
		out_pred = sess.run(output, feed_dict={input: input_img})
		for i in range(n_time_inference):
			t1 = time.time()
			out_pred = sess.run(output, feed_dict={input: input_img})
			t2 = time.time()
			delta_time = t2 - t1
			total_time += delta_time
			print("needed time in inference-" + str(i) + ": ", delta_time)
		avg_time_tensorRT = total_time / n_time_inference
		print("average inference time: ", avg_time_tensorRT)