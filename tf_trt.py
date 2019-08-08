import tensorflow as tf 
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50))) as sess:
	saver = tf.train.import_meta_graph("tf_model/model.meta")
	saver.restore(sess, "tf_model/model")
	your_outputs = ['out_put_1/Sigmoid']
	frozen_graph = tf.graph_util.convert_variables_to_constants(
		sess, # session
		tf.get_default_graph().as_graph_def(),# graph+weight from the session
		output_node_names=your_outputs)
	with gfile.FastGFile("tf_model/frozen_model.pb", 'wb') as f:
		f.write(frozen_graph.SerializeToString())

trt_graph = trt.create_inference_graph(
	input_graph_def=frozen_graph,# frozen model
	outputs=your_outputs,
	max_batch_size=2,# specify your max batch size
	max_workspace_size_bytes=2*(10**9),# specify the max workspace
	precision_mode="FP32")

with gfile.FastGFile("trt/TensorRT_model.pb", 'wb') as f:
	f.write(trt_graph.SerializeToString())
