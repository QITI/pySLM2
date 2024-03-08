import os
print("CUDA_VISIBLE_DEVICES before TensorFlow import:", os.environ.get('CUDA_VISIBLE_DEVICES'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import tensorflow as tf
print("CUDA_VISIBLE_DEVICES after TensorFlow import:", os.environ.get('CUDA_VISIBLE_DEVICES'))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
