import os
import tensorflow as tf


def configure(force_cpu=False, gpu_mem_limit=None):
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        if gpu_mem_limit and len(tf.config.experimental.list_physical_devices('GPU')):
            tf.config.experimental.set_virtual_device_configuration(
                tf.config.experimental.list_physical_devices('GPU')[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem_limit)])
