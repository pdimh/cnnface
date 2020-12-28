import os
import tensorflow as tf


def _config(force_cpu=False, gpu_mem_limit=None):
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        if gpu_mem_limit:
            tf.config.experimental.set_virtual_device_configuration(
                tf.config.experimental.list_physical_devices('GPU')[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem_limit)])


def configure(config):
    _config(int(config['FORCE_CPU']),
            int(config['GPU_MEM_LIMIT']) if config['GPU_MEM_LIMIT'] else None)
