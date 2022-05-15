import sys
import tensorflow as tf
import pandas as pd
import sklearn as sk
import keras

check_gpu = (len(tf.config.list_physical_devices('GPU'))>0)

print(check_gpu)