import tensorflow as tf
import numpy as np
import time
import os

print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# For consistency across runs
np.random.seed(42)
tf.random.set_seed(42)
