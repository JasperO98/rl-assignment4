import sys
import tensorflow as tf

# fix AlphaZero General imports
sys.path.append('alphazero')

# limit GPU memory usage
for device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
