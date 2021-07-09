from tensorflow.keras.losses import Huber
from tensorflow.keras import backend as BK

def huber_loss(delta=10):
    return Huber(delta)

def mapping_to_target_range(x, target_min=0, target_max=100):
    x02 = BK.tanh(x) + 1  # x in range(0,2)
    scale = (target_max - target_min) / 2.
    return x02 * scale + target_min

