from combine_control_2 import SmartControl
import tensorflow as tf
import numpy as np 
from math import pi

control = SmartControl()

print('Alpha:',control.alpha.numpy())
print('Setpoint:',control.setpoint)
print('Gains K',control.K.numpy(),'rho', control.rho.numpy())
raw_state = np.random.rand(2,)
#a = tf.convert_to_tensor(np.atleast_2d(a), dtype=tf.float32)

raw_state = np.atleast_2d([0.01,15.2]) #tf.expand_dims(raw_state,axis=1)
print('Raw State:',raw_state)
print('Control Signal:',control.smc_regular(raw_state).numpy())

