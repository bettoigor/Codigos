import sys, time, os
from combine_control import Control
from combine_control_2 import SMCControl
import numpy as np 
from datetime import date, datetime
from math import pi
import tensorflow as tf

# Assumes this script is being run from grl/bin
sys.path.append('/home/adalberto/Documentos/Reinforcement_Learning/grl/build')

import grlpy

def get_timedate():

    today = datetime.now()
    at_now = today.strftime("%d-%m-%Y %H:%M:%S:")
    timedate = at_now

    return timedate


def log(file_name, text):

    fout = open("log/log_SMC_Control_Try_Traking_" + file_name + ".csv", "a")
    fout.write(text+'\n')
    fout.close()



# Load configurations
envconf = grlpy.Configurator("/home/adalberto/Documentos/Reinforcement_Learning/grl/cfg/matlab/pendulum_swingup_viewer_5s.yaml")

# Instantiate configurations (construct objects)
envinst = envconf.instantiate()

# Initializing the control object
alpha = [5.5,4.5]
rho = [2.95,1.5]
K = [0.3185,0.005]


control = Control(alpha=alpha,rho=rho, K=K)
control2 = SMCControl(alpha=alpha,rho=rho, K=K,setpoint=0)

# Get reference to environment and agent
env = grlpy.Environment(envinst["environment"])
epochs = 5
max_torque = 3
curve = np.zeros(epochs)
agent_name = 'SMC_pure'
noisy = True
path = 'paper/'

# 2000 episodes
#for r in range(2000):  
obs = env.start(0)

# log file
header = 'Date '\
         'Time '\
         'Episode '\
         'Position '\
         'Velocity '\
         'Control '\
         'Alpha '\
         'Rho '\
         'K '
log(agent_name,header)



for i in range(epochs):
  terminal = 0
  count = 0
  # restart environment and agent
  obs = env.start(0)
  action = control.start(obs)

  #run episodes
  while not terminal:
    prev_obs = obs
    (obs, reward, terminal) = env.step([action])
    
    #(action, error) = control.smc(obs)
    (action, error) = control2.smc_regular(obs)
    curve[i] += reward
    count+=1

    # external random noise
    if count > 200 and count < 206 and noisy:
      action = tf.convert_to_tensor(max_torque, dtype=tf.float32)

    if count > 400 and count < 406 and noisy:
      action = tf.convert_to_tensor(-max_torque, dtype=tf.float32)

    os.system('clear')
    print(' ********** Training **************',
          '\n *'
          '\n * Episode:',i,'of ',epochs,
          '\n * Position:',round(control2.get_lim_angle(obs),4),
          '\n * Velocity:',round(obs[1],4),
          '\n * Error:',round(error,4),
          '\n * Action:',action.numpy(),
          '\n * External noise:',noisy,
          '\n * Current Reward:',round(reward,3),
          '\n * Accumulaterd Reward:',round(curve[i],3),
          '\n **********************************')

    # save log
    text = get_timedate()+\
         ' '+str(i)+\
         ' '+str(control2.get_lim_angle(obs))+\
         ' '+str(obs[1])+\
         ' '+str(action.numpy())+\
         ' '+str(control2.alpha[0])+\
         ' '+str(control2.rho[0])+\
         ' '+str(control2.K[0])
    log(agent_name,text)

    time.sleep(0.001)
  print('\nEpisode ',i,' terminated!')
  time.sleep(0.5)
