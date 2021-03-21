import sys, time, os
from combine_control_2 import PDControl
import numpy as np 
from datetime import date, datetime

# Assumes this script is being run from grl/bin
sys.path.append('/home/adalberto/Documentos/Reinforcement_Learning/grl/build')

import grlpy

def get_timedate():

    today = datetime.now()
    at_now = today.strftime("%d-%m-%Y %H:%M:%S:")
    timedate = at_now

    return timedate


def log(file_name, text):

    fout = open("log/log_PD_Control_Try_" + file_name + ".csv", "a")
    fout.write(text+'\n')
    fout.close()


# Load configurations
envconf = grlpy.Configurator("/home/adalberto/Documentos/Reinforcement_Learning/grl/cfg/matlab/pendulum_swingup_viewer.yaml")

# Instantiate configurations (construct objects)
envinst = envconf.instantiate()
control = PDControl(K_p=20,K_d=0.5)
agent_name = 'PD_pure'

# Get reference to environment and agent
env = grlpy.Environment(envinst["environment"])
epochs = 5
curve = np.zeros(epochs)

# 2000 episodes
obs = env.start(0)

# log file
header = 'Date '\
         'Time '\
         'Episode '\
         'Position '\
         'Velocity '\
         'Control '\
         'K_p '\
         'K_d '

log(agent_name,header)



for i in range(epochs):
  terminal = 0

  # restart environment and agent
  obs = env.start(0)
  action = control.start(obs)

  #run episodes
  while not terminal:
    prev_obs = obs
    (obs, reward, terminal) = env.step([action])
    norm_obs = control.get_normalized_state(obs)
    (action, error) = control.pd(obs)
    curve[i] += reward

    os.system('clear')
    print(' ********** Training **************',
          '\n *'
          '\n * Episode:',i,'of ',epochs,
          '\n * Position:',round(control.get_lim_angle(obs),4),
          '\n * Velocity:',round(obs[1],4),
          '\n * Error:',round(error,4),
          '\n * Action:',round(action.numpy(),4),
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
