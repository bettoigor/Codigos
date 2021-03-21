import sys, time, os
from combine_control_2 import SMCControl, Memory, SmartTD3, SmartControl, TD3
import numpy as np 
from math import pi
from datetime import date, datetime
import tensorflow as tf

# Assumes this script is being run from grl/bin
sys.path.append('/home/adalberto/Documentos/Reinforcement_Learning/grl/build')

import grlpy

def get_timedate():

    today = datetime.now()
    at_now = today.strftime("%d-%m-%Y %H:%M:%S:")
    timedate = at_now

    return timedate


def log(file_name, text, path=None):

    if path is None:
      path=''

    fout = open(path+"log/log_TD3_SMC_try_" + file_name + "_no_noise.csv", "a")
    fout.write(text+'\n')
    fout.close()



# Environment configurations
envconf = grlpy.Configurator("/home/adalberto/Documentos/Reinforcement_Learning/grl/cfg/matlab/pendulum_swingup_viewer_5s.yaml")
envinst = envconf.instantiate()
env = grlpy.Environment(envinst["environment"])

# Initializing the control object
# Cabalistic values
#'''
alpha = [0.,4.5]
rho = [0.,1]
K = [0.,0.015]
'''
alpha = [1.,4.5]
rho = [1.,1]
K = [1.,0.015]
'''
a = 1.
r = 1.
k = 1.

control = SMCControl(alpha=alpha,rho=rho, K=K,setpoint=0)

# Initializing RL objects
num_states = 3
num_actions = 3

# Hyperparameters
gains = [0.,0.,0.]
epochs = 1
curve = np.zeros(epochs)
max_torque = 3.

# automated test loop
num_tests = 1
save_log = False
noisy = True
path = 'paper/'
for test in range(0,num_tests):

  # loading agent
  agent_name = 'TD3_SMC_Agent_1'
  agent = TD3(num_states,num_actions, load_agent=True, path=path, name=agent_name ,load_critic=False)


# log file
  if save_log:
    header = 'Date '\
             'Time '\
             'Episode '\
             'Position '\
             'Velocity '\
             'Control '\
             'Alpha '\
             'Rho '\
             'K '
    log(agent_name,header,path=path)


  # Work Loop
  for i in range(epochs):
      # restarting simulation
      terminal = 0
      raw_obs = env.start(0)
      ctrl_action = control.start(raw_obs)
      norm_obs = control.get_normalized_state(raw_obs)
      count = 0

      # run epsodes
      while not terminal:

          # applying the controller action in the environment
          (raw_obs, reward, terminal) = env.step([ctrl_action])
          norm_obs = control.get_normalized_state(raw_obs)
          curve[i] += reward
          count+=1

          # getting crontroller action
          ctrl_action, error = control.smc_regular(raw_obs)
          
          # external noise
          if count > 200 and count < 204 and noisy:
            ctrl_action = tf.convert_to_tensor(max_torque, dtype=tf.float32)

          if count > 400 and count < 404 and noisy:
            ctrl_action = tf.convert_to_tensor(-max_torque, dtype=tf.float32)    

          # getting adjustment action from the Actor agent
          adj_action = agent.actor(norm_obs, noise=None)
          gains = [abs(adj_action[0]),
                      abs(adj_action[1]),
                      abs(adj_action[2])]
          control.set_gain(gains)

          #print(chr(27) + "[2J") 
          os.system('clear')
          print(' ********** Trying **************',
                '\n *'
                '\n * Momdel Name:',agent_name,
                '\n * Episode:',i,'of ',epochs,
                '\n * Position:',round(control.get_lim_angle(raw_obs),4),
                '\n * Position:',round(raw_obs[0],4),
                '\n * Velocity:',round(raw_obs[1],4),
                '\n * Action:',round(ctrl_action.numpy(),4),
                '\n * Error:',round(error,5),
                '\n * Adj Action:',adj_action,        
                '\n * External noise:',noisy,        
                '\n * alpha:',round(control.alpha[0],5),
                '\n * rho:',round(control.rho[0],5),
                '\n * K:',round(control.K[0],5),              
                '\n * Current Reward:',round(reward,3),
                '\n * Accumulaterd Reward:',round(curve[i],3),
                '\n **********************************')
          # save log
          text = get_timedate()+\
               ' '+str(i)+\
               ' '+str(control.get_lim_angle(raw_obs))+\
               ' '+str(raw_obs[1])+\
               ' '+str(ctrl_action.numpy())+\
               ' '+str(control.alpha[0])+\
               ' '+str(control.rho[0])+\
               ' '+str(control.K[0])
          if save_log:
            log(agent_name,text,path=path)

      print('Done!')
      time.sleep(0.5)
    

