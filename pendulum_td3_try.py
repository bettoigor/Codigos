import sys, time, os
from combine_control import Control, Memory, TD3
import numpy as np 
from datetime import date, datetime
from beepy import beep

# Assumes this script is being run from grl/bin
sys.path.append('/home/adalberto/Documentos/Reinforcement_Learning/grl/build')

import grlpy

def get_timedate():

    today = datetime.now()
    at_now = today.strftime("%d-%m-%Y %H:%M:%S:")
    timedate = at_now

    return timedate


def log(file_name, text):

    fout = open("paper/log/log_TD3_try_" + file_name + "_no_noise.csv", "a")
    fout.write(text+'\n')
    fout.close()



# Load configurations
envconf = grlpy.Configurator("/home/adalberto/Documentos/Reinforcement_Learning/grl/cfg/matlab/pendulum_swingup_viewer_5s.yaml")

# Instantiate configurations (construct objects)
envinst = envconf.instantiate()
control = Control(alpha=[4.5,4.5],rho=[2.285,2], K=[0.2185,0.015])

# Get reference to environment and agent
env = grlpy.Environment(envinst["environment"])
path = "paper/"

# training paramters
epochs = 3
max_torque = 3
curve = np.zeros(epochs)
save_log = False
noisy = True
num_test = 1

# automated testing loop
for test in range(num_test):

  # loading agent
  agent_name = 'Agent_auto_0'#+str(test)
  agent = TD3(3,1, load_agent=True, name=agent_name, path=path, load_critic=False)

  # log file
  if save_log:
    log(agent_name,get_timedate()+' Start!')
    header = 'Date '\
             'Time '\
             'Episode '\
             'Position '\
             'Velocity '\
             'Control '
    log(agent_name,header)


  for i in range(epochs):
    terminal = 0
    count = 0
    # restart environment and agent
    raw_obs = env.start(0)
    obs = control.get_normalized_state(raw_obs)
    action = control.start(obs)

    #run episodesyou u
    while not terminal:
      #save previous observation
      prev_obs = obs
      count+=1

      # get action
      action = agent.actor(obs) * max_torque
      # external noise
      #ext_noise = np.random.rand(1)
      if count > 200 and count < 206 and noisy:
        action = [max_torque]

      if count > 400 and count < 406 and noisy:
        action = [-max_torque]

      # apply action and receves reward 
      (raw_obs, reward, terminal) = env.step([action])
      obs = control.get_normalized_state(raw_obs)
      curve[i] += reward

      os.system('clear')
      print(' ********** Trying #'+str(test),' **************',
          '\n *'
          '\n * Agent Name:',agent_name,
          '\n * With external noise:',noisy,
          '\n * Episode:',i+1,'of ',epochs,
          '\n * Position:',round(control.get_lim_angle(raw_obs),4),
          '\n * Velocity:',round(raw_obs[1],4),
          '\n * Action:',action[0],
          '\n * Current Reward:',round(reward,3),
          '\n * Accumulaterd Reward:',round(curve[i],3),
          '\n * Count:',count,
          '\n ************************************')

      # save log
      text = get_timedate()+\
           ','+str(i)+\
           ','+str(control.get_lim_angle(raw_obs))+\
           ','+str(raw_obs[1])+\
           ','+str(action[0])
      if save_log:
        log(agent_name,text)
  #beep(sound=4)
#beep(sound=1)
