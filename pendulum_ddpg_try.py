import sys, time, os
from combine_control import Control, Memory,DDPG, TD3
import numpy as np 


# Assumes this script is being run from grl/bin
sys.path.append('/home/adalberto/Documentos/Reinforcement_Learning/grl/build')

import grlpy

# Load configurations
envconf = grlpy.Configurator("/home/adalberto/Documentos/Reinforcement_Learning/grl/cfg/matlab/pendulum_swingup.yaml")

# Instantiate configurations (construct objects)
envinst = envconf.instantiate()
control = Control(alpha=[4.5,4.5],rho=[2.285,2], K=[0.2185,0.015])

# Get reference to environment and agent
env = grlpy.Environment(envinst["environment"])
memory = Memory(3,1)
agent_name = 'Agent_08_3'
agent = DDPG(3,1, load_agent=True, name=agent_name)
gamma = 0.99
obs = env.start(0)
batch_size = 32
epochs = 20
max_torque = 3
curve = np.zeros(epochs)

for i in range(epochs):
  terminal = 0

  # restart environment and agent
  raw_obs = env.start(0)
  obs = control.get_normalized_state(raw_obs)
  action = control.start(obs)

  #run episodes
  while not terminal:
    #save previous observation
    prev_obs = obs

    # get action
    action = agent.actor(obs) * max_torque

    # apply action and receves reward 
    (raw_obs, reward, terminal) = env.step([action])
    obs = control.get_normalized_state(raw_obs)
    curve[i] += reward

    os.system('clear')
    print(' ********** Trying **************',
        '\n *'
        '\n * Agent Name:',agent_name,
        '\n * Episode:',i,'of ',epochs,
        '\n * Position:',round(control.get_lim_angle(raw_obs),4),
        '\n * Velocity:',round(raw_obs[1],4),
        '\n * Action:',action,
        '\n * Current Reward:',round(reward,3),
        '\n * Accumulaterd Reward:',round(curve[i],3),
        '\n **********************************')

  #agent.save_model(agent_name)
  #print('Model saved!')
    #print('Position:',obs,'Velocity:',round(obs[1],4),'U:',round(action[0],8),end='\r')
    #print('Memory:',bs,'U:',action,end='\r')
    #time.sleep(0.001)
  print('\nEpisode ',i,' terminated!')
  time.sleep(0.5)

