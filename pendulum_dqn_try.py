import sys, time, os
from combine_control import Control, Memory, DQN
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
net_name = 'Net_05'
epochs = 20
torques = [-3,0,3]
curve = np.zeros(epochs)

network = DQN(states=3, actions=1,load_model=True, name=net_name)

for i in range(epochs):
  # restart environment and agent
  terminal = 0
  #obs = env.start(0)
  obs = control.get_normalized_state(env.start(0))

  #run episodes
  while not terminal:

    # get action
    net_obs = network(obs, torques)
    action = torques[np.argmax(net_obs)]

    # apply action and receves reward 
    (obs, reward, terminal) = env.step([action])
    obs = control.get_normalized_state(obs)
    curve[i]+=reward
    os.system('clear')
    print(' ********** Training **************',
          '\n *'
          '\n * Episode:',i,'of ',epochs,
          '\n * Position:',round(obs[0],4),
          '\n * Velocity:',round(obs[2],4),
          '\n * Action:',action,
          '\n * Accumulaterd Reward:',round(curve[i],3),
          '\n **********************************')
  print('\nEpisode ',i,' terminated!')
  time.sleep(0.5)

