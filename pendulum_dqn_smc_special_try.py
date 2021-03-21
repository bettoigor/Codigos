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
control = Control(alpha=[4.5,4.5],rho=[2.285,1.], K=[0.2185,0.015])

# Get reference to environment and agent
env = grlpy.Environment(envinst["environment"])
net_name = 'Net_06_DQN_SMC_SPECIAL_2'
epochs = 20
torques = [-3,0,3]
act_index = [0,1,2,3,4]

#torques = [-3,0,3,4]
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
    net_obs = network(obs, act_index)
    action = act_index[np.argmax(net_obs)]
    smc_ = False

    if action < 3:
      true_action = torques[action]
      smc_ = False
    else:
      if action == 3:
        true_action = control.smc_regular(obs) 
      if action == 4:
        true_action = control.smc_special(obs)
      smc_ = True


    # apply action and receves reward 
    (obs, reward, terminal) = env.step([true_action])
    obs = control.get_normalized_state(obs)
    curve[i]+=reward
    os.system('clear')
    print(' ********** Training **************',
          '\n *'
          '\n * Episode:',i,'of ',epochs,
          '\n * Position:',round(control.get_lim_angle(obs),4),
          '\n * Velocity:',round(obs[1],4),
          '\n * Action:',action,
          '\n * SMC:',smc_,
          '\n * Accumulaterd Reward:',round(curve[i],3),
          '\n **********************************')
  print('\nEpisode ',i,' terminated!')
  time.sleep(0.5)

