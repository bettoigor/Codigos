import sys, time
from combine_control import Control, Memory, DDPG
import numpy as np 


# Assumes this script is being run from grl/bin
sys.path.append('/home/adalberto/Documentos/Reinforcement_Learning/grl/build')

import grlpy

# Load configurations
envconf = grlpy.Configurator("/home/adalberto/Documentos/Reinforcement_Learning/grl/cfg/matlab/pendulum_swingup.yaml")
agentconf = grlpy.Configurator("/home/adalberto/Documentos/Reinforcement_Learning/grl/cfg/matlab/sarsa.yaml")

# Instantiate configurations (construct objects)
envinst = envconf.instantiate()
agentinst = agentconf.instantiate()
control = Control(alpha=[4.5,4.5],rho=[2.285,10], K=[0.2185,0.45])

# Get reference to environment and agent
env = grlpy.Environment(envinst["environment"])
agent = grlpy.Agent(agentinst["agent"])
memory = Memory(2,1)

# 2000 episodes
#for r in range(2000):  
obs = env.start(0)
while True:
  terminal = 0

  # restart environment and agent
  obs = env.start(0)
  action = control.start(obs)

  #run episodes
  while True:
    prev_obs = obs
    (obs, reward, terminal) = env.step([action])
    (action, error) = control.smc(obs)

    memory.add(prev_obs, action, reward, obs, terminal)
    #if terminal: 
      #print(error, obs[1], action)
    print(round(error,4), round(obs[1],4), round(action,4), len(memory))
    time.sleep(0.001)
  '''    
  if terminal:
    print('end!')
    break
    #agent.end(obs, reward)
  else:
    pass
    #action = control.smc(obs)
  print(control.get_lim_angle(obs))
  '''
