import sys, time, os
from combine_control import Control, Memory,DDPG
import numpy as np 
from datetime import date, datetime
import matplotlib.pyplot as plt

# Assumes this script is being run from grl/bin
sys.path.append('/home/adalberto/Documentos/Reinforcement_Learning/grl/build')

import grlpy



def get_timedate():

    today = datetime.now()
    at_now = today.strftime("%d-%m-%Y %H:%M:%S:")
    timedate = at_now

    return timedate


def log(file_name, text):

    fout = open("log/log_DDPG_" + file_name + ".csv", "a")
    fout.write(text+'\n')
    fout.close()



# Load configurations
envconf = grlpy.Configurator("/home/adalberto/Documentos/Reinforcement_Learning/grl/cfg/matlab/pendulum_swingup.yaml")

# Instantiate configurations (construct objects)
envinst = envconf.instantiate()
env = grlpy.Environment(envinst["environment"])
control = Control(alpha=[4.5,4.5],rho=[2.285,1], K=[0.2185,0.015])
memory = Memory(3,1)
agent_name = 'Agent_06'
max_torque = 3
agent = DDPG(3,1, max_action=max_torque, hiddens=[15,15], name=agent_name)
#agent = DDPG(3,1, hiddens=[35,35], name=agent_name)

# Hyperparameters
gamma = 0.96
batch_size = 32
epochs = 100

prev_curve = 0
curve = np.zeros(epochs)

# log file
log(agent_name,get_timedate()+' Start!')
header = 'Date '\
         'Time '\
         'Episode '\
         'Position '\
         'Velocity '\
         'Reward '
log(agent_name,header)


for i in range(epochs):
  terminal = 0

  # restart environment and agent
  raw_obs = env.start(0)
  obs = control.get_normalized_state(raw_obs)

  #run episodes
  while not terminal:
    #save previous observation
    prev_obs = obs

    # get action
    action = agent.actor(obs)

    # apply action and receves reward 
    (raw_obs, reward, terminal) = env.step([action])
    obs = control.get_normalized_state(raw_obs)
    curve[i] += reward

    # save transition
    memory.add(prev_obs, action, reward, obs, terminal)
    
    # Training the agent
    if len(memory) > 500:
      # memory sampling
      bs, ba, br, bsp, bd = memory.sample(batch_size)

      # creating targets
      qsp = agent.critic(bsp)
      target = br + (1 - bd)*gamma*qsp
      
      # training the agent (actor and critic)
      agent.train(bs,ba,target)
    
    os.system('clear')
    print(' ********** Training **************',
          '\n *'
          '\n * Episode:',i,'of ',epochs,
          '\n * Position:',round(control.get_lim_angle(raw_obs),4),
          '\n * Velocity:',round(obs[1],4),
          '\n * Action:',action,
          '\n * Current Reward:',round(reward,3),
          '\n * Accumulaterd Reward:',round(curve[i],3),
          '\n * Previous Accumulated Reward:',round(prev_curve,3),
          '\n **********************************')
  agent.save_model(agent_name)
  
  # save log
  text = get_timedate()+\
         ' '+str(i)+\
         ' '+str(control.get_lim_angle(raw_obs))+\
         ' '+str(obs[2])+\
         ' '+str(curve[i])
  log(agent_name,text)
  prev_curve = curve[i]
  print('\nEpisode ',i,' terminated!')
  #time.sleep(0.5)

  #ploting
  fig = plt.figure()
  plt.plot(curve)
  plt.grid()
  fig.suptitle('Reward Received - DDGP', fontsize=18)
  plt.xlabel('Epochs', fontsize=14)
  plt.ylabel('Reward', fontsize=14)
  plt.savefig('images/DDPG_out_'+agent_name+'.png')
  plt.show(block=False)
  #plt.pause(1)
  plt.close('all')
  #time.sleep(0.5)

log(agent_name,get_timedate()+' Stop!')
fig = plt.figure()
plt.plot(curve)
plt.grid()
fig.suptitle('Reward Received - DDGP', fontsize=18)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Reward', fontsize=14)
plt.savefig('images/DDPG_out_'+agent_name+'.png')
plt.show(block=True)
