import sys, time, os
from combine_control import Control, Memory, DQN
import numpy as np 
import matplotlib.pyplot as plt
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
    '''
    Writing in the .csv log file
    '''
    fout = open("log/log_DQN_" + file_name + ".csv", "a")
    fout.write(text+'\n')
    fout.close()


# Load configurations
envconf = grlpy.Configurator("/home/adalberto/Documentos/Reinforcement_Learning/grl/cfg/matlab/pendulum_swingup.yaml")

# Instantiate configurations (construct objects)
envinst = envconf.instantiate()
env = grlpy.Environment(envinst["environment"])
control = Control(alpha=[4.5,4.5],rho=[2.285,2], K=[0.2185,0.015])
memory = Memory(3,1)

# Hyperparameters
gamma = 0.99
epochs = 200
epsilon = 0.1
initial_epsilon = epsilon
final_epsilon = 0.001
e_greedy = True
torques = [-3,0,3]
batch_size = 32
interval = 200
n = 0
curve = np.zeros(epochs)
prev_curve = 0

# Network paramteres
net_name = 'Net_05'
target_name = 'Target_05'
network = DQN(3,1,[35, 35],name=net_name)
target = DQN(3,1,[35, 35],name=target_name)

# log file
log(net_name,get_timedate()+' Start!')
header = 'Date '\
         'Time '\
         'Episode '\
         'Position '\
         'Velocity '\
         'Reward '
log(net_name,header)


for i in range(epochs):
  # restart environment and agent
  terminal = 0
  raw_obs = env.start(0)
  obs = control.get_normalized_state(raw_obs) 

  #run episodes
  while not terminal:
    n+=1
    # get action
    net_obs = network(obs, torques)
    action = torques[np.argmax(net_obs)]
    if np.random.rand() < epsilon:
      action = np.random.choice(torques)

    #save previous observation
    prev_obs = obs

    # apply action and receves reward 
    (raw_obs, reward, terminal) = env.step([action])
    obs = control.get_normalized_state(raw_obs)
    curve[i] += reward
    
    # save transition
    memory.add(prev_obs, action, reward, obs, terminal)
    
    # Training the agent
    if len(memory) > 1000:
      # memory sampling
      bs, ba, br, bsp, bd = memory.sample(batch_size)

      # creating targets
      qsp = np.amax(target(bsp, torques), axis=1, keepdims=True)
      y = br + (1-bd)*gamma*qsp
      network.train(bs, ba, y)

      # training the target network
      if n%interval==0:
        target <<= network

    os.system('clear')
    print(' ********** Training **************',
          '\n *'
          '\n * Episode:',i,'of ',epochs,
          '\n * E-Greedy:',e_greedy,
          '\n * Position:',round(obs[0],4),
          '\n * Velocity:',round(obs[2],4),
          '\n * Action:',action,
          '\n * Current Reward:',round(reward,3),
          '\n * Accumulaterd Reward:',round(curve[i],3),
          '\n * Previous Accumulated Reward:',round(prev_curve,3),
          '\n **********************************')
  

  # E-greedy reduce
  if e_greedy and epsilon > final_epsilon:
    epsilon -= (initial_epsilon - final_epsilon)/epochs

  # save log
  text = get_timedate()+\
         ' '+str(i)+\
         ' '+str(raw_obs[0])+\
         ' '+str(obs[2])+\
         ' '+str(curve[i])
  log(net_name,text)

  prev_curve = curve[i]
  network.save_model(net_name)
  target.save_model(target_name)
  print('Model saved!')

  print('\nEpisode ',i,' terminated!')

  #ploting
  fig = plt.figure()
  plt.plot(curve)
  fig.suptitle('Reward Received', fontsize=20)
  plt.xlabel('Epochs', fontsize=14)
  plt.ylabel('Reward', fontsize=14)
  plt.savefig('images/DQN_out_'+net_name+'.png')
  plt.show(block=False)
  #plt.pause(1)
  plt.close('all')
  #time.sleep(0.5)

log(net_name,get_timedate()+' Stop!')
fig = plt.figure()
plt.plot(curve)
fig.suptitle('Reward Received', fontsize=20)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Reward', fontsize=14)
plt.savefig('images/DQN_out_'+net_name+'.png')
plt.show(block=True)
