import sys, time
from combine_control_2 import Control, Memory, SmartTD3, SmartControl
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

num_states = 3
num_actions = 1
memory = Memory(num_states,1)
agent_name = 'Agent_10_SmartTD3_1'
target_agent_name = 'Agent_target'
max_torque = 3
hidden = [50,50]

agent = SmartTD3(num_states,num_actions, hiddens=hidden)

target_agent = SmartTD3(num_states,num_actions, hiddens=hidden)


# Hyperparameters
gamma = 0.98
batch_size = 32
noise_e = 0.2
noise_t = 0.1
epochs = 600
interval = 200
n = 0
actor_update = 4
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

print('Done!\n')


for i in range(epochs):
  terminal = 0

  # restart environment and agent
  raw_obs = env.start(0)
  obs = control.get_normalized_state(raw_obs)


  #run episodes
  while not terminal:
    n+=1

    #save previous observation
    prev_obs = obs

    # get action with noise added
    action = agent.actor(raw_obs, noise=None)

    # apply action and receves reward 
    (raw_obs, reward, terminal) = env.step([action.numpy()])
    obs = control.get_normalized_state(raw_obs)
    curve[i] += reward

    # save transition
    memory.add(prev_obs, action.numpy(), reward, obs, terminal)
    
    # Training the agent
    if len(memory) > 1000:
      # memory sampling
      bs, ba, br, bsp, bd = memory.sample(batch_size)

      # creating targets
      qsp = target_agent.critic(s=bsp, noise=None) #  adding noise to the action
      target = br + (1 - bd)*gamma*qsp
      
      # training the agent (actor and critic)
      agent.train(bs,ba,target,n%actor_update == 0)
      
      # updating target networks (Time delayed)
      if n%interval==0:
        target_agent <<= agent
      

    print(chr(27) + "[2J") 
    print(' ********** Training **************',
          '\n *'
          '\n * Episode:',i,'of ',epochs,
          '\n * Position:',round(control.get_lim_angle(raw_obs),4),
          '\n * Velocity:',round(raw_obs[1],4),
          '\n * Action:',action.numpy(),          
          '\n * Alpha:',agent.alpha[0].numpy(),          
          '\n * K:',agent.K[0].numpy(),          
          '\n * rho:',agent.rho[0].numpy(),          
          '\n * Current Reward:',round(reward,3),
          '\n * Accumulaterd Reward:',round(curve[i],3),
          '\n * Previous Accumulated Reward:',round(prev_curve,3),
          '\n * Memory Size:',len(memory),
          '\n **********************************')


  if False:
    agent.save_model(agent_name)
  
  # save log
  text = get_timedate()+\
         ' '+str(i)+\
         ' '+str(control.get_lim_angle(raw_obs))+\
         ' '+str(obs[1 ])+\
         ' '+str(curve[i])
  #log(agent_name,text)
  prev_curve = curve[i]
  print('\nEpisode ',i,' terminated!')
  
  #ploting
  fig = plt.figure()
  plt.plot(curve)
  plt.grid()
  fig.suptitle('Reward Received - TD3', fontsize=18)
  plt.xlabel('Epochs', fontsize=14)
  plt.ylabel('Reward', fontsize=14)
  plt.savefig('images/TD3_out_'+agent_name+'.png')
#plt.pause(1)
plt.close('all')
#time.sleep(0.5)  
log(agent_name,get_timedate()+' Stop!')
fig = plt.figure()
plt.plot(curve)
plt.grid()
fig.suptitle('Reward Received - TD3', fontsize=18)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Reward', fontsize=14)
plt.savefig('images/TD3_out_'+agent_name+'.png')
plt.show(block=True)

