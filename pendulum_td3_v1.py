import sys, time#, os
from combine_control_2 import SMCControl, Memory, TD3
import numpy as np 
from datetime import date, datetime
import matplotlib.pyplot as plt


# Assume this script is being run from grl/bin
sys.path.append('/home/adalberto/Documentos/Reinforcement_Learning/grl/build')

import grlpy

import grlpy


def get_timedate():

    today = datetime.now()
    at_now = today.strftime("%d-%m-%Y %H:%M:%S:")
    timedate = at_now

    return timedate


def log(file_name, text,path=None):

    fout = open(path+"log/log_TD3_training" + file_name + ".csv", "a")
    fout.write(text+'\n')
    fout.close()



# Loading configurations
envconf = grlpy.Configurator("/home/adalberto/Documentos/Reinforcement_Learning/grl/cfg/matlab/pendulum_swingup.yaml")

# Instantiating configurations (constructing objects)
envinst = envconf.instantiate()
env = grlpy.Environment(envinst["environment"])
control = SMCControl(alpha=[4.5,4.5],rho=[2.285,1], K=[0.2185,0.015])
num_states = 3
num_actions = 1
path = 'paper/'
max_torque = 3
hidden = [400,400]

# Hyperparameters
gamma = 0.98
batch_size = 32

epochs = 200
interval = 100
actor_update = 2
prev_curve = 0

# Creating automatizated testset
num_tests = 10

for test in range(0,num_tests):
  # variable hyperparameters
  noise_e = 0.1
  noise_t = 0.1
  curve = np.zeros(epochs)
  n = 0

  # Agente and Target instatiation
  agent_name = 'Agent_auto_'+str(test)
  target_agent_name = 'Agent_target'
  agent = TD3(num_states,num_actions, hiddens=hidden,
                 load_agent=False, name=agent_name, save_critic=False)

  target_agent = TD3(num_states,num_actions, hiddens=hidden, 
                 load_agent=False, name=target_agent_name, save_critic=False)
  
  memory = Memory(num_states,1)


  # log file
  log(agent_name,get_timedate()+' Start!',path=path)
  header = 'Date '\
           'Time '\
           'Episode '\
           'Position '\
           'Velocity '\
           'Reward '
  log(agent_name,header,path=path)
  

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
      action = agent.actor(obs, noise_e)

      # apply action and receves reward 
      (raw_obs, reward, terminal) = env.step([action * max_torque])
      obs = control.get_normalized_state(raw_obs)
      curve[i] += reward

      # save transition
      memory.add(prev_obs, action, reward, obs, terminal)
      
      # Training the agent
      if len(memory) > 1000:
        # memory sampling
        bs, ba, br, bsp, bd = memory.sample(batch_size)

        # creating targets
        qsp = target_agent.critic(s=bsp, noise=noise_t) #  adding noise to the action
        target = br + (1 - bd)*gamma*qsp
        
        # training the agent (actor and critic)
        agent.train(bs,ba,target,n%actor_update == 0)
        
        # updating target networks (Time delayed)
        if n%interval==0:
          target_agent <<= agent
        


      #os.system('clear')
      print(chr(27) + "[2J") 
      print(' ********** Training #'+str(test)+' **************',
            '\n *'
            '\n * Episode:',i,'of ',epochs,
            '\n * Position:',round(control.get_lim_angle(raw_obs),4),
            '\n * Velocity:',round(raw_obs[1],4),
            '\n * Action:',action*max_torque,
            '\n * > 3',abs(action*max_torque) > 3,
            '\n * Current Reward:',round(reward,3),
            '\n * Accumulaterd Reward:',round(curve[i],3),
            '\n * Previous Accumulated Reward:',round(prev_curve,3),
            '\n * Memory Size:',len(memory),
            '\n *********************************************')
    
    
    # save log
    text = get_timedate()+\
           ','+str(i)+\
           ','+str(control.get_lim_angle(raw_obs))+\
           ','+str(raw_obs[1])+\
           ','+str(curve[i])
    log(agent_name,text,path=path)
    prev_curve = curve[i]
    print('\nEpisode ',i,' terminated!')
    #time.sleep(0.5)

    if i%10==0:
      agent.save_model(name=agent_name, path=path)

    #ploting
    fig = plt.figure()
    plt.plot(curve)
    plt.grid()
    fig.suptitle('Reward Received - TD3 '+agent_name, fontsize=18)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.savefig(path+'images/TD3_out_'+agent_name+'.png')
  
  plt.close('all')
  log(agent_name,get_timedate()+' Stop!',path=path)
  fig = plt.figure()
  plt.plot(curve)
  plt.grid()
  fig.suptitle('Reward Received - TD3 '+agent_name, fontsize=18)
  plt.xlabel('Epochs', fontsize=14)
  plt.ylabel('Reward', fontsize=14)
  plt.savefig(path+'images/TD3_out_'+agent_name+'.png')
  plt.show(block=False)
  time.sleep(0.5)
  plt.close('all')