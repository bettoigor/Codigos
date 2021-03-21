import sys, time, os
from combine_control_3 import SMCControl, Memory, TD3
import numpy as np 
from datetime import date, datetime
import matplotlib.pyplot as plt
from beepy import beep
import tensorflow as tf


# Assumes this script is being run from grl/bin
sys.path.append('/home/adalberto/Documentos/Reinforcement_Learning/grl/build')

import grlpy


def get_timedate():

    today = datetime.now()
    at_now = today.strftime("%d-%m-%Y %H:%M:%S:")
    timedate = at_now

    return timedate


def log(file_name, text,path=None):
    if path is None:
      path = ''

    fout = open(path+"log/log_TD3_SMC_training_" + file_name + ".csv", "a")
    fout.write(text+'\n')
    fout.close()




# Environment configurations
envconf = grlpy.Configurator("/home/adalberto/Documentos/Reinforcement_Learning/grl/cfg/matlab/pendulum_swingup.yaml")
envinst = envconf.instantiate()
env = grlpy.Environment(envinst["environment"])

# Initializing the control object
alpha = [4.5,4.5]
rho = [2.85,1]
K = [0.3185,0.015]


control = SMCControl(alpha=alpha,rho=rho, K=K)

# Initializing RL objects
num_states = 3
num_actions = 3
max_torque = 3
hidden = [400,400]


# Fixed Hyperparameters
gamma = 0.98
batch_size = 32
e_greedy = True
epochs = 200
interval = 100
actor_update = 2
initial_noise_e = 0.3
final_noise_e = 0.1
noise_t = 0.1 # target noise
tau = 0.001
decay = 0.97
lim_alpha = 1 
lim_rho = 1
lim_K = 1

# simulation control variables
path = 'paper/'
log_file = True

#agent.save_model(name=agent_name, path=path, save_critic=False)
os.system('clear')
print('Initialization is Done!\n')
time.sleep(1)
os.system('clear')
 
# Creating automatizated test
num_tests = 1

for test in range(0,num_tests):
  # variable hyperparamters
  noise_e = initial_noise_e   # exploration noise
  prev_curve = 0
  curve = np.zeros(epochs)
  gains = [0.,0.,0.]
  n = 0

  # creating agent and target networks
  agent_name = 'TD3_SMC_Agent_v3_1'#+str(test)
  target_agent_name = 'Agent_target'

  agent = TD3(num_states,num_actions, hiddens=hidden, act_out_layer='linear',
                 load_agent=False, name=agent_name,tau=tau)

  target_agent = TD3(num_states,num_actions, hiddens=hidden, act_out_layer='linear', 
                 load_agent=False, name=target_agent_name)

  memory = Memory(num_states,num_actions)


  # log file
  if log_file:
    log(agent_name,get_timedate()+' Start!',path=path)
    header = 'Date '\
             'Time '\
             'Episode '\
             'Position '\
             'Velocity '\
             'Control '\
             'Epsilon'
    log(agent_name,header,path=path)



  # Work Loop
  for i in range(epochs):
      # restarting simulation
      terminal = 0
      raw_obs = env.start(0)
      ctrl_action = control.start(raw_obs)
      norm_obs = control.get_normalized_state(raw_obs)
      
      # run epsodes
      while not terminal:
          # epsode counter
          n+=1

          # saving previous observation
          prev_obs = norm_obs

          # applying the controller action in the environment
          (raw_obs, reward, terminal) = env.step([ctrl_action])
          norm_obs = control.get_normalized_state(raw_obs)


          # getting adjustment action from the Actor agent
          adj_action = abs(agent.actor(norm_obs, noise_e))
          '''
          gains = [adj_action[0]*lim_alpha,
                   adj_action[1]*lim_rho,
                   adj_action[2]*lim_K]     
          '''          
          gains = [abs(adj_action[0]),
                    abs(adj_action[1]),
                    abs(adj_action[2])]     
          
          if n%actor_update == 0:
            control.set_gain(gains)
          
          # getting crontroller action
          ctrl_action, error = control.smc_regular(raw_obs)
                              
          
          if len(memory) <= 1000:
              adj_action = [control.alpha[0],
                            control.rho[0],
                            control.K[0]]
          

          # save transition
          memory.add(prev_obs, adj_action, reward, norm_obs, terminal)
          curve[i] += reward

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

          
          print(chr(27) + "[2J") 
          print(' ********** Training '+str(test)+' **************',
                '\n *'
                '\n * Momdel Name:',agent_name,
                '\n * Episode:',i,'of ',epochs,
                '\n * Position:',round(control.get_lim_angle(raw_obs),4),
                '\n * Velocity:',round(raw_obs[1],4),
                '\n * Action:',ctrl_action.numpy(),
                '\n * Adj Action:',adj_action,        
                '\n * alpha:',round(control.alpha[0],5),
                '\n * rho:',round(control.rho[0],5),
                '\n * K:',round(control.K[0],5),              
                '\n * Error:',round(error,5),
                '\n * Current Reward:',round(reward,3),
                '\n * Accumulated Reward:',round(curve[i],3),
                '\n * Previous Accumulated Reward:',round(prev_curve,3),
                '\n * Memory Size:',len(memory),
                '\n * Current Exploration Noise:',round(noise_e,5),
                '\n **********************************')
      
      # E-greedy reduce
      if e_greedy and noise_e > final_noise_e:
          noise_e -= (initial_noise_e - final_noise_e)/(epochs)
          #noise_e = max(final_noise_e, noise_e*decay)
      # save log
      text = get_timedate()+\
              ','+str(i)+\
              ','+str(control.get_lim_angle(raw_obs))+\
              ','+str(raw_obs[1])+\
              ','+str(ctrl_action.numpy())+\
              ','+str(noise_e)
      if log_file:
        log(agent_name,text,path=path)

      prev_curve = curve[i]
      
      #ploting
      fig = plt.figure()
      plt.plot(curve)
      plt.grid()
      fig.suptitle('Received Reward - '+agent_name, fontsize=16)
      plt.xlabel('Epochs', fontsize=14)
      plt.ylabel('Reward', fontsize=14)
      plt.savefig(path+'images/TD3_out_'+agent_name+'.png')

      if i%10==0:
          agent.save_model(name=agent_name, path=path)
  
  beep(sound=4)

  #plt.pause(1)
  plt.close('all')
  log(agent_name,get_timedate()+' Stop!',path=path)
  fig = plt.figure()
  plt.plot(curve)
  plt.grid()
  fig.suptitle('Received Reward - '+agent_name, fontsize=16)
  plt.xlabel('Epochs', fontsize=14)
  plt.ylabel('Reward', fontsize=14)
  plt.savefig(path+'images/TD3_out_'+agent_name+'.png')
  plt.show(block=False)
  time.sleep(0.1)
  plt.close('all')

beep(sound=1)
