import sys, time, os
from combine_control_2 import SMCControl, Memory, SmartTD3, SmartControl, TD3
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

# Environment configurations
envconf = grlpy.Configurator("/home/adalberto/Documentos/Reinforcement_Learning/grl/cfg/matlab/pendulum_swingup.yaml")
envinst = envconf.instantiate()
env = grlpy.Environment(envinst["environment"])

# Initializing the control object
#'''
alpha = [4.5,4.5]
rho = [2.85,2.85]
K = [0.3185,0.3185]

# Cabalistic values
#a = 5.
#r = 3.
#k = 1.

a = 1.
r = 1.
k = 1.


control = SMCControl(alpha=alpha,rho=rho, K=K)

# Initializing RL objects
num_states = 3
num_actions = 3
memory = Memory(num_states,num_actions)
agent_name = 'Agent_SmartTD3_02_Special_SMC'
target_agent_name = 'Agent_target'
max_torque = 3
hidden = [400,400]
#load_agent = input('Type YES if you want to create a new agent:')
#load_agent = False if load_agent == 'YES' else True
agent = TD3(num_states,num_actions, hiddens=hidden,
               load_agent=False, name=agent_name)

target_agent = TD3(num_states,num_actions, hiddens=hidden, 
               load_agent=False, name=target_agent_name)


# Hyperparameters
gamma = 0.98
batch_size = 32
noise_e = 0.3
initial_noise_e = noise_e
final_noise_e = 0.1
noise_t = 0.1
e_greedy = True
epochs = 200
interval = 200
n = 0
actor_update = 2
prev_curve = 0
curve = np.zeros(epochs)
gains = [0.,0.,0.]

# log file
log(agent_name,get_timedate()+' Start!')
header = 'Date '\
         'Time '\
         'Episode '\
         'Position '\
         'Velocity '\
         'Control '
log(agent_name,header)

os.system('clear')
print('Initialization is Done!\n')
time.sleep(1)
os.system('clear')


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
        
        # getting crontroller action
        ctrl_action, error = control.smc_special(raw_obs)

        # getting adjustment action from the Actor agent
        adj_action = agent.actor(norm_obs, noise_e)
        
        if len(memory) <= 1000:
            adj_action = [control.alpha[1],control.rho[1],control.K[1]]

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
          gains = [abs(adj_action[0]),
                    abs(adj_action[1]),
                    abs(adj_action[2])]
          
          if n%actor_update == 0:
            control.set_specialGain(gains)
          

          # updating target networks (Time delayed)
          if n%interval==0:
            target_agent <<= agent

        print(chr(27) + "[2J") 
        print(' ********** Training **************',
              '\n *'
              '\n * Episode:',i,'of ',epochs,
              '\n * Position:',round(control.get_lim_angle(raw_obs),4),
              '\n * Velocity:',round(raw_obs[1],4),
              '\n * Action:',ctrl_action.numpy(),
              '\n * Adj Action:',adj_action,        
              '\n * alpha:',round(control.alpha[1],5),
              '\n * Error:',round(error,5),
              '\n * rho:',round(control.rho[1],5),
              '\n * K:',round(control.K[1],5),              
              '\n * Current Reward:',round(reward,3),
              '\n * Accumulaterd Reward:',round(curve[i],3),
              '\n * Previous Accumulated Reward:',round(prev_curve,3),
              '\n * Memory Size:',len(memory),
              '\n **********************************')
    
    # E-greedy reduce
    if e_greedy and noise_e > final_noise_e:
        noise_e -= (initial_noise_e - final_noise_e)/(epochs*2)

    # save log
    text = get_timedate()+\
         ' '+str(i)+\
         ' '+str(control.get_lim_angle(raw_obs))+\
         ' '+str(norm_obs[1 ])+\
         ' '+str(ctrl_action.numpy())
    log(agent_name,text)

    prev_curve = curve[i]
    #ploting
    fig = plt.figure()
    plt.plot(curve)
    plt.grid()
    fig.suptitle('Received Reward - SmartTD3'+agent_name, fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.savefig('images/TD3_out_'+agent_name+'.png')
    #plt.show(block=False)

    if i%10==0:
        agent.save_model(agent_name)

#plt.pause(1)
plt.close('all')
#time.sleep(0.5)
log(agent_name,get_timedate()+' Stop!')
fig = plt.figure()
plt.plot(curve)
plt.grid()
fig.suptitle('Received Reward - SmartTD3'+agent_name, fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Reward', fontsize=14)
plt.savefig('images/TD3_out_'+agent_name+'.png')
plt.show(block=True)
