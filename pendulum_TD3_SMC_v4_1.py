import sys, time, os
from combine_control_3 import SMCControl, Memory, TD3
import numpy as np 
from datetime import date, datetime
import matplotlib.pyplot as plt
from beepy import beep
import tensorflow as tf
import simulation_lib as sim

# Assumes this script is being run from grl/bin
sys.path.append('/home/adalberto/Documentos/Reinforcement_Learning/grl/build')

import grlpy

"""
            System initialization
"""

# Environment configurations
envconf = grlpy.Configurator("/home/adalberto/Documentos/Reinforcement_Learning/grl/cfg/matlab/pendulum_swingup.yaml")
envinst = envconf.instantiate()
env = grlpy.Environment(envinst["environment"])

# simulation control variables
path = 'paper/'
log_file = True
obs_steps = 1000
training_steps = 10
epochs = 20
eval_steps = 1
log_file = True
average = []
agent_eval = 0
reward = np.array([])
eval_lim = -1100
average_lim = -1400

# Defining Hyperparameters
gamma = 0.98
batch_size = 32
e_greedy = True
interval = 50
actor_update = 2
initial_noise_e = 1
final_noise_e = 0.01
noise_t = 0.1 # target noise
noise_e = initial_noise_e
hyper = [
            gamma,
            batch_size,
            interval,
            actor_update,  
            noise_t,
            noise_e,
            initial_noise_e,
            final_noise_e,
            epochs,
            e_greedy
        ]

# Control parameters
alpha = [4.5,4.5]
rho = [2.85,1]
K = [0.3185,0.015]
max_torque = 3

control = SMCControl(alpha=alpha, rho=rho, K=K, max_torque=max_torque)

# Agent parameters
num_states = 3
num_actions = 3
hidden = [400,400]
act_out_layer = 'liear'
tau = 0.5

# Creating agent and target networks
agent_name = 'TD3_SMC_Agent_v42_0'#+str(test)
target_agent_name = 'Agent_target'

agent = TD3(num_states ,num_actions, hiddens=hidden, act_out_layer=act_out_layer,
             name=agent_name,tau=tau)

target_agent = TD3(num_states, num_actions, hiddens=hidden, act_out_layer=act_out_layer, 
             name=target_agent_name)

replay_memory = Memory(num_states,num_actions)


"""
        Initialization Checkout
"""
# Creating log file
if log_file:
    sim.log_trainer(agent_name,path=path,act='w')
    sim.log_evaluater(agent_name,path=path,act='w')
    '''
    header = 'Position '\
             'Velocity '\
             'Control '\
             'Reward'
    sim.log(agent_name,header,path=path)
    '''
os.system('clear')
print('Initialization is Done! \nLog file '+agent_name+'.log successfuly created!')
time.sleep(1)
os.system('clear')


"""
            Main loop
"""

# Generating observation
print('Creating the initial replay memory...')
#replay_memory = sim.observer(env, control, replay_memory, obs_steps)
replay_memory = sim.observer_actor(env, agent,control, replay_memory, obs_steps,gain_lim)

# Performing training
for i in range(epochs):
    print('Training Epoch:',i)

    # Training the agent during 10 times 
    agent, target_agent, control, replay_memory, curve, hyper[5] = sim.trainer(env, agent, target_agent, 
                                                                        control,training_steps, 
                                                                        replay_memory, hyper, gain_lim, 
                                                                        path, log_file)
    
    # saving the model
    agent.save_model(name=agent_name, path=path)
    
    # getting the accumulated rewards
    reward = np.concatenate((reward,curve))

    # Evaluating the agent
    agent_eval = sim.evaluator(env, agent, control,log_file=log_file)
    average.append(agent_eval)
    if len(average) < 5:
        mov_average = np.average(average)
    else:
        mov_average = np.average(average[len(average)-5:len(average)])
    print('\nLast eval reward:',agent_eval,'\tMoving Eval average',mov_average)
    
    # Plotting results
    sim.plot_reward(reward, path, agent_name)
    sim.plot_average(average,path, agent_name)   
    
    '''
    # Stops training if the agent learned enough    
    if agent_eval >= eval_lim and mov_average > average_lim:
        break

    '''

#beep(sound=1)