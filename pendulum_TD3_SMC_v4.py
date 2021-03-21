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
            Service fuctions
"""

def get_timedate():

    today = datetime.now()
    at_now = today.strftime("%d-%m-%Y %H:%M:%S:")
    timedate = at_now

    return timedate


def log(file_name, text=None, path=None, act='a'):
    if path is None:
      path = ''

    fout = open(path+"log/log_TD3_SMC_training_" + file_name + ".csv", act)
    if text is not None:
        fout.write(text+'\n')
    fout.close()


def plot_reward(reward, path=None, agent_name='generic'):
    if path is None:
        path = ''
    
    #ploting
    fig = plt.figure()
    plt.plot(reward)
    plt.grid()
    fig.suptitle('Reward Received - TD3 '+agent_name, fontsize=18)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.savefig(path+'images/TD3_out_'+agent_name+'.png')


def observer(replay_memory, obs_steps):
    """
        Filling the replay memory with transtions using initial control
        gains.
    """

    global env
    global control

    
    while len(replay_memory) < obs_steps:

        # restarting simulation
        terminal = 0
        raw_obs = env.start(0)
        ctrl_action = control.start(raw_obs)
        norm_obs = control.get_normalized_state(raw_obs)

        while not terminal:

            # saving previous observation
            prev_obs = norm_obs

            # applying the controller action in the environment
            (raw_obs, reward, terminal) = env.step([ctrl_action])
            norm_obs = control.get_normalized_state(raw_obs)

            # getting adj action
            adj_action = [control.alpha[0],
                        control.rho[0],
                        control.K[0]]

            # save transition
            replay_memory.add(prev_obs, adj_action, reward, norm_obs, terminal)

    return replay_memory


def evaluator(noisy=None):

    global agent
    global control
    global env

    # Initializing evaluation parameters
    curve = 0
    count = 0
    terminal = 0
    raw_obs = env.start(0)
    ctrl_action = control.start(raw_obs)
    norm_obs = control.get_normalized_state(raw_obs)
    
    print('Evaluating agent...')
    while not terminal:
        # applying the controller action in the environment
        (raw_obs, reward, terminal) = env.step([ctrl_action])
        norm_obs = control.get_normalized_state(raw_obs)
        curve += reward
        count+=1

        # getting crontroller action
        ctrl_action, error = control.smc_regular(raw_obs)

        # putting external noise
        if count > 200 and count < 206 and noisy is not None:
            ctrl_action = tf.convert_to_tensor(max_torque, dtype=tf.float32)
        
        if count > 400 and count < 406 and noisy is not None:
            ctrl_action = tf.convert_to_tensor(-max_torque, dtype=tf.float32)    

        # getting adjustment action from the Actor agent
        adj_action = agent.actor(norm_obs, noise=None)
        gains = [abs(adj_action[0]),
                  abs(adj_action[1]),
                  abs(adj_action[2])]
        
        # Applying adjust action
        control.set_gain(gains)
        
        text = str(control.get_lim_angle(raw_obs))+\
                   ','+str(raw_obs[1])+\
                   ','+str(ctrl_action.numpy())+\
                   ','+str(curve)

    return curve


def trainer(training_steps, replay_memory, hyperparams, path=None, log_file=False):
    
    global env
    global agent
    global target_agent
    global control


    # recovering the hyperparameters
    gamma = hyperparams[0]
    batch_size = hyperparams[1]
    interval = hyperparams[2]
    actor_update = hyperparams[3]
    noise_t = hyperparams[4]
    noise_e = hyperparams[5]
    
    # save the accumulated rewards
    curve = np.zeros(training_steps)

    for i in range(training_steps):
        terminal = 0
        raw_obs = env.start(0)
        ctrl_action = control.start(raw_obs)
        norm_obs = control.get_normalized_state(raw_obs)
        n = 0
        
        while not terminal:
            # saving previous observation
            prev_obs = norm_obs

            # applying the controller action in the environment
            (raw_obs, reward, terminal) = env.step([ctrl_action])
            norm_obs = control.get_normalized_state(raw_obs)
            curve[i] += reward

            # getting adjustment action from the Actor agent
            adj_action = agent.actor(norm_obs, noise_e)
            gains = [abs(adj_action[0]),
                     abs(adj_action[1]),
                     abs(adj_action[2])]     
            
            # setting control gains
            if n%actor_update == 0:
                control.set_gain(gains)

            # getting crontroller action
            ctrl_action, error = control.smc_regular(raw_obs)

            # save transition
            replay_memory.add(prev_obs, adj_action, reward, norm_obs, terminal)

            # memory sampling
            bs, ba, br, bsp, bd = replay_memory.sample(batch_size)

            # creating targets
            qsp = target_agent.critic(s=bsp, noise=noise_t) #  adding noise to the action
            target = br + (1 - bd)*gamma*qsp

            # training only the agent
            agent.train(bs,ba,target,actor_update=n%actor_update == 0)

            # updating the target network
            if n%interval==0:
                target_agent <<= agent

            # iteration counter for delayed update
            n+=1 
            
        print('Training... Step '+str(i+1))
        # save log
        if log_file:
            text = str(control.get_lim_angle(raw_obs))+\
                   ','+str(raw_obs[1])+\
                   ','+str(ctrl_action.numpy())+\
                   ','+str(curve[i])+\
                   ','+str(gains)
            print(text)
            log(agent.name,text,path=path)


    return replay_memory, curve




"""
            System initialization
"""

# Environment configurations
envconf = grlpy.Configurator("/home/adalberto/Documentos/Reinforcement_Learning/grl/cfg/matlab/pendulum_swingup_viewer_3s.yaml")
envinst = envconf.instantiate()
env = grlpy.Environment(envinst["environment"])

# simulation control variables
path = 'paper/'
log_file = True
obs_steps = 1000
training_steps = 10
epochs = 200
eval_steps = 1
epochs = 200
log_file = True
averange = []
agent_eval = 0
reward = np.array([])

# Control parameters
alpha = [4.5,4.5]
rho = [2.85,1]
K = [0.3185,0.015]
max_torque = 3

control = SMCControl(alpha=alpha, rho=rho, K=K, max_torque=max_torque)

# Agent parameters
num_states = 3
num_actions = 3
hidden = [400,300]
act_out_layer = 'linear'
tau = 0.005

# Creating agent and target networks
agent_name = 'TD3_SMC_Agent_v4_1'#+str(test)
target_agent_name = 'Agent_target'

agent = TD3(num_states ,num_actions, hiddens=hidden, act_out_layer=act_out_layer,
             name=agent_name,tau=tau)

target_agent = TD3(num_states, num_actions, hiddens=hidden, act_out_layer=act_out_layer, 
             name=target_agent_name)

replay_memory = Memory(num_states,num_actions)

# Defining Hyperparameters
gamma = 0.98
batch_size = 32
e_greedy = True
interval = 10
actor_update = 2
initial_noise_e = 0.3
final_noise_e = 0.1
noise_t = 0.1 # target noise
noise_e = initial_noise_e
hyper = [
            gamma,
            batch_size,
            interval,
            actor_update,
            noise_t,
            noise_e
        ]

"""
        Initialization Checkout
"""
# Creating log file
if log_file:
    log(agent_name,path=path,act='w')
    '''
    header = 'Position '\
             'Velocity '\
             'Control '\
             'Reward'
    sim.log(agent_name,header,path=path)
    '''
os.system('clear')
print('Initialization is Done!\nLog file '+agent_name+'.log successfuly created!')
time.sleep(1)
os.system('clear')


"""
            Main loop
"""

# Generating observation
print('Creating the initial replay memory...')
replay_memory = observer(replay_memory, obs_steps)


# Performing training
for i in range(epochs):
    
    # Training the agent during 10 times 
    replay_memory, curve = trainer(training_steps, replay_memory, 
                                        hyper,path, log_file)
    
    # saving the model
    agent.save_model(name=agent_name, path=path)
    
    # getting the accumulated rewards
    reward = np.concatenate((reward,curve))

    # Evaluating the agent
    agent_eval = evaluator()
    averange.append(agent_eval)
    mov_averange = sum(averange)/len(averange)
    print('Moving Averange:',mov_averange)
    
    # Stops training if the agent learned enough    
    if mov_averange >= -1000:
        break

    # E-greedy reduce
    if e_greedy and noise_e > final_noise_e:
        noise_e -= (initial_noise_e - final_noise_e)/(epochs)
        hyper[5] = noise_e

    plot_reward(reward, path, agent_name)

beep(sound=1)