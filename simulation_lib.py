import sys, time, os
from combine_control_3 import SMCControl, Memory, TD3
import numpy as np 
from datetime import date, datetime
import matplotlib.pyplot as plt
from beepy import beep
import tensorflow as tf


def get_timedate():

    today = datetime.now()
    at_now = today.strftime("%d-%m-%Y %H:%M:%S:")
    timedate = at_now

    return timedate

def log_trainer(file_name, text=None, path=None, act='a'):
    if path is None:
      path = ''

    fout = open(path+"log/log_TD3_SMC_training_" + file_name + ".csv", act)
    if text is not None:
        fout.write(text+'\n')
    fout.close()

def log_evaluater(file_name, text=None, path=None, act='a'):
    if path is None:
      path = ''

    fout = open(path+"log/log_TD3_SMC_evaluation_" + file_name + ".csv", act)
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
    fig.suptitle('Training Reward - TD3+SMC '+agent_name, fontsize=18)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.savefig(path+'images/TD3_SMC_training'+agent_name+'.png')

def plot_average(average, path=None, agent_name='generic'):
    if path is None:
        path = ''
    
    #ploting
    fig = plt.figure()
    plt.plot(average)
    plt.grid()
    fig.suptitle('Reward at Evaluation - TD3+SMC '+agent_name, fontsize=18)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.savefig(path+'images/TD3_SMC_evaluation'+agent_name+'.png')

def observer(env, control, replay_memory, obs_steps):
    """
        Filling the replay memory with transtions using initial control
        gains.
    """
    
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

             # getting crontroller action
            ctrl_action, error = control.smc_regular(raw_obs)

            # getting adj action
            adj_action = [control.alpha[0],
                        control.rho[0],
                        control.K[0]]

            # save transition
            replay_memory.add(prev_obs, adj_action, reward, norm_obs, terminal)

    return replay_memory

def observer_actor(env, agent, control, replay_memory, obs_steps, gain_lim):
    """
        Filling the replay memory with transtions using initial control
        gains.
    """
    
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
            adj_action = agent.actor(norm_obs, noise=None)

            gains = [abs(control.alpha[0]),
                          abs(control.rho[0]),
                          abs(control.K[0])]

            # Applying adjust action
            control.set_gain(gains)

            # getting crontroller action
            ctrl_action, error = control.smc_regular(raw_obs)

            # save transition
            replay_memory.add(prev_obs, adj_action, reward, norm_obs, terminal)

    return replay_memory

def evaluator(env, agent, control, noisy=None, log_file=False, path=None):

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

        # getting crontroller action
        ctrl_action, error = control.smc_regular(raw_obs)
        
        text = str(control.get_lim_angle(raw_obs))+\
                   ','+str(raw_obs[1])+\
                   ','+str(ctrl_action.numpy())+\
                   ','+str(curve)

        if log_file:
            log_evaluater(agent.name,text=str(curve),path=path)

    return curve

def trainer(env, agent, target_agent, control, training_steps,replay_memory, hyperparams, path=None, log_file=False):
    
    # recovering the hyperparameters
    gamma = hyperparams[0]
    batch_size = hyperparams[1]
    interval = hyperparams[2]
    actor_update = hyperparams[3]
    noise_t = hyperparams[4]
    noise_e = hyperparams[5]
    initial_noise_e = hyperparams[6]
    final_noise_e = hyperparams[7]
    epochs = hyperparams[8]
    e_greedy = hyperparams[9]
    

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
            #if n%actor_update == 0:
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
                   ','+str(curve[i])
            print(text, adj_action)
            log_trainer(agent.name,text,path=path)

        # E-greedy reduce
        if e_greedy and noise_e > final_noise_e:
            noise_e -= (initial_noise_e - final_noise_e)/(epochs)
            



    return agent, target_agent, control, replay_memory, curve, noise_e

