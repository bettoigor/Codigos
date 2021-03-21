import sys, time, os
from combine_control_2 import SMCControl, Memory, SmartTD3, SmartControl, TD3
import numpy as np 
from math import pi
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

    fout = open("log/log_smartTD3_SMC_Try_" + file_name + ".csv", "a")
    fout.write(text+'\n')
    fout.close()



# Environment configurations
envconf = grlpy.Configurator("/home/adalberto/Documentos/Reinforcement_Learning/grl/cfg/matlab/pendulum_swingup_viewer.yaml")
envinst = envconf.instantiate()
env = grlpy.Environment(envinst["environment"])

# Initializing the control object
# Cabalistic values
#'''
alpha = [0.,4.5]
rho = [0.,1]
K = [0.,0.015]
'''
alpha = [1.,4.5]
rho = [1.,1]
K = [1.,0.015]
'''
a = 1.
r = 1.
k = 1.

control = SMCControl(alpha=alpha,rho=rho, K=K,setpoint=0)

# Initializing RL objects
num_states = 3
num_actions = 3
agent_name = 'Agent_SmartTD3_02_Special_SMC'
agent = TD3(num_states,num_actions, load_agent=True, name=agent_name)

# Hyperparameters
gains = [0.,0.,0.]
epochs = 2
curve = np.zeros(epochs)


# log file
header = 'Date '\
         'Time '\
         'Episode '\
         'Position '\
         'Velocity '\
         'Control '\
         'Alpha '\
         'Rho '\
         'K '
log(agent_name,header)



# Work Loop
for i in range(epochs):
    # restarting simulation
    terminal = 0
    raw_obs = env.start(0)
    ctrl_action = control.start(raw_obs)
    norm_obs = control.get_normalized_state(raw_obs)
    
    # run epsodes
    while not terminal:

        # applying the controller action in the environment
        (raw_obs, reward, terminal) = env.step([ctrl_action])
        norm_obs = control.get_normalized_state(raw_obs)
        curve[i] += reward

        # getting crontroller action
        ctrl_action, error = control.smc_special(raw_obs)

        # getting adjustment action from the Actor agent
        adj_action = agent.actor(norm_obs, noise=None)
        gains = [abs(adj_action[0]),
                    abs(adj_action[1]),
                    abs(adj_action[2])]
        control.set_specialGain(gains)

        #print(chr(27) + "[2J") 
        os.system('clear')
        print(' ********** Trying **************',
              '\n *'
              '\n * Momdel Name:',agent_name,
              '\n * Episode:',i,'of ',epochs,
              '\n * Position:',round(control.get_lim_angle(raw_obs),4),
              '\n * Velocity:',round(raw_obs[1],4),
              '\n * Action:',ctrl_action.numpy(),
              '\n * Error:',round(error,5),
              '\n * Adj Action:',adj_action,        
              '\n * alpha:',round(control.alpha[0],5),
              '\n * rho:',round(control.rho[0],5),
              '\n * K:',round(control.K[0],5),              
              '\n * Current Reward:',round(reward,3),
              '\n * Accumulaterd Reward:',round(curve[i],3),
              '\n **********************************')
        # save log
        text = get_timedate()+\
             ' '+str(i)+\
             ' '+str(control.get_lim_angle(raw_obs))+\
             ' '+str(raw_obs[1])+\
             ' '+str(ctrl_action.numpy())+\
             ' '+str(control.alpha[0])+\
             ' '+str(control.rho[0])+\
             ' '+str(control.K[0])
        log(agent_name,text)

    print('Done!')
    time.sleep(0.5)
    

