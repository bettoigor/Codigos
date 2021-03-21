#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#############################################
#                                           #
# Inverted Pendulum Control with            #
# Reinforcement Learning Algorithm          #
#                                           #
# Running with Python 3 and tensorflow      #
#                                           #
#         *** Training Module ***           #
#                                           #
# Author: Adalberto Oliveira                #
# Doctoral in robotic - PUC-Rio             #
# Version: 5.1.1109                         #
# Date: 9-11-2020                           #
#                                           #
#############################################

import reinforcementlearning as rl
import numpy as np
import tensorflow as tf
import roslib, rospy, random, time, math, csv, collections, os, sys, copy
from math import sin, cos, tan
from datetime import date, datetime
import scipy.stats
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import load_model


def log(text):
	'''
	Writing in the .csv log file
	'''

	global DATA_DIR
	global NETWORK_NAME

	today = datetime.now()
	at_now = today.strftime("%d-%m-%Y %H:%M:%S: ")
	fout = open(DATA_DIR+"/Log/log-" + NETWORK_NAME + "_plot.csv", "a")
	fout.write(at_now+text+'\n')
	fout.close()



# loading network
#path = str(sys.argv[1])
DATA_DIR = os.getcwd()#path
#NETWORK_NAME = 'pendulum_v5_1_0928_3'
NETWORK_NAME = str(sys.argv[1])
print('Local path:'+os.getcwd())

# Trying to load the model
print('Trying to load network in:',DATA_DIR+"/network/rl-network-" + NETWORK_NAME + ".h5")
try:
	model = load_model("network/"+ NETWORK_NAME + ".h5")
	model.compile(optimizer=Adam(lr=1e-4), loss="mse")
	text = 'Network rl-network-' + NETWORK_NAME + '.h5 succesfully loaded!'
	print(text)
	log(text)

except:
	today = datetime.now()
	at_now = today.strftime("%d-%m-%Y %H:%M:%S")
	text = 'Could not load the network from '+DATA_DIR+'/network/rl-network-' + NETWORK_NAME + '.h5\n'
	log(text)
	log('Stop\n')
	sys.exit(at_now+' Can not load network '+ NETWORK_NAME)


# creating state space
phi_t, dphi_t = np.meshgrid(np.linspace(-np.pi,np.pi, 64), np.linspace(-8,8, 64))
obs = np.hstack((np.reshape(np.sin(phi_t), (phi_t.size, 1)),
                 np.reshape(np.cos(phi_t), (phi_t.size, 1)),
                 np.reshape(       dphi_t, (dphi_t.size, 1))))
act = np.array([0,1,2,3]).reshape(4,1)

# creating training batch
rs = (obs.shape[0],act.shape[0])
Q = np.zeros(rs)

for act_ in range(len(act)):
	aa = np.ones((obs.shape[0],1))*act[act_]
	x = np.hstack((obs,aa))
	q = model.predict(x)
	Q[:,act_] = q[:,0]

value_function = np.reshape(np.amax(Q, axis=1), phi_t.shape)
policy = np.vectorize(lambda x: act[x])(np.reshape(np.argmax(Q, axis=1), phi_t.shape))

# creating figure
fig, axs = plt.subplots(1,2)
# value function
h = axs[0].contourf(phi_t, dphi_t, value_function, 256)
fig.colorbar(h, ax=axs[0])
# policy
h = axs[1].contourf(phi_t, dphi_t, policy, 256)
fig.colorbar(h, ax=axs[1])

plt.show()
'''
plt.show(block=False)
plt.pause(0.1)
plt.close('all')
'''
log('Model inference array shape '+str(Q.shape))
log('Stop\n')

#/home/adalberto/catkin_ws/src/inverted_pendulum/scripts/network
