import numpy as np
import tensorflow as tf
import os
from math import pi, cos, sin, tan, atan2, sqrt


class SmartControl(object):
    #def __init__(self, alpha=[4.5,4.5], rho=[6.5,20], K=[1,0.2], max_torque=3, setpoint=0):
    def __init__(self, alpha=[1.,1.], rho=[1.,1.], K=[.1,.1], max_torque=3, 
                    setpoint=0, state_raw=[0.,0.]):


        self.alpha = tf.Variable(alpha, name='alpha')
        self.rho = tf.Variable(rho, name='rho')
        self.K = tf.Variable(K, name='K')
        self.max_torque = max_torque
        self.setpoint = setpoint
        self.max_torque = max_torque
        self.state_raw = state_raw
        self.__opt = tf.keras.optimizers.Adam()

        super(SmartControl, self).__init__()
    
    def get_normalized_state(self, state_raw):
        s1 = cos(state_raw[0])
        s2 = sin(state_raw[0])
        s3 = state_raw[1]

        normalized_state = [s1, s2, s3]

        return normalized_state

    def get_raw_state(self, norm_state):
        
        raw_state = np.zeros([len(norm_state),2])
        for i in range(len(norm_state)):
            raw_state[i][0] = atan2(norm_state[i][1],norm_state[i][0])
            raw_state[i][1] = norm_state[i][2]

        return raw_state

    def get_lim_angle(self, state_raw):
        angle = state_raw[0] - pi

        return angle


    def smc(self, state_raw):

        '''
        Receives state position (x, dot_x) and returns the SMC control action
        '''

        # states
        phi = self.get_lim_angle(state_raw)
        dphi = state_raw[1]

        # Defining system states
        x1 = self.setpoint - phi
        x2 = - dphi
        error = x1

                
        # Sliding surface
        s = x2 + self.alpha[0]*x1

        # Classic Sliding mode control
        u = -self.K[0]*x2 + self.rho[0]*(tf.math.tanh(s))


        # Switching to special sliding mode
        if abs(error) < 0.1:
            # sliding surface
            #s_s = x2 + alpha_s*x1
            s_s = x2 + self.alpha[1]*x1

            # control signal
            u = -self.K[1]*x2 + \
                self.rho[1]*(tf.math.sign(s_s))*(tf.math.sqrt(tf.math.abs(s_s)))
           

        # motor limitation      
        if abs(u.numpy()) > self.max_torque:
            u = self.max_torque*tf.math.sign(u)
       
        return u

    def learn(self,grad):

    	self.__opt.apply_gradients(zip(grad, [self.alpha,self.rho,self.K]))



########### MAIN CODE ######################

control = SmartControl()
__opt = tf.keras.optimizers.Adam()
os.system('clear')

local = False
if local:
    raw_state = np.random.rand(2)    
    x1 = raw_state[0]
    x2 = raw_state[1] #tf.Variable(0.25,name='x2')
    alpha = tf.Variable([.2,1.],name='alpha_l')
    rho = tf.Variable([2.,2.],name='rho_l')
    K = tf.Variable([0.1,0.1], name='K_l')
    print('alpha',alpha.numpy(),',rho',rho.numpy(),',K',K.numpy())



print('Initial values of Gain:',control.alpha.numpy(), 
        control.rho.numpy(), control.K.numpy(),'\n')
for i in range(5):
    with tf.GradientTape() as tape:
        if local:
            s = x2 + alpha[0]*x1
            smc = -K[0]*x2 - rho[0]*tf.math.sign(s)
        else:            
            smc = control.smc([2.2436,-3.8546])#np.random.rand(2))

    var = [var for var in tape.watched_variables()]
    print('Watched Variables:',var[0].name,var[1].name,var[2].name)
    

    if local:
        # for local computation
        grad = tape.gradient(smc, var)
        print('Gradients:',grad)
        __opt.apply_gradients(zip(grad, var))
        print('alpha',alpha.numpy(),',rho',rho.numpy(),',K',K.numpy(),'\n')
    
    else:
        # for class computation
        #grad = tape.gradient(smc, [control.alpha,control.rho,control.K])
        grad = tape.gradient(smc, var)
        print('Gradients:',grad,'\n')
        #__opt.apply_gradients(zip(grad, var))
        control.learn(grad)
        print(smc.numpy())
        print('Current values of Gain:',control.alpha.numpy(), 
        control.rho.numpy(), control.K.numpy(),'\n')
