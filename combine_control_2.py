import numpy
import numpy as np
import time
from math import sin, cos, tan, atan2, sqrt, pi, atan
import scipy.stats
import tensorflow as tf
import matplotlib.pyplot as plt

class SMCControl:
    #def __init__(self, alpha=[4.5,4.5], rho=[6.5,20], K=[1,0.2], max_torque=3, setpoint=0):
    def __init__(self, alpha=[1.,1.], rho=[1.,1.], K=[1.,1.], max_torque=3, setpoint=0):

        self.alpha = alpha
        self.rho = rho
        self.max_torque = max_torque
        self.K = K
        self.setpoint = setpoint

    def set_gain(self, gain):
        """
        Setting the control gains (partial) from external
        """

        self.alpha[0] = gain[0]
        self.rho[0] = gain[1]
        self.K[0] = gain[2]

    def set_specialGain(self, gain):
        """
        Setting the all control gains from external
        """

        self.alpha[1] = gain[0]
        self.rho[1] = gain[1]
        self.K[1] = gain[2]
	
    def get_normalized_state(self, state_raw):
        """
        Normalizes the system states between -1 and 1
        """

        s1 = cos(state_raw[0])
        s2 = sin(state_raw[0])
        s3 = state_raw[1]

        normalized_state = [s1, s2, s3]

        return normalized_state


    def get_lim_angle(self, state_raw):
        """
        Defines the angular position between -pi and pi
        """

        angle = state_raw[0] - 3.1416

        return angle

    def start(self,state):
        return [0]	

    def smc(self, state_raw):

        """
        	Receives state position (x, dot_x) and returns the SMC control action
        	"""

        # states
        phi = self.get_lim_angle(state_raw)
        dphi = state_raw[1]

        # For regular smc
        alpha = self.alpha[0]
        rho = self.rho[0]
        K2 = self.K[0]

        # for special smc
        alpha_s = self.alpha[1]
        rho_s = self.rho[1]
        K2_s = self.K[1]

        # Defining system states
        x1 = self.setpoint - phi
        x2 = - dphi
        error = x1

        # Sliding surface
        s = x2 + alpha*x1

        # Classic Sliding mode control
        u = -K2*x2 + rho*(np.sign(s))


        # Switching to special sliding mode
        if abs(error) < 0.1:
            # sliding surface
            s_s = x2 + alpha_s*x1

            # control signal
            u = -K2_s*x2 + rho_s*(np.sign(s_s))*(sqrt(abs(s_s)))
            #u = -K2_s*x2 + rho_s*(np.sign(s_s))*(abs(s_s)*sqrt(abs(s_s)))


        # motor limitation      
        u = tf.clip_by_value(u, clip_value_min=-self.max_torque,
                                clip_value_max=self.max_torque)      


        return u, error


    def smc_regular(self, state_raw):

        """
        Receives state position (x, dot_x) and returns the SMC control action
        """

        # states
        phi = self.get_lim_angle(state_raw)
        dphi = state_raw[1]

        # For regular smc
        alpha = self.alpha[0]
        rho = self.rho[0]
        K2 = self.K[0]

        # Defining system states
        x1 = self.setpoint - phi
        x2 = - dphi
        error = x1

        # Sliding surface
        s = x2 + alpha*x1

        # Classic Sliding mode control
        u = -K2*x2 + rho*(np.tanh(s))

        # motor limitation      
        u = tf.clip_by_value(u, clip_value_min=-self.max_torque,
                                clip_value_max=self.max_torque)      

        '''
        if abs(u) > self.max_torque:
        u = self.max_torque*np.sign(u)
        '''

        return u, x1


    def smc_special(self, state_raw):

        """
        Receives state position (x, dot_x) and returns the SMC control action
        """

        # states
        phi = self.get_lim_angle(state_raw)
        dphi = state_raw[1]

        # For regular smc
        alpha = self.alpha[1]
        rho = self.rho[1]
        K2 = self.K[1]

        # Defining system states
        x1 = self.setpoint - phi
        x2 = - dphi
        error = x1

        # Sliding surface
        s = x2 + alpha*x1

        # Special Sliding mode control
        #u = -K2*x2 + rho*(np.sign(s))*(sqrt(abs(s)))
        u = -K2*x2 + rho*(np.sign(s))*(abs(s)*sqrt(abs(s)))

        # motor limitation      
        u = tf.clip_by_value(u, clip_value_min=-self.max_torque,
                                clip_value_max=self.max_torque)      

        '''
        if abs(u) > self.max_torque:
        u = self.max_torque*np.sign(u)
        '''

        return u, x1

    def smc_special_old(self, state_raw):

        """
        Receives state position (x, dot_x) and returns the SMC control action
        """

        # states
        phi = self.get_lim_angle(state_raw)
        dphi = state_raw[1]

        # for special smc
        alpha_s = self.alpha[1]
        rho_s = self.rho[1]
        K2_s = self.K[1]

        # Defining system states
        x1 = self.setpoint - phi
        x2 = - dphi
        error = x1


        # Special sliding mode

        # sliding surface
        s_s = x2 + alpha_s*x1

        # control signal
        u = -K2_s*x2 + rho_s*(np.sign(s_s))*(sqrt(abs(s_s)))
        #u = -K2_s*x2 + rho_s*(np.sign(s_s))*(abs(s_s)*sqrt(abs(s_s)))

        # motor limitation      
        u = tf.clip_by_value(u, clip_value_min=-self.max_torque,
                                clip_value_max=self.max_torque)      

        '''
        # motor limitation      
        if abs(u) > self.max_torque:
        u = self.max_torque*np.sign(u)
        '''

        return u, x1


class PDControl(SMCControl):
    def __init__(self, K_p = 1, K_d = 1, max_torque=3, setpoint=0):

        self.K_p = K_p
        self.K_d = K_d
        self.max_torque = max_torque
        self.setpoint = setpoint

    def set_gain(self, gain):
        """
        Setting control gains from external source
        """

        self.K_p = gain[0]
        self.K_d = gain[1]

    def start(self,state):
        return [0]  

    def pd(self, state_raw):

        """
        Receives state position (x, dot_x) and returns the PD control action
        """

        # System inputs
        phi = self.get_lim_angle(state_raw)
        dphi = state_raw[1]

        # Recovering PD gains
        K_p = self.K_p
        K_d = self.K_d

        # Defining system states
        x1 = self.setpoint - phi
        x2 = - dphi
        error = x1

        # Classic Proportional-Derivative control
        u = -K_p*x1 - K_d       

        # motor limitation      
        u = tf.clip_by_value(u, clip_value_min=-self.max_torque,
                             clip_value_max=self.max_torque)      

        return u, error


class Memory:
    """Replay memory
       
       METHODS
           add    -- Add transition to memory.
           sample -- Sample minibatch from memory.
    """
    def __init__(self, states, actions, size=1000000):
        """Creates a new replay memory.
        
           Memory(states, action) creates a new replay memory for storing
           transitions with `states` observation dimensions and `actions`
           action dimensions. It can store 1000000 transitions.
           
           Memory(states, actions, size) additionally specifies how many
           transitions can be stored.
        """

        self.s = np.ndarray([size, states])
        self.a = np.ndarray([size, actions])
        self.r = np.ndarray([size, 1])
        self.sp = np.ndarray([size, states])
        self.done = np.ndarray([size, 1])
        self.n = 0
    
    def __len__(self):
        """Returns the number of transitions currently stored in the memory."""

        return self.n
    
    def add(self, s, a, r, sp, done):
        """Adds a transition to the replay memory.
        
           Memory.add(s, a, r, sp, done) adds a new transition to the
           replay memory starting in state `s`, taking action `a`,
           receiving reward `r` and ending up in state `sp`. `done`
           specifies whether the episode finished at state `sp`.
        """

        self.s[self.n, :] = s
        self.a[self.n, :] = a
        self.r[self.n, :] = r
        self.sp[self.n, :] = sp
        self.done[self.n, :] = done
        self.n += 1
    
    def sample(self, size):
        """Get random minibatch from memory.
        
        s, a, r, sp, done = Memory.sample(batch) samples a random
        minibatch of `size` transitions from the replay memory. All
        returned variables are vectors of length `size`.
        """

        idx = np.random.randint(0, self.n, size)

        return self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.done[idx]


class Network(object):
    def __init__(self, states, actions, tau=0):
        self.states = states
        self.actions = actions
        self.tau = tau
    
    def __ilshift__(self, other):
        """Copies network weights.
        
           network2 <<= network1 copies the weights from `network1` into `network2`. The
           networks must have the same structure.
        """

        if isinstance(self, DQN) or isinstance(self, Model):
            self.__model.set_weights(other.__model.get_weights())
            
        if isinstance(self, DDPG):
            self._DDPG__actor.set_weights(other._DDPG__actor.get_weights())
            self._DDPG__critic.set_weights(other._DDPG__critic.get_weights())

        # Modified to fit the TD3 algorithm
        if isinstance(self, TD3):
            if tau==0:
                self._TD3__actor.set_weights(other._TD3__actor.get_weights())
                self._TD3__critic_1.set_weights(other._TD3__critic_1.get_weights())
                self._TD3__critic_2.set_weights(other._TD3__critic_2.get_weights())
            else:
                self._TD3__actor.set_weights((1-self.tau)*_TD3__actor.get_weights() + 
                                            self.tau*other._TD3__actor.get_weights())
                
                self._TD3__critic_1.set_weights((1-self.tau)*_TD3__critic_1.get_weights() +
                                            self.tau*other._TD3__critic_1.get_weights())
                
                self._TD3__critic_2.set_weights((1-self.tau)*_TD3__critic_2.get_weights() +
                                            self.tau*other._TD3__critic_2.get_weights())

        # Modified to fit the SmartTD3 algorithm
        if isinstance(self, SmartTD3):
            self._SmartTD3__critic_1.set_weights(other._SmartTD3__critic_1.get_weights())
            self._SmartTD3__critic_2.set_weights(other._SmartTD3__critic_2.get_weights())


        return self

    def combine(self, s, a, force=False):
        """Combines state and action vectors into single network input.
        
           m, reshape = Network.combine(s, a) has five cases. In all cases,
           `m` is a matrix and `reshape` is a shape to which the network Q output
           should be reshaped. The shape will be such that states are in 
           rows and actions are in columns of `m`.
           
            1) `s` and `a` are vectors. They will be concatenated.
            2) `s` is a matrix and `a` is a vector. `a` will be replicated for
               each `s`.
            3) `s` is a vector and `a` is a matrix. `s` will be replicated for
               each `a`.
            4) `s` and `a` are matrices with the same number of rows. They will
               be concatenated.
            5) `s` and `a` are matrices with different numbers of rows or
               force=True. Each `s` will be replicated for each `a`.
              
           EXAMPLE
               >>> print(network.combine([1, 2], 5))
               (array([[1., 2., 5.]], dtype=float32), (1, 1))
               >>> print(network.combine([[1, 2], [3, 4]], 5))
               (array([[1., 2., 5.],
                       [3., 4., 5.]], dtype=float32), (2, 1))
               >>> print(network.combine([1, 2], [5, 6])) # single action only
               (array([[1., 2., 5.],
                       [1., 2., 6.]], dtype=float32), (1, 2))
               >>> print(network.combine([1, 2], [[5], [6]]))
               (array([[1., 2., 5.],
                      [1., 2., 6.]], dtype=float32), (1, 2))
               >>> print(network.combine([[1, 2], [3, 4]], [5, 6])) # single action only
               (array([[1., 2., 5.],
                       [3., 4., 6.]], dtype=float32), (2, 1))
               >>> print(network.combine([[1, 2], [3, 4]], [[5], [6]]))
               (array([[1., 2., 5.],
                       [3., 4., 6.]], dtype=float32), (2, 1))
               >>> print(network.combine([[1, 2], [3, 4]], [[5], [6]], force=True))
               (array([[1., 2., 5.],
                       [1., 2., 6.],
                       [3., 4., 5.],
                       [3., 4., 6.]], dtype=float32), (2, 2))
        """
        
        # Convert scalars to vectors
        s = np.atleast_1d(np.asarray(s, dtype=np.float32))
        a = np.atleast_1d(np.asarray(a, dtype=np.float32))
        
        # Convert vectors to matrices for single-state environments
        if self.states == 1 and len(s.shape) == 1 and s.shape[0] > 1:
            s = np.atleast_2d(s).transpose()
            
        # Convert vectors to matrices for single-action environments
        if self.actions == 1 and len(a.shape) == 1 and a.shape[0] > 1:
            a = np.atleast_2d(a).transpose()

        # Normalize to matrices
        s = np.atleast_2d(s)
        a = np.atleast_2d(a)

        # Sanity checking
        if len(s.shape) > 2 or len(a.shape) > 2:
            raise ValueError("Input dimensionality not supported")
        
        if s.shape[1] != self.states:
            raise ValueError("State dimensionality does not match network")
            
        if a.shape[1] != self.actions:
            raise ValueError("Action dimensionality does not match network")
            
        # Replicate if necessary
        if s.shape[0] != a.shape[0] or force:
            reshape = (s.shape[0], a.shape[0])
            s = np.repeat(s, np.repeat(reshape[1], reshape[0]), axis=0)
            a = np.tile(a, (reshape[0], 1))
        else:
            reshape = (s.shape[0], 1)

        m = np.hstack((s, a))

        return m, reshape


class DQN(Network):
    """Deep learning-based Q approximator.

       METHODS
           train       -- Train network.
           __call__    -- Evaluate network.
    """

    def __init__(self, states, actions=1, hiddens=[25, 25], load_model=False, 
    			name='generic_DQN'):
        """Creates a new Q approximator.
        
           DQN(states, actions) creates a Q approximator with `states`
           observation dimensions and `actions` action dimensions. It has
           two hidden layers with 25 neurons each. All layers except
           the last use ReLU activation."
           
           DQN(states, actions, hiddens) additionally specifies the
           number of neurons in the hidden layers.

           EXAMPLE
               >>> dqn = DQN(2, 1, [10, 10])
        """
        
        super(DQN, self).__init__(states, actions)
        
        # Loading a previous trained model
        if load_model:
	        self.__model = tf.keras.models.load_model("models/DQN_model_"+name)
	        self.__model.compile(loss=tf.keras.losses.MeanSquaredError(),
                             optimizer=tf.keras.optimizers.Adam())


        # Creating a new model
        else:
	        inputs = tf.keras.Input(shape=(states+actions,))
	        layer = inputs
	        for h in hiddens:
	            layer = tf.keras.layers.Dense(h, activation='relu')(layer)
	        outputs = tf.keras.layers.Dense(1, activation='linear')(layer)

	        self.__model = tf.keras.Model(inputs, outputs)
	        self.__model.compile(loss=tf.keras.losses.MeanSquaredError(),
	                             optimizer=tf.keras.optimizers.Adam())

	        # Saving model
	        self.save_model(name)


    def train(self, s, a, target):
        """Trains the Q approximator.
        
           DQN.train(s, a, target) trains the Q approximator such that
           it approaches DQN(s, a) = target.
           
           `s` is a matrix specifying a batch of observations, in which
           each row is an observation. `a` is a vector specifying an
           action for every observation in the batch. `target` is a vector
           specifying a target value for each observation-action pair in
           the batch.
           
           EXAMPLE
               >>> dqn = DQN(2, 1)
               >>> dqn.train([[0.1, 2], [0.4, 3], [0.2, 5]], [-1, 1, 0], [12, 16, 19])
        """
           
        return self.__model.train_on_batch(self.combine(s, a), np.atleast_1d(target))

    def __call__(self, s, a):
        """Evaluates the Q approximator.
        
           DQN(s, a) returns the value of the approximator at observation
           `s` and action `a`.
           
           `s` is either a vector specifying a single observation, or a
           matrix in which each row specifies one observation in a batch.
           If `a` is the same size as the number of rows in `s`, it specifies
           the action at which to evaluate each observation in the batch.
           Otherwise, it specifies the action(s) at which the evaluate ALL
           observations in the batch.
           
           EXAMPLE
               >>> dqn = DQN(2, 1)
               >>> # single observation and action
               >>> print(dqn([0.1, 2], -1))
               [[ 12 ]]
               >>> # batch of observations and actions
               >>> print(dqn([[0.1, 2], [0.4, 3]], [-1, 1]))
               [[12]
                [16]]
               >>> # evaluate single observation at multiple actions
               >>> print(dqn([0.1, 2], [-1, 1]))
               [[12  -12]]
        """

        inp, reshape = self.combine(s, a)
        return np.reshape(np.asarray(self.__model(inp)), reshape)

    def __ilshift__(self, other):
        """Copies network weights.
        
           network2 <<= network1 copies the weights from `network1` into `network2`. The
           networks must have the same structure.
        """

        self.__model.set_weights(other.__model.get_weights())

        return self

    def save_model(self, name):
    	self.__model.save("models/DQN_model_"+name)


class DDPG(Network):
    """Deep Deterministic Policy Gradient

       METHODS
           train       -- Train network.
           critic      -- Evaluate critic network.
           actor       -- Evaluate actor network.
    """

    def __init__(self, states, actions=1, max_action=1, hiddens=[25, 25], load_agent=False, 
    			name='generic'):
        """Creates a new DDPG network.
        
           DDPG(states, actions) creates a DDPG network with `states`
           observation dimensions and `actions` action dimensions. It has
           two hidden layers with 25 neurons each. All layers except
           the last use ReLU activation. The last actor layer uses the
           hyperbolic tangent. As such, all actions are scaled to [-1, 1]."
           
           DDPG(states, actions, hiddens) additionally specifies the
           number of neurons in the hidden layers.

           EXAMPLE
               >>> ddpg = DDPG(2, 1, [10, 10])
        """

        self.max_action = max_action
        self.actions = actions

        super(DDPG, self).__init__(states, actions)

        # Loading a previous created agent
        if load_agent:

	        # Actor
	        self.__actor = tf.keras.models.load_model("models/DDPG_actor_"+ name)
	        self.__opt = tf.keras.optimizers.Adam()

	        # Critic
	        self.__critic = tf.keras.models.load_model("models/DDPG_critic_"+ name)
	        self.__critic.compile(loss=tf.keras.losses.MeanSquaredError(),
	                              optimizer=tf.keras.optimizers.Adam())
        
        # Creating a new agent
        else:    	
	        # Actor
	        inputs = tf.keras.Input(shape=(states,))
	        layer = inputs
	        for h in hiddens:
	            layer = tf.keras.layers.Dense(h, activation='relu')(layer)
	        outputs = tf.keras.layers.Dense(actions, activation='tanh')(layer)
	        self.__actor = tf.keras.Model(inputs, outputs)
	        self.__opt = tf.keras.optimizers.Adam()
	        
	        # Critic
	        inputs = tf.keras.Input(shape=(states+actions,))
	        layer = inputs
	        for h in hiddens:
	            layer = tf.keras.layers.Dense(h, activation='relu')(layer)
	        outputs = tf.keras.layers.Dense(1, activation='linear')(layer)

	        self.__critic = tf.keras.Model(inputs, outputs)
	        self.__critic.compile(loss=tf.keras.losses.MeanSquaredError(),
	                              optimizer=tf.keras.optimizers.Adam())
	        # Saving model
	        self.save_model(name)
	        

    def train(self, s, a, target):
        """Trains both critic and actor.
        
           DDPG.train(s, a, target) trains the critic such that
           it approaches DDPG.critic(s, a) = target, and the actor to
           approach DDPG.actor(s) = max_a'(DDPG.critic(s, a'))
           
           `s` is a matrix specifying a batch of observations, in which
           each row is an observation. `a` is a vector specifying an
           action for every observation in the batch. `target` is a vector
           specifying a target value for each observation-action pair in
           the batch.
           
           EXAMPLE
               >>> ddpg = DDPG(2, 1)
               >>> ddpg.train([[0.1, 2], [0.4, 3], [0.2, 5]], [-1, 1, 0], 
               					[12, 16, 19])
        """
        
        # Critic
        self.__critic.train_on_batch(self.combine(s, a), np.atleast_1d(target))
        
        # Actor
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        with tf.GradientTape() as tape:
            q = -self.__critic.call(tf.concat([s, self.__actor(s)], 1))
        grad = tape.gradient(q, self.__actor.variables)
        self.__opt.apply_gradients(zip(grad, self.__actor.variables))
        
    def critic(self, s, a=None):
        """Evaluates the value function (critic).
        
           DDPG.critic(s) returns the value of the approximator at observation
           `s` and the actor's action.

           DDPG.critic(s, a) returns the value of the approximator at observation
           `s` and action `a`.
           
           `s` is either a vector specifying a single observation, or a
           matrix in which each row specifies one observation in a batch.
           If `a` is the same size as the number of rows in `s`, it specifies
           the action at which to evaluate each observation in the batch.
           Otherwise, it specifies the action(s) at which the evaluate ALL
           observations in the batch.
           
           EXAMPLE
               >>> ddpg = DQN(2, 1)
               >>> # single observation and action
               >>> print(ddpg.critic([0.1, 2], -1))
               [[ 12 ]]
               >>> # batch of observations and actions
               >>> print(ddpg.critic([[0.1, 2], [0.4, 3]], [-1, 1]))
               [[12]
                [16]]
               >>> # evaluate single observation at multiple actions
               >>> print(ddpg.critic([0.1, 2], [-1, 1]))
               [[12  -12]]
        """
        
        if a is None:
            s = tf.convert_to_tensor(np.atleast_2d(s), dtype=tf.float32)
            out = self.__critic(tf.concat([s, self.__actor(s)], 1)).numpy()
            return out
        else:
            inp, reshape = self.combine(s, a)
            return np.reshape(np.asarray(self.__critic(inp)), reshape)

    def actor(self, s, noise=None):
        """Evaluates the policy(actor).
        
           DDPG.actor(s) returns the action to take in state `s`.
           
           `s` is either a vector specifying a single observation, or a
           matrix in which each row specifies one observation in a batch.
           
           EXAMPLE
               >>> ddpg = DDPG(2, 1)
               >>> # single observation
               >>> print(ddpg.actor([0.1, 2]))
               [-0.23]
               >>> # batch of observations
               >>> print(dqn([[0.1, 2], [0.4, 3]]))
               [[-0.23]
                [0.81]]
        """
        
        single = len(np.asarray(s).shape) == 1

        s = tf.convert_to_tensor(np.atleast_2d(s), dtype=tf.float32)
        out = self.__actor(s).numpy() * self.max_action

        # Applying noise to actor
        if noise is not None:
        	out += noise * np.random.randn(self.actions)
        	out = np.clip(out, -self.max_action, self.max_action)

        if single:
        	out = out[0]

        return out

    def __ilshift__(self, other):
        self._DDPG__actor.set_weights(other._DDPG__actor.get_weights())
        self._DDPG__critic.set_weights(other._DDPG__critic.get_weights())

        return self
    
    def save_model(self, name):
    	self.__actor.save("models/DDPG_actor_"+name)
    	self.__critic.save("models/DDPG_critic_"+name)


class TD3(Network):
    """Deep Deterministic Policy Gradient

       METHODS
           train       -- Train network.
           critic      -- Evaluate critic network.
           actor       -- Evaluate actor network.
    """

    def __init__(self, states, actions=1, hiddens=[25, 25], act_out_layer='tanh',load_agent=False, 
                    name='generic', path=None, load_critic=True, save_critic=True,tau=0):
        """Creates a new DDPG network.
        
           DDPG(states, actions) creates a DDPG network with `states`
           observation dimensions and `actions` action dimensions. It has
           two hidden layers with 25 neurons each. All layers except
           the last use ReLU activation. The last actor layer uses the
           hyperbolic tangent. As such, all actions are scaled to [-1, 1]."
           
           DDPG(states, actions, hiddenns) additionally specifies the
           number of neurons in the hidden layers.

           EXAMPLE
               >>> ddpg = DDPG(2, 1, [10, 10])
        """

        self.states = states
        self.actions = actions
        self.tau = tau
        self.name = name

        super(TD3, self).__init__(states, actions)

        if path is None:
            path = "models/"
        else:
            path = path+"models/"

        # Loading a previous created agent
        if load_agent:

            # Actor
            #self.__actor = tf.keras.models.load_model("models/TD3_actor_"+ name)
            self.__actor = tf.keras.models.load_model(path+"TD3_actor_"+ name)
            self.__opt = tf.keras.optimizers.Adam()

            if load_critic:
                # Critic #1
                #self.__critic_1 = tf.keras.models.load_model("models/TD3_critic_1_"+ name)
                self.__critic_1 = tf.keras.models.load_model(path+"TD3_critic_1_"+ name)
                self.__critic_1.compile(loss=tf.keras.losses.MeanSquaredError(),
                                      optimizer=tf.keras.optimizers.Adam())
            
                # Critic #2
                #self.__critic_2 = tf.keras.models.load_model("models/TD3_critic_2_"+ name)          
                self.__critic_2 = tf.keras.models.load_model(path+"TD3_critic_2_"+ name)	        
                self.__critic_2.compile(loss=tf.keras.losses.MeanSquaredError(),
                                      optimizer=tf.keras.optimizers.Adam())
                

            
                '''
                self.__actor.summary()
                self.__critic_1.summary()
                self.__critic_2.summary()
                input('Press any key to continue...')
                '''

        # Creating a new agent
        else:    	
            # Actor
            inputs = tf.keras.Input(shape=(states,))
            layer = inputs
            for h in hiddens:

                layer = tf.keras.layers.Dense(h, activation='relu')(layer)
            #outputs = tf.keras.layers.Dense(actions, activation='linear')(layer)
            outputs = tf.keras.layers.Dense(actions, activation=act_out_layer)(layer)
            self.__actor = tf.keras.Model(inputs, outputs)
            self.__opt = tf.keras.optimizers.Adam()
            

            # Critic #1
            inputs = tf.keras.Input(shape=(states+actions,))
            layer = inputs
            for h in hiddens:
                layer = tf.keras.layers.Dense(h, activation='relu')(layer)
            outputs = tf.keras.layers.Dense(1, activation='linear')(layer)

            self.__critic_1 = tf.keras.Model(inputs, outputs)
            self.__critic_1.compile(loss=tf.keras.losses.MeanSquaredError(),
                                  optimizer=tf.keras.optimizers.Adam())
            
            # Critic #2
            inputs = tf.keras.Input(shape=(states+actions,))
            layer = inputs
            for h in hiddens:
                layer = tf.keras.layers.Dense(h, activation='relu')(layer)
            outputs = tf.keras.layers.Dense(1, activation='linear')(layer)

            self.__critic_2 = tf.keras.Model(inputs, outputs)
            self.__critic_2.compile(loss=tf.keras.losses.MeanSquaredError(),
                                  optimizer=tf.keras.optimizers.Adam())
            '''
            self.__actor.summary()
            self.__critic_1.summary()
            self.__critic_2.summary()
            input('Press any key to continue...')
            '''
            # Saving model 
            self.save_model(name=name, path=path, save_critic=save_critic)
	        

    def train(self, s, a, target, actor_update=True):
        """Trains both critic and actor.
        
           DDPG.train(s, a, target) trains the critic such that
           it approaches DDPG.critic(s, a) = target, and the actor to
           approach DDPG.actor(s) = max_a'(DDPG.critic(s, a'))
           
           `s` is a matrix specifying a batch of observations, in which
           each row is an observation. `a` is a vector specifying an
           action for every observation in the batch. `target` is a vector
           specifying a target value for each observation-action pair in
           the batch.
           
           EXAMPLE
               >>> ddpg = DDPG(2, 1)
               >>> ddpg.train([[0.1, 2], [0.4, 3], [0.2, 5]], [-1, 1, 0], 
               					[12, 16, 19])
        """
        
        # Critic
        self.__critic_1.train_on_batch(self.combine(s, a), np.atleast_1d(target))
        self.__critic_2.train_on_batch(self.combine(s, a), np.atleast_1d(target))
        

        # For time delayed case
        if actor_update:
        	# Actor
	        s = tf.convert_to_tensor(s, dtype=tf.float32)
	        with tf.GradientTape() as tape:
	            q_1 = -self.__critic_1.call(tf.concat([s, self.__actor(s)], 1))
	            q_2 = -self.__critic_2.call(tf.concat([s, self.__actor(s)], 1))

	            q = tf.math.minimum(q_1, q_2)

	        grad = tape.gradient(q, self.__actor.variables)
	        self.__opt.apply_gradients(zip(grad, self.__actor.variables))
        
    
    def critic(self, s, a=None, noise=None):
        """Evaluates the value function (critic).
        
           DDPG.critic(s) returns the value of the approximator at observation
           `s` and the actor's action.

           DDPG.critic(s, a) returns the value of the approximator at observation
           `s` and action `a`.
           
           `s` is either a vector specifying a single observation, or a
           matrix in which each row specifies one observation in a batch.
           If `a` is the same size as the number of rows in `s`, it specifies
           the action at which to evaluate each observation in the batch.
           Otherwise, it specifies the action(s) at which the evaluate ALL
           observations in the batch.

           If `noise`~=None, DDPG.critic(s,noise!=None) will return the value 
           of approximator at observation `s` with noisy action `a`
           
		

           EXAMPLE
               >>> ddpg = DQN(2, 1)
               >>> # single observation and action
               >>> print(ddpg.critic([0.1, 2], -1))
               [[ 12 ]]
               >>> # batch of observations and actions
               >>> print(ddpg.critic([[0.1, 2], [0.4, 3]], [-1, 1]))
               [[12]
                [16]]
               >>> # evaluate single observation at multiple actions
               >>> print(ddpg.critic([0.1, 2], [-1, 1]))
               [[12  -12]]
        """
        
        if a is None:
            s = tf.convert_to_tensor(np.atleast_2d(s), dtype=tf.float32)
            a = self.__actor(s, noise)
            q_1 = self.__critic_1(tf.concat([s, a], 1)).numpy()
            q_2 = self.__critic_2(tf.concat([s, a], 1)).numpy()
            out = tf.math.minimum(q_1, q_2)

            return out

        else:
            if noise is not None:
                a+=np.clip(noise * np.random.randn(self.actions),-0.1,0.1)
        
            inp, reshape = self.combine(s, a)

            q_1 = np.reshape(np.asarray(self.__critic_1(inp)), reshape)
            q_2 = np.reshape(np.asarray(self.__critic_2(inp)), reshape)

            out = tf.math.minimum(q_1, q_2)

            return out

    
    def actor(self, s, noise=None):
        """Evaluates the policy(actor).
        
           DDPG.actor(s) returns the action to take in state `s`.
           
           `s` is either a vector specifying a single observation, or a
           matrix in which each row specifies one observation in a batch.
           
           EXAMPLE
               >>> ddpg = DDPG(2, 1)
               >>> # single observation
               >>> print(ddpg.actor([0.1, 2]))
               [-0.23]
               >>> # batch of observations
               >>> print(dqn([[0.1, 2], [0.4, 3]]))
               [[-0.23]
                [0.81]]
        """
        
        single = len(np.asarray(s).shape) == 1

        s = tf.convert_to_tensor(np.atleast_2d(s), dtype=tf.float32)
        out = self.__actor(s).numpy()

        # Applying noise to actor for e-greedy
        if noise is not None:
        	out += noise * np.random.randn(self.actions)
        	#out = np.clip(out, 0,1)

        if single:
            out = out[0]

        return out


    def __ilshift__(self, other):
        self._TD3__actor.set_weights(other._TD3__actor.get_weights())
        self._TD3__critic_1.set_weights(other._TD3__critic_1.get_weights())
        self._TD3__critic_2.set_weights(other._TD3__critic_2.get_weights())

        return self
    
    
    def save_model(self, name="generic", path=None, save_critic=True):
        if path is None:
            path = ""

        self.__actor.save(path+"models/TD3_actor_"+name)

        if save_critic:
            self.__critic_1.save(path+"models/TD3_critic_1_"+name)
            self.__critic_2.save(path+"models/TD3_critic_2_"+name)


class SmartControl(object):
    #def __init__(self, alpha=[4.5,4.5], rho=[2.85,1.], K=[0.3185,0.005], max_torque=3, setpoint=0):
    def __init__(self, alpha=[4.5,1.], rho=[1.,1.], K=[0.1,0.005], max_torque=3, 
                    setpoint=0, state_raw=[0.,0.]):

        self.alpha = tf.Variable(alpha,trainable=True, name='alpha')
        self.rho = tf.Variable(rho, trainable=True, name='rho')
        self.K = tf.Variable(K,dtype=tf.float32, trainable=True, name='K')
        self.max_torque = max_torque
        self.setpoint = setpoint
        self.max_torque = max_torque
        self.state_raw = state_raw
        self.__opt = tf.keras.optimizers.Adam()
        
        #self.__opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, 
        #                                    beta_2=0.999, epsilon=1e-07, 
        #                                    amsgrad=False, name='Adam')
        
        #self.__opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.98)
        #self.__opt = tf.keras.optimizers.Adagrad(learning_rate=0.1, 
        #                initial_accumulator_value=0.1, epsilon=1e-06,
        #                name='Adagrad')

        #super(SmartControl, self).__init__()
   
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
        angle = state_raw[:,] - pi

        return angle

    def smc(self, state_raw):

        '''
        Receives state position (x, dot_x) and returns the SMC control action
        '''

        # states
       # states
        phi = self.get_lim_angle(state_raw[:,0])
        dphi = state_raw[:,1]

        # Defining system states
        x1 = self.setpoint - phi
        x2 = - dphi
        error = x1
        #print('States x1',x1,'x2',x2)
         # Selecting the SMC mode
        if abs(error) > 0.1:
            # Sliding surface
            s = x2 + self.alpha[0]*x1
            #print('s',s.numpy())
            # Classic SMC
            u = -self.K[0]*x2 + self.rho[0]*(tf.math.sign(s))
            #print('Control',u)
        else:
            # sliding surface
            s = x2 + self.alpha[1]*x1

            # Special SMC
            u = -self.K[1]*x2 + \
                self.rho[1]*(tf.math.sign(s))*(tf.math.sqrt(tf.math.abs(s)))
        
        # motor limitation      
        
        u = tf.clip_by_value(u, 
                             clip_value_min=-self.max_torque,
                             clip_value_max=self.max_torque)      
        
        return u

    def smc_regular(self, state_raw):

        '''
        Receives state position (x, dot_x) and returns the SMC control action
        '''
        phi = self.get_lim_angle(state_raw[:,0])
        dphi = state_raw[:,1]

        # Defining system states
        x1 = self.setpoint - phi
        x2 = - dphi
        error = x1
        #print('States x1',x1,'x2',x2)

        # Sliding surface
        s = x2 + self.alpha[0]*x1

        # Classic SMC
        u = -self.K[0]*x2 + self.rho[0]*(tf.math.tanh(s))

        # motor limitation      
        u = tf.clip_by_value(u, 
                             clip_value_min=-self.max_torque,
                             clip_value_max=self.max_torque)      
        

        return u

    def smc_special(self, state_raw):

        '''
        Receives state position (x, dot_x) and returns the SMC control action
        '''
        # states
        phi = self.get_lim_angle(state_raw)
        dphi = state_raw[1]

        # for special smc
        alpha_s = self.alpha[1]
        rho_s = self.rho[1]
        K2_s = self.K[1]

        # Defining system states
        x1 = self.setpoint - phi
        x2 = - dphi
        error = x1


        # Special sliding mode

        # sliding surface
        s_s = x2 + alpha_s*x1

        # control signal
        u = -K2_s*x2 + rho_s*(np.sign(s_s))*(sqrt(abs(s_s)))
        #u = -K2_s*x2 + rho_s*(np.sign(s_s))*(abs(s_s)*sqrt(abs(s_s)))

        # motor limitation      
        if abs(u) > self.max_torque:
            u = self.max_torque*np.sign(u)

        return u

    def learn(self,grad):

        self.__opt.apply_gradients(zip(grad, [self.alpha, self.rho, self.K]))


class SmartTD3(Network):
    """Deep Deterministic Policy Gradient

       METHODS
           train       -- Train network.
           critic      -- Evaluate critic network.
           actor       -- Evaluate actor network.
    """

    def __init__(self, states, actions=1, hiddens=[25, 25]):
        """Creates a new DDPG network.
        
           DDPG(states, actions) create0s a DDPG network with `states`
           observation dimensions and `actions` action dimensions. It has
           two hidden layers with 25 neurons each. All layers except
           the last use ReLU activation. The last actor layer uses the
           hyperbolic tangent. As such, all actions are scaled to [-1, 1]."
           
           DDPG(states, actions, hiddenns) additionally specifies the
           number of neurons in the hidden layers.

           EXAMPLE
               >>> ddpg = DDPG(2, 1, [10, 10])
        """

        self.states = states
        self.actions = actions


        super(SmartTD3, self).__init__(states, actions)

        # Actor using SMC
        self.__actor = SmartControl()
        self.alpha = self.__actor.alpha
        self.K = self.__actor.K
        self.rho = self.__actor.rho

        '''
        inputs = tf.keras.Input(shape=(states,))
        layer = inputs
        for h in hiddens:
            layer = tf.keras.layers.Dense(h, activation='relu')(layer)
        outputs = tf.keras.layers.Dense(actions, activation='tanh')(layer)
        self.__actor = tf.keras.Model(inputs, outputs)
        self.__opt = tf.keras.optimizers.Adam()
        '''


        # Critic #1
        inputs = tf.keras.Input(shape=(states+actions,))
        layer = inputs
        for h in hiddens:
            layer = tf.keras.layers.Dense(h, activation='relu')(layer)
        outputs = tf.keras.layers.Dense(1, activation='linear')(layer)

        self.__critic_1 = tf.keras.Model(inputs, outputs)
        self.__critic_1.compile(loss=tf.keras.losses.MeanSquaredError(),
                              optimizer=tf.keras.optimizers.Adam())
        
        # Critic #2
        inputs = tf.keras.Input(shape=(states+actions,))
        layer = inputs
        for h in hiddens:
            layer = tf.keras.layers.Dense(h, activation='relu')(layer)
        outputs = tf.keras.layers.Dense(1, activation='linear')(layer)

        self.__critic_2 = tf.keras.Model(inputs, outputs)
        self.__critic_2.compile(loss=tf.keras.losses.MeanSquaredError(),
                              optimizer=tf.keras.optimizers.Adam())

    def train(self, s, a, target, actor_update=True):
        """Trains both critic and actor.
        
           DDPG.train(s, a, target) trains the critic such that
           it approaches DDPG.critic(s, a) = target, and the actor to
           approach DDPG.actor(s) = max_a'(DDPG.critic(s, a'))
           
           `s` is a matrix specifying a batch of observations, in which
           each row is an observation. `a` is a vector specifying an
           action for every observation in the batch. `target` is a vector
           specifying a target value for each observation-action pair in
           the batch.
           
           EXAMPLE
               >>> ddpg = DDPG(2, 1)
               >>> ddpg.train([[0.1, 2], [0.4, 3], [0.2, 5]], [-1, 1, 0], 
               					[12, 16, 19])
        """
        
        # Critic
        self.__critic_1.train_on_batch(self.combine(s, a), np.atleast_1d(target))
        self.__critic_2.train_on_batch(self.combine(s, a), np.atleast_1d(target))
        
        # Actor (with delayed update)
        if actor_update:
            
            s = tf.convert_to_tensor(s, dtype=tf.float32)
            raw_state = self.__actor.get_raw_state(s)
            
            with tf.GradientTape() as tape:

                a = self.__actor.smc_regular(raw_state)
                a = tf.expand_dims(a,axis=1)

                q_1 = -self.__critic_1.call(tf.concat([s, a], 1))
                q_2 = -self.__critic_2.call(tf.concat([s, a], 1))

                q = tf.math.minimum(q_1, q_2)

            #grad = tape.gradient(q, [self.__actor.alpha,self.__actor.rho,self.__actor.K])
            grad = tape.gradient([q_1,q_2], [self.__actor.alpha,self.__actor.rho,self.__actor.K])
            self.__actor.learn(grad)
        
    
    def critic(self, s, a=None, noise=None):
        """Evaluates the value function (critic).
        
           DDPG.critic(s) returns the value of the approximator at observation
           `s` and the actor's action.

           DDPG.critic(s, a) returns the value of the approximator at observation
           `s` and action `a`.
           
           `s` is either a vector specifying a single observation, or a
           matrix in which each row specifies one observation in a batch.
           If `a` is the same size as the number of rows in `s`, it specifies
           the action at which to evaluate each observation in the batch.
           Otherwise, it specifies the action(s) at which the evaluate ALL
           observations in the batch.

           If `noise`~=None, DDPG.critic(s,noise!=None) will return the value 
           of approximator at observation `s` with noisy action `a`
           
		

           EXAMPLE
               >>> ddpg = DQN(2, 1)
               >>> # single observation and action
               >>> print(ddpg.critic([0.1, 2], -1))
               [[ 12 ]]
               >>> # batch of observations and actions
               >>> print(ddpg.critic([[0.1, 2], [0.4, 3]], [-1, 1]))
               [[12]
                [16]]
               >>> # evaluate single observation at multiple actions
               >>> print(ddpg.critic([0.1, 2], [-1, 1]))
               [[12  -12]]
        """
        
        if a is None:
            s = tf.convert_to_tensor(np.atleast_2d(s), dtype=tf.float32)
            raw_state = self.__actor.get_raw_state(s)
            #print('Raw state',raw_state)

            a = self.__actor.smc_regular(raw_state)
            a = tf.expand_dims(a,axis=1)
            #a = tf.convert_to_tensor(np.atleast_2d(a), dtype=tf.float32)
            q_1 = self.__critic_1(tf.concat([s, a], 1)).numpy()
            q_2 = self.__critic_2(tf.concat([s, a], 1)).numpy()

            out = tf.math.minimum(q_1, q_2)
            #out = np.amin([q_1, q_2], axis=0)

            return out

        else:
            if noise is not None:
                a+=np.clip(noise * np.random.randn(self.actions),-0.1,0.1)
            

            inp, reshape = self.combine(s, a)

            q_1 = np.reshape(np.asarray(self.__critic_1(inp)), reshape)
            q_2 = np.reshape(np.asarray(self.__critic_2(inp)), reshape)

            q = tf.math.minimum(q_1, q_2)
            #q = np.amin([q_1, q_2], axis=0)

            return q


    def actor(self,state_raw,noise=None):

        #getting the action from SMC control
        state_raw = np.atleast_2d(state_raw)
        out = self.__actor.smc_regular(state_raw)

        # Applying noise to actor for e-greedy
        if noise is not None:
            out += noise * np.random.randn(self.actions)
            #out = np.clip(out, -self.__actor.max_torque,self.__actor.max_torque)
        out = tf.clip_by_value(out,
                               clip_value_min=-self.__actor.max_torque,
                               clip_value_max=self.__actor.max_torque)
        return out


    def old_actor(self, s, noise=None):
        """Evaluates the policy(actor).
        
           DDPG.actor(s) returns the action to take in state `s`.
           
           `s` is either a vector specifying a single observation, or a
           matrix in which each row specifies one observation in a batch.
           
           EXAMPLE
               >>> ddpg = DDPG(2, 1)
               >>> # single observation
               >>> print(ddpg.actor([0.1, 2]))
               [-0.23]
               >>> # batch of observations
               >>> print(dqn([[0.1, 2], [0.4, 3]]))
               [[-0.23]
                [0.81]]
        """
        
        single = len(np.asarray(s).shape) == 1

        s = tf.convert_to_tensor(np.atleast_2d(s), dtype=tf.float32)
        out = self.__actor(s).numpy()

        # Applying noise to actor for e-greedy
        if noise is not None:
        	out += noise * np.random.randn(self.actions)
        	out = np.clip(out, -1,1)

        if single:
        	out = out[0]

        return out


    def __ilshift__(self, other):
        self._SmartTD3__critic_1.set_weights(other._SmartTD3__critic_1.get_weights())
        self._SmartTD3__critic_2.set_weights(other._SmartTD3__critic_2.get_weights())

        return self
    
    