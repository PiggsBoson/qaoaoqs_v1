from scipy.sparse.linalg import cg
import scipy
from utils import *
import numpy as np
import random
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class Reinforce():
    """reinforcement learning agent to maximize the expected total rewar
    """

    def __init__(self, sess, init_val, optimizer, global_step, args):
        """initilization of the reinforce class

        Arguments:
            sess -- tensorflow session
            init_val {scale} -- intialization value
            optimizer  -- tensorflow optimizer
            global_step {tensor} -- tensorflow tensor about the global steps
            args {named_tuple} -- configuration for the problem
        """
        self.sess = sess
        self.init_val = init_val
        self.optimizer = optimizer
        self.clip_val = 1.0
        self.batch_size = args.batch_size
        self.p = args.p
        self.global_step = global_step
        self.softplus = args.softplus
        self.distribution = args.distribution
        self.scale = args.scale
        self.T = args.T_tot

        self.testcase = args.testcase

        self.create_variables()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.sess.run(tf.variables_initializer(var_lists))

    def get_action(self, is_train):
        return self.sess.run(self.predicted_action, {self.is_train: is_train})

    def get_action_and_prob(self, is_train):
        """Print the action and probability

        Arguments:
            is_train {bool} -- indicators of training or testing phase

        Returns:
            action and log probs
        """
        return self.sess.run([self.predicted_action, self.logprobs], {self.is_train: is_train})

    def policy_network(self, reuse=False):
        '''
            Policy network is a main network for searching optimal protocol
        '''
        total_actions_num = []
        total_logprobs = []

        with tf.variable_scope('policy_network_variables', reuse=tf.AUTO_REUSE):
            if self.testcase == 'XXnYY':
            #Initialize to euqal duration for both Hamiltonians
                each_T = self.T/2
                protocol_a = np.random.uniform(size = self.p)
                protocol_a /= np.sum(protocol_a)
                protocol_a *= each_T
                protocol_b = np.random.uniform(size = self.p)
                protocol_b /= np.sum(protocol_b)
                protocol_b *= each_T
                protocol_init = [item for pair in zip(protocol_a, protocol_b + [0]) 
							for item in pair]

            for i in range(self.p):
                #Modified here for different initial means
                if self.testcase == 'XXnYY':
                #Initialize to euqal duration for both Hamiltonians
                    mean = tf.get_variable(name='mean_%d' % i, shape=[
                        1, 2], dtype=tf.float32, initializer=tf.constant_initializer(value=[protocol_init[2*i], protocol_init[2*i+1]] ))
                else:
                    mean = tf.get_variable(name='mean_%d' % i, shape=[
                        1, 2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.1))

                if self.softplus:
                    std = tf.get_variable(name='std_%d' % i, shape=[
                        1, 2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.01))
                    std = tf.nn.softplus(std)
                    logstd = tf.log(std)
                else:
                    logstd = tf.get_variable(name='logstd_%d' % i, shape=[
                        1, 2], dtype=tf.float32, initializer=tf.constant_initializer([-6, -6]))
                    # hard-code here: constraint the logstd
                    logstd = tf.clip_by_value(logstd, -20., 2.)
                    std = tf.exp(logstd)
                
                if self.distribution == 'logit-normal':
                    vec_actions_c = tf.cond(self.is_train,
                                    lambda:  mean + std * tf.random.normal(
                                        shape=[self.batch_size, 2], mean=0.0, stddev=1.0, dtype=tf.float32),
                                    lambda:  mean * tf.ones([self.batch_size, 2]))

                    vec_logprob_c = - 1 / 2 * tf.math.log(2 * np.pi) - logstd - 0.5 * (
                        vec_actions_c - mean) * (vec_actions_c - mean) / std ** 2

                    # sigmoid transformation
                    vec_actions_c = tf.sigmoid(vec_actions_c) * self.scale 

                    vec_logprob_c -= tf.log( vec_actions_c * (1 - vec_actions_c / self.scale ) + 1e-6 )
                    total_actions_num.append(vec_actions_c)
                    total_logprobs.append(vec_logprob_c)
                elif self.distribution =='beta':
                    pass #TODO
                else:
                    actions = tf.cond(self.is_train,
                                    lambda:  mean + std * tf.random.normal(
                                        shape=[self.batch_size, 2], mean=0.0, stddev=1.0, dtype=tf.float32),
                                    lambda:  mean * tf.ones([self.batch_size, 2]))

                    sy_logprob_n = - 1 / 2 * tf.math.log(2 * np.pi) - logstd - 0.5 * (
                        actions - mean) * (actions - mean) / std ** 2

                    total_actions_num.append(actions)
                    total_logprobs.append(sy_logprob_n)

        logprobs_tf = tf.squeeze(tf.stack(total_logprobs, axis=1))
        actions_num_tf = tf.squeeze(
            tf.stack(total_actions_num, axis=1))

        return logprobs_tf, actions_num_tf

    def replay_network(self, state, reuse=True):
        '''
            Replay network is a main network for replaying the optimal protocol
        '''
        total_logprobs = []

        with tf.variable_scope('policy_network_variables', reuse=reuse):
            for i in range(self.p):
                mean = tf.get_variable(name='mean_%d' % i, shape=[
                    1, 2], dtype=tf.float32)

                if self.softplus:
                    std = tf.get_variable(name='std_%d' % i, shape=[
                        1, 2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.01))
                    std = tf.nn.softplus(std)
                    logstd = tf.log(std)
                else:
                    logstd = tf.get_variable(name='logstd_%d' % i, shape=[
                                             1, 2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-1.0, stddev=0.1))
                    # hard-code here: constraint the logstd
                    logstd = tf.clip_by_value(logstd, -20., 2.)
                    std = tf.exp(logstd)
                
                if self.distribution == 'logit-normal':
                    def logit(x):
                        """ Computes the logit function, i.e. the logistic sigmoid inverse. """
                        return - tf.log(1. / x - 1.)
                    vec_action_c = state[:, i, :]
                    vec_action_c_ori = logit(vec_action_c / self.scale )

                    # numerical stable
                    vec_action_c_ori = tf.clip_by_value(vec_action_c_ori, -6, 6)

                    # actions has something to do with the duration
                    vec_logprob_c = - 1 / 2 * tf.math.log(2 * np.pi) - logstd - 0.5 * (
                        vec_action_c_ori - mean) * (vec_action_c_ori - mean) / std ** 2

                    vec_logprob_c -= tf.log( vec_action_c * (1 - vec_action_c / self.scale ) + 1e-6 )
                    
                    total_logprobs.append(vec_logprob_c)
                elif self.distribution =='beta':
                    pass #TODO
                else:
                    actions = state[:, i, :]
                    sy_logprob_n = - 1 / 2 * tf.math.log(2 * np.pi) - logstd - 0.5 * (
                        actions - mean) * (actions - mean) / std ** 2

                    total_logprobs.append(sy_logprob_n)

            logprobs_tf = tf.squeeze(tf.stack(total_logprobs, axis=1))

            return logprobs_tf

    def create_variables(self):
        with tf.name_scope("model_inputs"):
            self.state = tf.placeholder(
                    tf.float32, [None, self.p, 2], name="state")
            self.is_train = tf.placeholder(tf.bool, name='istrain')

        with tf.name_scope("predict_actions"):
            # initialize policy network
            with tf.variable_scope("policy_network"):

                self.logprobs, self.predicted_action = self.policy_network(reuse=False)
                self.logprobs_replay = self.replay_network(self.state, reuse=True)

            print("policy outputs: ", self.logprobs.shape)
            print('predicted_action: ', self.predicted_action.shape)

        # regularization loss
        policy_network_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")

        self.params = policy_network_variables

        print(policy_network_variables)
        print('Total parameters of the policy network: ',
              count_total_parameters(policy_network_variables))

        # compute loss and gradients
        with tf.name_scope("compute_gradients"):
            # gradients for selecting action from policy network
            self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")

            # compute policy loss and regularization loss
            self.pg_loss = -tf.reduce_mean(
                self.logprobs_replay, [1, 2])

            # vanilla policy gradient
            self.loss = tf.reduce_mean(
                self.pg_loss * [self.rewards - tf.reduce_mean(self.rewards)])

            self.entropy = tf.reduce_mean(self.pg_loss)

            # compute gradients
            self.gradients = self.optimizer.compute_gradients(self.loss)

            # compute policy gradients
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (
                        tf.clip_by_norm(grad, self.clip_val), var)

            # training update
            with tf.name_scope("train_policy_network"):
                # apply gradients to update policy network
                self.train_op = self.optimizer.apply_gradients(
                    self.gradients, global_step=self.global_step)

    def train_step(self, state, reward):
        """apply one gradient step of the policy gradient to the weights 
        
        Arguments:
            state {list} -- protocols
            reward {list} -- list of fidelity
        
        Returns:
            loss and entropy
        """
        _, ls, ent, step = self.sess.run([self.train_op, self.loss, self.entropy, self.global_step],
                                            {self.state: state,
                                                self.rewards: reward})

        return ls, ent
