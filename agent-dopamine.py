from dopamine.agents.dqn.dqn_agent import DQNAgent
import tensorflow as tf
import gym
import time

from dopamine.discrete_domains import gym_lib
import dopamine.discrete_domains.run_experiment
import dopamine.agents.dqn.dqn_agent
import dopamine.replay_memory.circular_replay_buffer
import gin.tf.external_configurables
import gin.tf

#graph
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Graph(object):
    def __init__(self):
        plt.ion() 
        plt.figure(1)
        plt.clf() 

        self.data=np.array([])

    def draw(self,data):
        plt.clf()
        self.data=np.append(self.data,data)
        index=np.arange(0,np.size(self.data))
        plt.plot(index,self.data)
        plt.draw()
        plt.pause(0.01)

class Pole(object):
    def __init__(self, init_qdn=1,graph=None):
        if init_qdn == 1:
            config = tf.ConfigProto(allow_soft_placement=True)
            session = tf.Session('', config=config)
            self.dqn = DQNAgent(session, 2, observation_shape= (4,1),
                                observation_dtype=tf.float64,
                                stack_size=1,
                                network=gym_lib.cartpole_dqn_network,
                                gamma=0.80,
                                update_horizon=1,
                                min_replay_history=500, 
                                update_period=4,
                                target_update_period=100,
                                tf_device='/cpu:*' , # use '/cpu:*' for non-GPU version
                                optimizer=tf.train.AdamOptimizer(learning_rate=0.0001,epsilon = 0.0000003125))
            session.run(tf.global_variables_initializer())
            print('init complete')
        self.env = gym.make('CartPole-v0')
        self.env.reset()

        self.observation = -1
        self.reward = -1
        self.done = -1
        self.info = -1

        self.graph=graph

    def train(self, step, display=0,checkpoint_version=0):
        self.observation, self.reward, self.done, self.info = self.env.step(
            self.env.action_space.sample())
        N = step
        L = 50
        for i in range(step):
            first_step = self.dqn.begin_episode(self.observation)
            self.render(first_step, display)
            while True:
                this_step = self.dqn.step(self.reward, self.observation)
                self.render(this_step, display)
                if self.observation[0]<-3 or self.observation[0]>3:
                    self.done=True
                if self.done == True:
                    self.dqn.end_episode(self.reward)
                    self.env.reset()
                    break
            if self.graph != None and i % 30 == 0:
                self.eval(5)
            print("{{{0}>{1}}} {2}% ".format('='*round(i*L/N),
                                             '.'*round((N-i)*L/N), round(i*100/N)), end="\r")
        # self.dqn.bundle_and_checkpoint('./',1)
        print('train complete')

    def save_agent(self,checkpoint_dir='./checkpoint',checkpoint_version=0):
        if not self.dqn.bundle_and_checkpoint(checkpoint_dir,checkpoint_version) == None:
            print('Create checkpoint file successfully')
        else:
            print('Fail to create checkpoint')

    def end(self):
        self.env.close()

    def eval(self,n=10):
        self.observation, self.reward, self.done, self.info = self.env.step(
            self.env.action_space.sample())
        self.dqn.eval_mode=True
        total_reward=0
        for _ in range(n):
            reward=0
            while True:
                this_step = self.dqn.begin_episode(self.observation)
                self.render(this_step)
                reward+=self.reward
                if self.done == True:
                    self.env.reset()
                    break
            total_reward+=reward
        average_reward=total_reward/n
        if self.graph != None:
            self.graph.draw(average_reward)
        self.dqn.eval_mode=False
        

    def render(self, control=-1, display=0):
        if control == -1:
            control = self.env.action_space.sample()
        if display != 0:
            self.env.render()
        self.observation, self.reward, self.done, self.info = self.env.step(
            control)

    def test(self, step,checkpoint_version=0):
        if not self.dqn.unbundle('./checkpoint',checkpoint_version,'./checkpoint'):
            print('Can not read the checkpoint file, no such file or directory.')
        self.observation, self.reward, self.done, self.info = self.env.step(
            self.env.action_space.sample())
        self.dqn.eval_mode=True
        for _ in range(step):
            self.total_reward=0
            while True:
                this_step = self.dqn.begin_episode(self.observation)
                self.render(this_step, 1)
                self.total_reward+=self.reward
                if self.done == True:
                    print(self.total_reward)
                    self.env.reset()
                    break


if __name__ == "__main__":
    # graph=Graph()
    pole = Pole(graph=None)
    # pole.train(800)
    # pole.save_agent()
    pole.test(100)
