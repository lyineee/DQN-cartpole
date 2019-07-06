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

class Pole(object):
    def __init__(self, init_qdn=1):
        if init_qdn == 1:
            config = tf.ConfigProto(allow_soft_placement=True)
            session = tf.Session('', config=config)
            self.dqn = DQNAgent(session, 2, observation_shape= (4,1),
                                observation_dtype=tf.float64,
                                stack_size=1,
                                network=gym_lib.cartpole_dqn_network,
                                gamma=0.99,
                                update_horizon=1,
                                min_replay_history=500, 
                                update_period=4,
                                target_update_period=100,
                                tf_device='/cpu:*' , # use '/cpu:*' for non-GPU version
                                optimizer=tf.train.AdamOptimizer(epsilon = 0.0003125))
            session.run(tf.global_variables_initializer())
            print('init complete')
        self.env = gym.make('CartPole-v0')
        self.env.reset()

        self.observation = -1
        self.reward = -1
        self.done = -1
        self.info = -1

    def train(self, step, display=0):
        self.observation, self.reward, self.done, self.info = self.env.step(
            self.env.action_space.sample())
        # self.observation
        N = step
        L = 50
        for i in range(step):
            # start_time=time.time()
            # red=0
            first_step = self.dqn.begin_episode(self.observation)
            self.render(first_step, display)
            while True:
                # red+=self.reward
                this_step = self.dqn.step(self.reward, self.observation)
                self.render(this_step, display)
                if self.observation[0]<-3 or self.observation[0]>3:
                    self.done=True
                if self.done == True:
                    self.dqn.end_episode(self.reward)
                    self.env.reset()
                    break

            print("{{{0}>{1}}} {2}% ".format('='*round(i*L/N),
                                             '.'*round((N-i)*L/N), round(i*100/N)), end="\r")
        # self.dqn.bundle_and_checkpoint('./',1)
        print('train complete')

    def end(self):
        self.env.close()

    def render(self, control=-1, display=0):
        if control == -1:
            control = self.env.action_space.sample()
        if display != 0:
            self.env.render()
        self.observation, self.reward, self.done, self.info = self.env.step(
            control)

    def test(self, step):
        self.observation, self.reward, self.done, self.info = self.env.step(
            self.env.action_space.sample())
        self.dqn.eval_mode=True
        for i in range(step):
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
    pole = Pole()
    pole.train(1000)
    pole.test(100)
