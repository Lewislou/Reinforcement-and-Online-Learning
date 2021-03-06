# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:50:57 2020

@author: asus
"""

###Mountain car problem
import gym
import numpy as np
from numpy import mat
from rbf import bp_train,get_predict
from sklearn.preprocessing import normalize

env_name = 'MountainCar-v0'
env = gym.make(env_name)
obs = env.reset()
env.render()
# Some initializations
#
n_states = 40
episodes = 10
initial_lr = 1.0
min_lr = 0.005
gamma = 0.99
max_stps = 300
epsilon = 0.05
env = env.unwrapped
env.seed()
np.random.seed(0)
# Quantize the states
#

    
def discretization(env, obs):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_den = (env_high - env_low) / n_states
    pos_den = env_den[0]
    vel_den = env_den[1]
    pos_high = env_high[0]
    pos_low = env_low[0]
    vel_high = env_high[1]
    vel_low = env_low[1]
    pos_scaled = int((obs[0] - pos_low) / pos_den)
    vel_scaled = int((obs[1] - vel_low) / vel_den)
    return pos_scaled, vel_scaled
q_table = np.zeros((n_states, n_states, env.action_space.n))
total_steps = 0
for episode in range(episodes):
    print("Episode:", episode)
    obs = env.reset()
    total_reward = 0
    alpha = max(min_lr, initial_lr*(gamma**(episode//100)))
    steps = 0
    while steps <= 1000:
        env.render()
        pos, vel = discretization(env, obs)
        #state = normalize(np.array([pos, vel]).reshape(1,2), axis=1, norm='max')
        #state = np.matrix(state)
        #a = get_action(state)
        if np.random.uniform(low=0, high=1) < epsilon:
            a = np.random.choice(env.action_space.n)
        else:
            a = np.argmax(q_table[pos][vel])
        obs, reward, terminate,_ = env.step(a)
        total_reward += abs(obs[0]+0.5)
        pos_, vel_ = discretization(env, obs)
        q_table[pos][vel][a] = (1-alpha)*q_table[pos][vel][a] + alpha*(reward+gamma*np.max(q_table[pos_][vel_]))
        #center, delta, w, loss = bp_train(state, np.matrix(q_table[pos][vel]), 20, 1, 0.001, 3,center, delta, w)
        steps += 1
        if terminate:
            print("Finished after: " + str(episode) + " steps"+str(steps))
            print("Cumulated Reward: " + str(total_reward))
            print("Complete!")
            break

while True:
    env.render()
