# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:59:23 2020

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
min_lr = 0.05
gamma = 0.99
max_stps = 300
epsilon = 0.05
env = env.unwrapped
env.seed()
np.random.seed(0)
# Quantize the states
#
def get_action(state):
    if np.random.uniform(low=0, high=1) < epsilon:
        return np.random.choice(env.action_space.n)
    predict = get_predict(state, center, delta, w)   
    return np.argmax(predict)
    

def normalize_state(state):
    xbar = np.zeros((2, 2))
    xbar[0, :] = env.observation_space.low
    xbar[1, :] = env.observation_space.high
    y = np.zeros(len(state))
    for i in range(len(state)):
        y[i] = (state[i] - xbar[0, i]) / (xbar[1, i] - xbar[0, i])
    return y.reshape(1, -1)
#q_table = np.zeros((n_states, n_states, env.action_space.n))

total_steps = 0
for episode in range(episodes):
    print("Episode:", episode)
    obs = env.reset()
    total_reward = 0
    n_hidden = 20
    n = 2
    n_output = 3
    alpha = max(min_lr, initial_lr*(gamma**(episode//100)))
    target = np.zeros((1,env.action_space.n))
    #target = mat(np.random.rand(n_hidden,n))
    #alpha = 0.8
    steps = 0
    center = mat(np.random.rand(n_hidden,n))
    delta = mat(np.random.rand(1,n_hidden))
    w = mat(np.random.rand(n_hidden, n_output))
    while steps<=3000:
        env.render()
        #pos, vel = obs[0],obs[1]
        state1 = normalize_state(obs)
        state1 = np.matrix(state1)
        a = get_action(state1)
        #print('action',a)
        obs, reward, terminate,_ = env.step(a)
        total_reward += abs(obs[0]+0.5)
        state2 = normalize_state(obs)
        state2 = np.matrix(state2)
        predict = get_predict(state2, center, delta, w)  
        target[0][a] = (1-alpha) * target[0][a] + alpha*(reward+gamma*np.max(predict))
        
        center, delta, w, loss = bp_train(state1, np.matrix(target), n_hidden, 1, 0.01, 3,center, delta, w)
        steps += 1
        if terminate:
            print("Finished after: " + str(episode) + " steps"+str(steps))
            print("Cumulated Reward: " + str(total_reward))
            print("Complete!")
            break

#while True:
    #env.render()
