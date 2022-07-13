import gym
import random
import numpy as np
import time
from collections import deque
import pickle

from collections import defaultdict

EPISODES =   20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999

def default_Q_value():
    return 0

if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)

    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.

    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0

        obs = env.reset()
        done = False
        curr_done = False
        curr_obs = obs
        while curr_done == False:
            if random.uniform(0,1) < EPSILON:
                action = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(obs,i)] for i in range(env.action_space.n)])
                action =  np.argmax(prediction)
            # Update instance vars
            curr_obs, curr_done = obs, done
            obs,reward,done,info = env.step(action) # now obs and done contain the new state info 
            episode_reward += reward    
            # update current entry in Q-table
            max_q = np.array([Q_table[(obs, j)] for j in range(env.action_space.n)])
            if curr_done == True:
                Q_table[(curr_obs, action)] = Q_table[(curr_obs, action)] + LEARNING_RATE * (reward - Q_table[(curr_obs, action)])
            else:
                Q_table[(curr_obs, action)] = Q_table[(curr_obs, action)] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(max_q) - Q_table[(curr_obs, action)])
        
        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
        # update episode vars
        EPSILON = EPSILON * EPSILON_DECAY
        episode_reward_record.append(episode_reward)
        
    
    
    ####DO NOT MODIFY######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close() # added from piazza post 1279
    #######################







