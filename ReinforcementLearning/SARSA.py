from collections import deque
import gym
import random
import numpy as np
import time
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
        obs_prime = obs
        reward_prime = 0
        done_prime = False
        # get first action
        if random.uniform(0,1) < EPSILON:
            action = env.action_space.sample()
        else:
            prediction = np.array([Q_table[(obs,i)] for i in range(env.action_space.n)])
            action =  np.argmax(prediction)

        while done_prime == False:
            # Update instance vars
            obs, reward, done = obs_prime, reward_prime, done_prime
            obs_prime,reward_prime,done_prime,info = env.step(action) # get the next state
            # get a' from next state
            if random.uniform(0,1) < EPSILON:
                action_prime = env.action_space.sample()
            else:
                pred_prime = np.array([Q_table[(obs_prime,i)] for i in range(env.action_space.n)])
                action_prime =  np.argmax(pred_prime)
            # cumulate reward
            episode_reward += reward_prime
            # update current entry in Q-table
            Q_table[(obs, action)] = Q_table[(obs, action)] + LEARNING_RATE * (reward_prime + DISCOUNT_FACTOR * Q_table[(obs_prime, action_prime)] - Q_table[(obs, action)]) 
            # needed to update after using in Q-table
            action = action_prime 
        # done, update terminal state
        Q_table[(obs_prime, action_prime)] = Q_table[(obs_prime, action_prime)] + LEARNING_RATE * (reward_prime - Q_table[(obs_prime, action_prime)]) 
        
        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
        
        # update episode vars
        EPSILON = EPSILON * EPSILON_DECAY
        episode_reward_record.append(episode_reward)

    ####DO NOT MODIFY######
    model_file = open('SARSA_Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close() # added from piazza post 1279
    #######################



