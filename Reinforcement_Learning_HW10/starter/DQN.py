import torch.nn as nn
import torch.nn.functional as functional
from collections import deque
import gym
import random
import torch
import numpy as np
import torch.optim as optim
import time
import pickle


EARLY_STOPPING_THRESHOLD = 80 # we stop training and immediately save our model when we reach a this average score over the past 100 episodes
INPUT_DIMENSIONS = 4
OUTPUT_DIMENSIONS = 2
MAX_QUEUE_LENGTH = 1000000
EPSILON = 1
EPSILON_DECAY = .999
MIN_EPSILON = .05
EPOCHS =   2000
DISCOUNT_FACTOR = 0.995
TARGET_NETWORK_UPDATE_FREQUENCY = 5000
MINI_BATCH_SIZE = 32
PRETRAINING_LENGTH = 1000





class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIMENSIONS,12)
        self.fc2 = nn.Linear(12,12)
        self.fc3 = nn.Linear (12, OUTPUT_DIMENSIONS)


    def forward(self,x):
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ExperienceReplayBuffer():

    def __init__(self):
        #TODO complete ExperienceReplayBuffer __init__
        #Depends on MAX_QUEUE_LENGTH
        #HINT: use a deque object
    

    def sample_mini_batch(self):
        #TODO complete ExperienceReplayBuffer sample_mini_batch
        #Depends on MINI_BATCH_SIZE


    def append(self,experience):
        #TODO complete ExperienceReplayBuffer append


if __name__ == "__main__":




    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    policy_net = Network()
    target_policy_net = Network()

    target_policy_net.load_state_dict(policy_net.state_dict()) # here we update the target policy network to match the policy network


    env = gym.envs.make("CartPole-v1")
    env.seed(1)
    env.action_space.np_random.seed(1)


    queue = ExperienceReplayBuffer()

    optimizer = optim.Adam(policy_net.parameters(), lr=.001)

    step_counter = 0

    episode_reward_record = deque(maxlen=100)


    for i in range(EPOCHS):
        episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
           
           #TODO collect experience sample and add to experience replay buffer

            if step_counter >= PRETRAINING_LENGTH:
                experience = queue.sample_mini_batch()
               
                #TODO Fill in the missing code to perform a Q update on 'policy_net'
                #for the sampled experience minibatch 'experience'
                
                loss = functional.smooth_l1_loss(estimate,y_vector)
                loss.backward()
                optimizer.step()

            if step_counter % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                target_policy_net.load_state_dict(policy_net.state_dict()) # here we update the target policy network to match the policy network
            step_counter += 1

        EPSILON = EPSILON * EPSILON_DECAY
        if EPSILON < MIN_EPSILON:
            EPSILON = MIN_EPSILON

        if i%100 ==0 and i>0:
            last_100_avg = sum(list(episode_reward_record))/100
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(last_100_avg))
            print("EPSILON: " +  str(EPSILON))
            if last_100_avg > EARLY_STOPPING_THRESHOLD:
                break

    
    torch.save(policy_net.state_dict(), "DQN.mdl")
    pickle.dump([EPSILON], open("DQN_DATA.pkl",'wb'))






