import gym
import pickle
import random
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as functional
import torch


class Network(nn.Module):
    def __init__(self):
        INPUT_DIMENSIONS = 4
        OUTPUT_DIMENSIONS = 2
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIMENSIONS,12)
        self.fc2 = nn.Linear(12,12)
        self.fc3 = nn.Linear (12, OUTPUT_DIMENSIONS)


    def forward(self,x):
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def default_Q_value():
    return 0


def evaluate_frozen_lake(Q_table, EPSILON, visualize=False):
    total_reward = 0
    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)


    for i in range(100):
        obs = env.reset()
        done = False
        while done == False:
            if random.uniform(0,1) < EPSILON:
                action = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(obs,i)] for i in range(env.action_space.n)])
                action =  np.argmax(prediction)
            obs,reward,done,info = env.step(action)
            total_reward += reward
            if visualize:
                env.render()
                time.sleep(.01)
    score = total_reward/100
    return score
    


def test_Q_learning(visualize = False):
    loaded_data = pickle.load(open('Q_TABLE.pkl', 'rb'))
    Q_table = loaded_data[0]
    EPSILON = loaded_data[1]
    score = evaluate_frozen_lake(Q_table,EPSILON,visualize = False)
    points = 0
    if score >= 0.6:
        points = 30
    elif score >= 0.4:
        points = 15 
    print("Q Learning on FrozenLake-v0:")
    print("Average episode-reward over 100 episodes is " + str(score))
    print(str(points)  +" out of 30 points received.")
    return points

def test_SARSA(visualize = False):
    loaded_data = pickle.load(open('SARSA_Q_TABLE.pkl', 'rb'))
    Q_table = loaded_data[0]
    EPSILON = loaded_data[1]
    score = evaluate_frozen_lake(Q_table,EPSILON,visualize = False)
    points = 0
    if score >= 0.6:
        points = 30
    elif score >= 0.4:
        points = 15 
    print("SARSA Learning on FrozenLake-v0:")
    print("Average episode-reward over 100 episodes is " + str(score))
    print(str(points)  +" out of 30 points received.")
    return points


def test_DQN(visualize = False):


    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("CartPole-v1")
    env.seed(1)
    env.action_space.np_random.seed(1)


    EPSILON = pickle.load(open('DQN_DATA.pkl', 'rb'))[0]
    policy_net = Network()
    policy_net.load_state_dict(torch.load("DQN.mdl"))

   
    total_reward = 0
    for i in range(100):
        obs = env.reset()
        done = False
        while done == False:
            if random.uniform(0,1) < EPSILON:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    prediction = np.array(policy_net(torch.tensor(np.array(obs)).float()))
                    action =  np.argmax(prediction)
            obs,reward,done,info = env.step(action)
            total_reward += reward
            if visualize:
                env.render()
                time.sleep(.01)
    score = total_reward/100
    extra_credit_points = 0
    if score >= 70:
        extra_credit_points = 10


    print("DQN Learning on CartPole-v1:")
    print("Average episode-reward over 100 episodes is " + str(score))
    print(str(extra_credit_points)  +" out of 10 points received.")
    return extra_credit_points


   


if __name__ == "__main__":
    total_points = 0
    total_extra_credit_points = 0
    print('-' * 40)
    try:
        total_points += test_Q_learning()
    except Exception as e:
        print(e)
    print('-' * 40)
    try:
        total_points += test_SARSA()
    except Exception as e:
        print(e)
    print('-' * 40)
    try:
        total_extra_credit_points += test_DQN()
    except Exception as e:
        print(e)
    print('-' * 40)
    print(str(total_points)  +" out of 60 automated points received.")
    print(str(total_extra_credit_points)  +" out of 10 automated extra credit points received.")
    print('Remaining 40 points subject to manual validation')
