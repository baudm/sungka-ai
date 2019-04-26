import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import math
import argparse
# from tqdm import tqdm

# default hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', help='model save location')
parser.add_argument('--batch_size', default=BATCH_SIZE, help='batch size; default=%i' % BATCH_SIZE)
parser.add_argument('--lr', default=LR, help='learning rate; default=%i' % LR)
parser.add_argument('--gamma', default=GAMMA, help='gamma/discount factor; default=%i' % GAMMA)
parser.add_argument('--eps', default=EPISILO, help='epsilon/exploration coeff; default=%i' % EPISILO)
parser.add_argument('--mem_cap', default=MEMORY_CAPACITY, help='memory capacity; default=%i' % MEMORY_CAPACITY)
FLAGS = parser.parse_args()
save_path = FLAGS.save_path
BATCH_SIZE = FLAGS.batch_size
LR = FLAGS.lr
GAMMA = FLAGS.gamma
EPISILO = FLAGS.eps
MEMORY_CAPACITY = FLAGS.mem_cap

# env = gym.make("CartPole-v0")
# env = env.unwrapped
import SungkaEnv
env = SungkaEnv.SungkaEnv()
NUM_ACTIONS = env.action_space.n//2
NUM_STATES = np.prod(env.observation_space.shape)
print(NUM_STATES)
print(NUM_ACTIONS)
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self, num_states, num_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_states, 128)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(128,256)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(256,num_actions)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        # print(x.shape)
        # x = x.view(-1)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        # print(action_prob.shape)
        return action_prob

class DQN():
    """docstring for DQN"""
    def __init__(self, num_states, num_actions, epsilon):
        super(DQN, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.eval_net, self.target_net = Net(self.num_states, self.num_actions), Net(self.num_states, self.num_actions)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, self.num_states * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.epsilon_start = epsilon
        self.epsilon = epsilon

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= self.epsilon:# greedy policy
            action_value = self.eval_net.forward(state)
            # print("act val",action_value)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,self.num_actions)
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.num_states])
        batch_action = torch.LongTensor(batch_memory[:, self.num_states:self.num_states+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.num_states+1:self.num_states+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-self.num_states:])

        #q_eval
        # print(self.memory.shape)
        # print(batch_state.shape)
        # print('batch act',batch_action.shape)
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def ep_decay(self, EPS_DECAY, steps_done):
        EPS_END = 0.05
        EPS_START = self.epsilon_start
        self.epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        # print(self.epsilon)

def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward

def random_policy(player):
    if player == 1:
        return np.random.randint(0,7)
    elif player == 2:
        return np.random.randint(7,14)


def max_policy(player, board):
    if player == 1:
        return np.argmax(board[0:7])
    elif player == 2:
        return np.argmax(board[7:14]) + 7

def main():
    dqn = DQN(NUM_STATES, NUM_ACTIONS, EPISILO)
    episodes = 1000
    print("Collecting Experience....")
    reward_list = []
    reward_list_mean = []
    plt.ion()
    fig, ax = plt.subplots()
    # for i in tqdm(range(episodes)):
    for i in range(episodes):
        state = env.reset()
        ep_reward = 0
        dqn.ep_decay(episodes, i)
        while True:

            # env.render()
            action = dqn.choose_action(state)
            next_state, reward , done, info = env.step(action)
            # x, x_dot, theta, theta_dot = next_state
            # reward = reward_func(env, next_state)

            # Let player 2 play
            p2_reward = 0
            while info['next_player'] == 2 and not done:
                action2 = random_policy(info['next_player'])
                # action2 = max_policy(info['next_player'], next_state)
                # next_state, _ , done, info = env.step(action2)
                next_state, reward2 , done, info = env.step(action2)
                p2_reward+=reward2
                # state = next_state


            dqn.store_transition(state, action, reward-p2_reward, next_state)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                # if done:
                #     print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))

            if done:
                break

            state = next_state

        r = copy.copy(ep_reward)
        reward_list.append(r)
        reward_list_mean.append(np.mean(reward_list[-10:]))
        ax.set_xlim(0,1000)
        #ax.cla()
        ax.plot(reward_list, 'g-', label='total_loss')
        ax.plot(reward_list_mean, 'r-', label='ema_loss')
        plt.pause(0.001)

        # env.render()
        print("episode: {} , the episode reward is {}, average of last 10 eps is {}".format(i, ep_reward, reward_list_mean[-1]))
        if i % 100 == 0:
            s = save_path + '-' + str(i).zfill(5)
            print("saving model at episode %i in save_path=%s" % (i, s))
            torch.save(dqn, s)


if __name__ == '__main__':
    main()
