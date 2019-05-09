import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
import argparse

# default hyper-parameters
BATCH_SIZE = 128
LR = 1e-5
GAMMA = 0.9
EPISILON = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100
NUM_EPISODES = 10000
NUM_TEST = 100
OPP_POLICY = 'random'

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', help='model save location')
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batch size; default=%i' % BATCH_SIZE)
parser.add_argument('--lr', default=LR, type=float, help='learning rate; default=%i' % LR)
parser.add_argument('--gamma', default=GAMMA, type=float, help='gamma/discount factor; default=%i' % GAMMA)
parser.add_argument('--eps', default=EPISILON, type=float, help='epsilon/exploration coeff; default=%i' % EPISILON)
parser.add_argument('--mem_cap', default=MEMORY_CAPACITY, type=int, help='memory capacity; default=%i' % MEMORY_CAPACITY)
parser.add_argument('--num_episodes', default=NUM_EPISODES, type=int, help='number of episodes; default=%i' % NUM_EPISODES)
parser.add_argument('--num_test', default=NUM_TEST, type=int, help='number of test episodes; default=%i' % NUM_TEST)
parser.add_argument('--opp_policy', default=OPP_POLICY, help='opponent policy during training; default=%s' % OPP_POLICY)
FLAGS = parser.parse_args()
save_path = FLAGS.save_path
BATCH_SIZE = FLAGS.batch_size
LR = FLAGS.lr
GAMMA = FLAGS.gamma
EPISILON = FLAGS.eps
MEMORY_CAPACITY = FLAGS.mem_cap
NUM_EPISODES = FLAGS.num_episodes
NUM_TEST = FLAGS.num_test
OPP_POLICY = FLAGS.opp_policy
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
        # self.optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.epsilon_start = epsilon
        self.epsilon = epsilon

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_net = self.eval_net.to(self.device)
        self.target_net = self.target_net.to(self.device)

    def choose_action(self, player_id, state, epsilon=None):
        # Swap houses of player 1 and 2
        if player_id == 2:
            state = state[list(range(7, 14)) + list(range(0, 7))]

        state = torch.as_tensor(state, dtype=torch.float, device=self.device)
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() > epsilon:
            # Exploit
            with torch.no_grad():
                action_value = self.eval_net(state).cpu().numpy()
            action = action_value.argmax()
        else:
            # Explore
            action = np.random.randint(0, self.num_actions)

        # Shift action to Player 2's action space
        if player_id == 2:
            action += 7

        return action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):
        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.as_tensor(batch_memory[:, :self.num_states], dtype=torch.float, device=self.device)
        batch_action = torch.as_tensor(batch_memory[:, self.num_states:self.num_states+1], dtype=torch.long, device=self.device)
        batch_reward = torch.as_tensor(batch_memory[:, self.num_states+1:self.num_states+2], dtype=torch.float, device=self.device)
        batch_next_state = torch.as_tensor(batch_memory[:, -self.num_states:], dtype=torch.float, device=self.device)

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        with torch.no_grad():
            q_next = self.target_net(batch_next_state)
        q_target = batch_reward + GAMMA * q_next.max(1, keepdim=True)[0]
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def ep_decay(self, EPS_DECAY, steps_done):
        EPS_END = 0.05
        EPS_START = self.epsilon_start
        self.epsilon = EPS_END + (EPS_START - EPS_END) * (1 - steps_done / EPS_DECAY)
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
        # print(board)
        # print(board[8:15])
        # print(np.argmax(board[7:15]) + 7)

        return np.argmax(board[7:15]) + 7

def exact_policy(player, board):
    if player == 2: # if player 2, mirror the board first
        mirror_board = board[7:14]
        mirror_board = np.append(mirror_board, board[0:7])
        board = mirror_board

    # perform exact policy
    for i in reversed(range(7)):
        if board[i] == 7 - (i):
            action = i
            break
        else:
            action = np.argmax(board[0:7])

    if player == 1:
        return action
    elif player == 2:
        return action + 7


def choose_action(policy, player_id, state, net, eps=None):
    if policy == 'random':
        action = random_policy(player_id)
    elif policy == 'max':
        action = max_policy(player_id, state)
    elif policy == 'exact':
        action = exact_policy(player_id, state)
    elif policy == 'self':
        action = net.choose_action(player_id, state, eps)
    return action


def train_ep(net, policy, render=False):
    state = env.reset()
    ep_reward = 0
    ctr = 0
    if render:
        env.render()
    while True:
        # env.render()
        action = choose_action('self', 1, state, net, 0.05)
        next_state, reward , done, info = env.step(action)
        if render:
            print('Player 1 moves', action)
            env.render()

        # Let player 2 play
        p2_reward = 0
        while info['next_player'] == 2 and not done:
            action2 = choose_action(policy, 2, next_state, net, 0.05)
            next_state, reward2 , done, info = env.step(action2)
            if render:
                print('Player 2 moves', action2)
                env.render()
            p2_reward+=reward2


        net.store_transition(state, action, reward - p2_reward, next_state)
        ep_reward += reward

        if net.memory_counter >= MEMORY_CAPACITY:
            net.learn()

        if done:
            ctr = 0
            break
        ctr += 1

        state = next_state
    if ep_reward > 49:
        win = 1
    else:
        win = 0
    return ep_reward, win


def test_ep(net, policy, num_test, eps=0.05, render=False):

    test_reward = []
    test_win = 0

    for i in range(num_test):
        state = env.reset()
        ep_reward = 0
        if render:
            env.render()
        while True:
            # env.render()
            action = choose_action('self', 1, state, net, eps)
            next_state, reward , done, info = env.step(action)
            if render:
                print('Player 1 moves', action)
                env.render()

            p2_reward = 0
            while info['next_player'] == 2 and not done:
                action2 = choose_action(policy, 2, next_state, net, eps)
                next_state, reward2 , done, info = env.step(action2)
                if render:
                    print('Player 2 moves', action2)
                    env.render()
                p2_reward+=reward2
                # state = next_state

            ep_reward += reward
            if done:
                break

            state = next_state

        test_reward.append(ep_reward)
        if ep_reward > 49:
            test_win+=1

    return np.mean(test_reward), test_win/num_test


def train_ep_p2(net, policy, render=False):
    state = env.reset()
    ep_reward = 0
    ctr = 0
    if render:
        env.render()
    while True:
        # env.render()
        if ctr > 0: # skip player1's first turn so that he goes second
            action = choose_action('self', 1, state, net, 0.05)
            next_state, reward , done, info = env.step(action)
            if render:
                print('Player 1 moves', action)
                env.render()
        else:
            next_state = state
            reward = 0
            done = False
            info = {'next_player': 2}

        # Let player 2 play
        p2_reward = 0
        while info['next_player'] == 2 and not done:
            action2 = choose_action(policy, 2, next_state, net, 0.05)
            next_state, reward2 , done, info = env.step(action2)
            if render:
                print('Player 2 moves', action2)
                env.render()
            p2_reward+=reward2

        if ctr == 0:
            ctr+=1
            continue


        net.store_transition(state, action, reward - p2_reward, next_state)
        ep_reward += reward

        if net.memory_counter >= MEMORY_CAPACITY:
            net.learn()

        if done:
            ctr = 0
            break
        ctr += 1

        state = next_state
    if ep_reward > 49:
        win = 1
    else:
        win = 0
    return ep_reward, win


def test_ep_p2(net, policy, num_test, eps=0.05, render=False):

    test_reward = []
    test_win = 0

    for i in range(num_test):
        state = env.reset()
        ep_reward = 0
        ctr = 0
        if render:
            env.render()
        while True:
            # env.render()
            if ctr > 0: # skip player1's first turn so that he goes second
                action = choose_action('self', 1, state, net, eps)
                next_state, reward , done, info = env.step(action)
                if render:
                    print('Player 1 moves', action)
                    env.render()
            else:
                next_state = state
                reward = 0
                done = False
                info = {'next_player': 2}


            p2_reward = 0
            while info['next_player'] == 2 and not done:
                action2 = choose_action(policy, 2, next_state, net, eps)
                next_state, reward2 , done, info = env.step(action2)
                if render:
                    print('Player 2 moves', action2)
                    env.render()
                p2_reward+=reward2
                # state = next_state

            ep_reward += reward
            if done:
                break

            state = next_state
            ctr+=1

        test_reward.append(ep_reward)
        if ep_reward > 49:
            test_win+=1

    return np.mean(test_reward), test_win/num_test


def main_p2():
    dqn = DQN(NUM_STATES, NUM_ACTIONS, EPISILON)
    episodes = NUM_EPISODES
    print("Collecting Experience....")
    reward_list = []
    reward_list_mean = []
    win_list = []
    win_list_mean = []
    test_reward_rand = []
    test_win_rand = []
    test_reward_max = []
    test_win_max = []
    test_reward_self = []
    test_win_self = []
    test_reward_exact = []
    test_win_exact = []
    plt.ion()
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    for i in range(episodes):
        dqn.ep_decay(episodes, i)
        ep_reward, ep_win = train_ep_p2(dqn, OPP_POLICY)

        r = copy.copy(ep_reward)
        reward_list.append(r)
        reward_list_mean.append(np.mean(reward_list[-10:]))
        win_list.append(ep_win)
        win_list_mean.append(np.mean(win_list[-10:]))
        ax.set_xlim(0,episodes)
        print("episode: {} , the episode reward is {}, average of last 10 eps is {}, win = {}, win_mean = {}".format(i, ep_reward, reward_list_mean[-1], win_list[-1], win_list_mean[-1]))
        if i % 100 == 99 or i ==0:
            if i < episodes - 50:
                test_epsilon = 0.05
            else:
                test_epsilon = 1e-2
            t_reward_rand,t_win_rand = test_ep_p2(dqn, 'random', NUM_TEST, test_epsilon)
            t_reward_max,t_win_max = test_ep_p2(dqn, 'max', NUM_TEST, test_epsilon)
            t_reward_self,t_win_self = test_ep_p2(dqn, 'self', NUM_TEST, test_epsilon)
            t_reward_exact,t_win_exact = test_ep_p2(dqn, 'exact', NUM_TEST, test_epsilon)
            # t_reward,t_win = test_ep(dqn, 'max', NUM_TEST)
            print('[random policy] test: {}, test_reward: {}, test_win: {}'.format(i, t_reward_rand, t_win_rand))
            print('[max policy] test: {}, test_reward: {}, test_win: {}'.format(i, t_reward_max, t_win_max))
            print('[self policy] test: {}, test_reward: {}, test_win: {}'.format(i, t_reward_self, t_win_self))
            print('[exact policy] test: {}, test_reward: {}, test_win: {}'.format(i, t_reward_exact, t_win_exact))
            test_reward_rand.append(t_reward_rand)
            test_win_rand.append(t_win_rand*100)
            test_reward_max.append(t_reward_max)
            test_win_max.append(t_win_max*100)
            test_reward_self.append(t_reward_self)
            test_win_self.append(t_win_self*100)
            test_reward_exact.append(t_reward_exact)
            test_win_exact.append(t_win_exact*100)

            # SAVE
            s = save_path + '-p2-' + str(i).zfill(5)
            print("saving model at episode %i in save_path=%s" % (i, s))
            torch.save(dqn, s)
            np.savez(s+'-test_reward_rand', test_reward_rand)
            np.savez(s+'-test_win_rand', test_win_rand)
            np.savez(s+'-test_reward_max', test_reward_max)
            np.savez(s+'-test_win_max', test_win_max)
            np.savez(s+'-test_reward_max', test_reward_self)
            np.savez(s+'-test_win_max', test_win_self)
            np.savez(s+'-test_reward_max', test_reward_exact)
            np.savez(s+'-test_win_max', test_win_exact)
            np.savez(s+'-train_reward', reward_list)
            np.savez(s+'-train_reward_mean', reward_list_mean)


            # PLOT
            fig.suptitle('[Train] Score over Number of Episodes')
            ax.set_xlabel('Number of Training Episodes')
            ax.set_ylabel('Score')
            ax.plot(reward_list, 'g-', label='current')
            ax.plot(reward_list_mean, 'r-', label='ema')
            # ax.plot(np.array(win_list_mean)*100, 'b-', label='win_rate')

            fig2.suptitle('[Test] Score over Number of Episodes')
            ax2.set_xlabel('Number of Training Episodes (Hundreds)')
            ax2.set_ylabel('Mean Score')
            ax2.plot(test_reward_rand, 'r-', label='vs random policy')
            ax2.plot(test_reward_max, 'g-', label='vs max policy')
            ax2.plot(test_reward_self, 'b-', label='vs self policy')
            ax2.plot(test_reward_exact, 'y-', label='vs exact policy')

            fig3.suptitle('[Test] Win rate of agent')
            ax3.set_xlabel('Number of Training Episodes (Hundreds)')
            ax3.set_ylabel('Win rate (%)')
            ax3.plot(test_win_rand, 'r-', label='vs random policy')
            ax3.plot(test_win_max, 'g-', label='vs max policy')
            ax3.plot(test_win_self, 'b-', label='vs self policy')
            ax3.plot(test_win_exact, 'y-', label='vs exact policy')
            plt.pause(0.001)

            if i == 0:
                fig.legend(loc='lower right', bbox_to_anchor=(0.9, 0.11))
                fig2.legend(loc='lower right', bbox_to_anchor=(0.9, 0.11))
                fig3.legend(loc='lower right', bbox_to_anchor=(0.9, 0.11))

    fig.savefig('p2-train-rewards.png')
    fig2.savefig('p2-test-rewards.png')
    fig3.savefig('p2-win-rates.png')

    print('vs. random policy')
    _,_ = test_ep_p2(dqn, 'random', 1, 1e-2, render=True)
    print('vs. max policy')
    _,_ = test_ep_p2(dqn, 'max', 1, 1e-2, render=True)
    print('vs. self')
    _,_ = test_ep_p2(dqn, 'self', 1, 1e-2, render=True)


def main():
    dqn = DQN(NUM_STATES, NUM_ACTIONS, EPISILON)
    episodes = NUM_EPISODES
    print("Collecting Experience....")
    reward_list = []
    reward_list_mean = []
    win_list = []
    win_list_mean = []
    test_reward_rand = []
    test_win_rand = []
    test_reward_max = []
    test_win_max = []
    test_reward_self = []
    test_win_self = []
    test_reward_exact = []
    test_win_exact = []
    plt.ion()
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    for i in range(episodes):
        dqn.ep_decay(episodes, i)
        ep_reward, ep_win = train_ep(dqn, OPP_POLICY)

        r = copy.copy(ep_reward)
        reward_list.append(r)
        reward_list_mean.append(np.mean(reward_list[-10:]))
        win_list.append(ep_win)
        win_list_mean.append(np.mean(win_list[-10:]))
        ax.set_xlim(0,episodes)
        print("episode: {} , the episode reward is {}, average of last 10 eps is {}, win = {}, win_mean = {}".format(i, ep_reward, reward_list_mean[-1], win_list[-1], win_list_mean[-1]))
        if i % 100 == 99 or i ==0:
            if i < episodes - 50:
                test_epsilon = 0.05
            else:
                test_epsilon = 1e-2
            t_reward_rand,t_win_rand = test_ep(dqn, 'random', NUM_TEST, test_epsilon)
            t_reward_max,t_win_max = test_ep(dqn, 'max', NUM_TEST, test_epsilon)
            t_reward_self,t_win_self = test_ep(dqn, 'self', NUM_TEST, test_epsilon)
            t_reward_exact,t_win_exact = test_ep(dqn, 'exact', NUM_TEST, test_epsilon)
            # t_reward,t_win = test_ep(dqn, 'max', NUM_TEST)
            print('[random policy] test: {}, test_reward: {}, test_win: {}'.format(i, t_reward_rand, t_win_rand))
            print('[max policy] test: {}, test_reward: {}, test_win: {}'.format(i, t_reward_max, t_win_max))
            print('[self policy] test: {}, test_reward: {}, test_win: {}'.format(i, t_reward_self, t_win_self))
            print('[exact policy] test: {}, test_reward: {}, test_win: {}'.format(i, t_reward_exact, t_win_exact))
            test_reward_rand.append(t_reward_rand)
            test_win_rand.append(t_win_rand*100)
            test_reward_max.append(t_reward_max)
            test_win_max.append(t_win_max*100)
            test_reward_self.append(t_reward_self)
            test_win_self.append(t_win_self*100)
            test_reward_exact.append(t_reward_exact)
            test_win_exact.append(t_win_exact*100)

            # SAVE
            s = save_path + '-p1-' + str(i).zfill(5)
            print("saving model at episode %i in save_path=%s" % (i, s))
            torch.save(dqn, s)
            np.savez(s+'-test_reward_rand', test_reward_rand)
            np.savez(s+'-test_win_rand', test_win_rand)
            np.savez(s+'-test_reward_max', test_reward_max)
            np.savez(s+'-test_win_max', test_win_max)
            np.savez(s+'-test_reward_max', test_reward_self)
            np.savez(s+'-test_win_max', test_win_self)
            np.savez(s+'-test_reward_max', test_reward_exact)
            np.savez(s+'-test_win_max', test_win_exact)
            np.savez(s+'-train_reward', reward_list)
            np.savez(s+'-train_reward_mean', reward_list_mean)


            # PLOT
            fig.suptitle('[Train] Score over Number of Episodes')
            ax.set_xlabel('Number of Training Episodes')
            ax.set_ylabel('Score')
            ax.plot(reward_list, 'g-', label='current')
            ax.plot(reward_list_mean, 'r-', label='ema')
            # ax.plot(np.array(win_list_mean)*100, 'b-', label='win_rate')

            fig2.suptitle('[Test] Score over Number of Episodes')
            ax2.set_xlabel('Number of Training Episodes (Hundreds)')
            ax2.set_ylabel('Mean Score')
            ax2.plot(test_reward_rand, 'r-', label='vs random policy')
            ax2.plot(test_reward_max, 'g-', label='vs max policy')
            ax2.plot(test_reward_self, 'b-', label='vs self policy')
            ax2.plot(test_reward_exact, 'y-', label='vs exact policy')

            fig3.suptitle('[Test] Win rate of agent')
            ax3.set_xlabel('Number of Training Episodes (Hundreds)')
            ax3.set_ylabel('Win rate (%)')
            ax3.plot(test_win_rand, 'r-', label='vs random policy')
            ax3.plot(test_win_max, 'g-', label='vs max policy')
            ax3.plot(test_win_self, 'b-', label='vs self policy')
            ax3.plot(test_win_exact, 'y-', label='vs exact policy')
            plt.pause(0.001)

            if i == 0:
                fig.legend(loc='lower right', bbox_to_anchor=(0.9, 0.11))
                fig2.legend(loc='lower right', bbox_to_anchor=(0.9, 0.11))
                fig3.legend(loc='lower right', bbox_to_anchor=(0.9, 0.11))

    fig.savefig('p1-train-rewards.png')
    fig2.savefig('p1-test-rewards.png')
    fig3.savefig('p1-win-rates.png')

    print('vs. random policy')
    _,_ = test_ep(dqn, 'random', 1, 1e-2, render=True)
    print('vs. max policy')
    _,_ = test_ep(dqn, 'max', 1, 1e-2, render=True)
    print('vs. self')
    _,_ = test_ep(dqn, 'self', 1, 1e-2, render=True)


if __name__ == '__main__':
    # Make training reproducible
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()
    main_p2()
    # Pause execution
    input('Press [enter] to exit.')