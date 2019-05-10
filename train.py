#!/usr/bin/env python3

import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

import SungkaEnv
import options

from model import DQN
from policy import choose_action


env = SungkaEnv.SungkaEnv()
NUM_ACTIONS = env.action_space.n // 2
NUM_STATES = np.prod(env.observation_space.shape)
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape


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


def main_p2(opt):
    dqn = DQN(NUM_STATES, NUM_ACTIONS, opt.eps, opt)
    episodes = opt.num_episodes
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
        ep_reward, ep_win = train_ep_p2(dqn, opt.opp_policy)

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
            t_reward_rand,t_win_rand = test_ep_p2(dqn, 'random', opt.num_test, test_epsilon)
            t_reward_max,t_win_max = test_ep_p2(dqn, 'max', opt.num_test, test_epsilon)
            t_reward_self,t_win_self = test_ep_p2(dqn, 'self', opt.num_test, test_epsilon)
            t_reward_exact,t_win_exact = test_ep_p2(dqn, 'exact', opt.num_test, test_epsilon)
            # t_reward,t_win = test_ep(dqn, 'max', opt.num_test)
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
            s = os.path.join(opt.save_path, 'p2-' + str(i).zfill(5))
            print("saving model at episode %i in save_path=%s" % (i, s))
            dqn.save(s + '.pth')
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

    fig.savefig(os.path.join(opt.save_path, 'p2-train-rewards.png'))
    fig2.savefig(os.path.join(opt.save_path, 'p2-test-rewards.png'))
    fig3.savefig(os.path.join(opt.save_path, 'p2-win-rates.png'))

    print('vs. random policy')
    _,_ = test_ep_p2(dqn, 'random', 1, 1e-2, render=True)
    print('vs. max policy')
    _,_ = test_ep_p2(dqn, 'max', 1, 1e-2, render=True)
    print('vs. self')
    _,_ = test_ep_p2(dqn, 'self', 1, 1e-2, render=True)


def main(opt):
    dqn = DQN(NUM_STATES, NUM_ACTIONS, opt.eps, opt)
    episodes = opt.num_episodes
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
        ep_reward, ep_win = train_ep(dqn, opt.opp_policy)

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
            t_reward_rand,t_win_rand = test_ep(dqn, 'random', opt.num_test, test_epsilon)
            t_reward_max,t_win_max = test_ep(dqn, 'max', opt.num_test, test_epsilon)
            t_reward_self,t_win_self = test_ep(dqn, 'self', opt.num_test, test_epsilon)
            t_reward_exact,t_win_exact = test_ep(dqn, 'exact', opt.num_test, test_epsilon)
            # t_reward,t_win = test_ep(dqn, 'max', opt.num_test)
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
            s = os.path.join(opt.save_path, 'p1-' + str(i).zfill(5))
            print("saving model at episode %i in save_path=%s" % (i, s))
            dqn.save(s + '.pth')
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

    fig.savefig(os.path.join(opt.save_path, 'p1-train-rewards.png'))
    fig2.savefig(os.path.join(opt.save_path, 'p1-test-rewards.png'))
    fig3.savefig(os.path.join(opt.save_path, 'p1-win-rates.png'))

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

    Q_NETWORK_ITERATION = 100
    parser = options.get_parser()
    parser.add_argument('--save_path', required=True, help='model save location')
    parser.add_argument('--q_net_iter', default=Q_NETWORK_ITERATION, help='number of iterations before updating target')
    opt = parser.parse_args()

    os.makedirs(opt.save_path, exist_ok=True)

    main(opt)
    main_p2(opt)
    # Pause execution
    input('Press [enter] to exit.')
