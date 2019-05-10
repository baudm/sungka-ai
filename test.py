#/usr/bin/env python3

import numpy as np
import torch

import options
from model import DQN
from policy import choose_action

import SungkaEnv
from train import test_ep, test_ep_p2

env = SungkaEnv.SungkaEnv()
NUM_ACTIONS = env.action_space.n // 2
NUM_STATES = np.prod(env.observation_space.shape)


def test_ep_pvp(net, net2, num_test, eps=0.05, render=False):

    test_reward = []
    test_reward_p2 = []
    test_win = 0
    draw = 0

    for i in range(num_test):
        state = env.reset()
        ep_reward = 0
        p2_reward = 0
        ctr = 0
        if render:
            env.render()
        while True:
            # env.render()
            action = choose_action('self', 1, state, net, eps)
            next_state, reward , done, info = env.step(action)
            if render:
                print('Player 1 moves', action)
                env.render()

            while info['next_player'] == 2 and not done:
                action2 = choose_action('self', 2, next_state, net2, eps)
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

        test_reward_p2.append(p2_reward)
        test_reward.append(ep_reward)
        if ep_reward > 49:
            test_win+=1
        if ep_reward == 49:
            draw +=1

    return np.mean(test_reward), np.mean(test_reward_p2), test_win/num_test, draw/num_test


def load_and_play(opt):
    net = DQN(NUM_STATES, NUM_ACTIONS, opt.eps, opt)
    net.load(opt.load_path)
    if opt.player == 1:
        r, w = test_ep(net, opt.opp_policy, opt.num_test, opt.eps, render=opt.render)

    elif opt.player == 2:
        r, w = test_ep_p2(net, opt.opp_policy, opt.num_test, opt.eps, render=opt.render)

    print('average reward:', r)
    print('win rate:', w)


def load_and_test(opt):
    netp1 = DQN(NUM_STATES, NUM_ACTIONS, opt.eps, opt)
    netp1.load(opt.load_path)
    load_path2 = list(opt.load_path)
    print(opt.load_path)
    load_path2[-7] = '2'
    load_path2 = "".join(load_path2)
    print(load_path2)
    netp2 = DQN(NUM_STATES, NUM_ACTIONS, opt.eps, opt)
    netp2.load(load_path2)
    if opt.player == 1:
        r1, r2, w, d = test_ep_pvp(netp1, netp2, opt.num_test, opt.eps, render=opt.render)

        print('p1 average reward:', r1)
        print('p2 average reward:', r2)
        print('p1 win rate:', w)
        print('p2 win rate:', 1-w-d)
        print('draw rate:', d)

    elif opt.player == 2:
        r2, r1, w, d = test_ep_pvp(netp2, netp1, opt.num_test, opt.eps, render=opt.render)

        print('p1 average reward:', r1)
        print('p2 average reward:', r2)
        print('p1 win rate:', 1-w-d)
        print('p2 win rate:', w)
        print('draw rate:', d)


if __name__ == '__main__':
    # Make testing reproducible
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = options.get_parser()
    parser.set_defaults(num_test=1, eps=0.05)
    parser.add_argument('--load_path', required=True, help='model location')
    parser.add_argument('--player', default=1, type=int, help='player turn')
    parser.add_argument('--render', default=False, action='store_true', help='render')
    opt = parser.parse_args()

    load_and_play(opt)
    load_and_test(opt)
