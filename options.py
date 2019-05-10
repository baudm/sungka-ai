#!/usr/bin/env python3

import argparse

# default hyper-parameters
BATCH_SIZE = 128
LR = 1e-5
GAMMA = 0.9
EPISILON = 0.9
MEMORY_CAPACITY = 2000
NUM_EPISODES = 10000
NUM_TEST = 100
OPP_POLICY = 'random'


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batch size; default=%i' % BATCH_SIZE)
    parser.add_argument('--lr', default=LR, type=float, help='learning rate; default=%i' % LR)
    parser.add_argument('--gamma', default=GAMMA, type=float, help='gamma/discount factor; default=%i' % GAMMA)
    parser.add_argument('--eps', default=EPISILON, type=float, help='epsilon/exploration coeff; default=%i' % EPISILON)
    parser.add_argument('--mem_cap', default=MEMORY_CAPACITY, type=int, help='memory capacity; default=%i' % MEMORY_CAPACITY)
    parser.add_argument('--num_episodes', default=NUM_EPISODES, type=int, help='number of episodes; default=%i' % NUM_EPISODES)
    parser.add_argument('--num_test', default=NUM_TEST, type=int, help='number of test episodes; default=%i' % NUM_TEST)
    parser.add_argument('--opp_policy', default=OPP_POLICY, help='opponent policy during training; default=%s' % OPP_POLICY)
    return parser
