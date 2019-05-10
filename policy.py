#!/usr/bin/env python3

import numpy as np


def random_policy(player):
    if player == 1:
        return np.random.randint(0, 7)
    elif player == 2:
        return np.random.randint(7, 14)


def max_policy(player, board):
    if player == 1:
        return np.argmax(board[0:7])
    elif player == 2:
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


def human_policy(player):
    if player == 1:
        return int(input('Move: '))
    elif player == 2:
        return int(input('Move: ')) + 7


def choose_action(policy, player_id, state, net, eps=None):
    if policy == 'random':
        action = random_policy(player_id)
    elif policy == 'max':
        action = max_policy(player_id, state)
    elif policy == 'exact':
        action = exact_policy(player_id, state)
    elif policy == 'self':
        action = net.choose_action(player_id, state, eps)
    elif policy == 'human':
        action = human_policy(player_id)
    return action
