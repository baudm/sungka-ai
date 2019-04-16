import sys
# from gym.envs.toy_text import discrete
from gym import Env, spaces
from gym.utils import seeding
import numpy as np


class SungkaEnv(Env):
    def __init__(self):
        self.shape = (2, 8)
        # Set to 7
        self.board = np.ones(np.prod(self.shape), dtype=np.int8)*7
        # self.board = np.arange(np.prod(self.shape), dtype=np.int32)
        # Set scores to 0
        self.p1_score_ind = 7
        self.p2_score_ind = 15
        self.board[self.p1_score_ind] = 0
        self.board[self.p2_score_ind] = 0

        # The action space is the whole board except the holes for the player scores
        # Player 1's action space is from 0 to 6, while Player 2's is 7 to 13.
        # Player 2's actions will be shifted from {7-13} to {8-14}.
        action_shape = (2, 7)
        self.action_space = spaces.Discrete(np.prod(action_shape))
        # The observation space is just the board itself
        self.observation_space = spaces.Box(0, self.board.sum(), action_shape, dtype=np.int8)

        self.is_done = np.zeros(self.num_states)

        self.seed()
        self.lastaction=None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Reset environment state
        """
        # Set to 7
        self.board = np.ones(np.prod(self.shape), dtype=np.int32)*7
        # Set scores to 0
        self.board[self.p1_score_ind] = 0
        self.board[self.p2_score_ind] = 0

        self.lastaction=None
        return self.board

    def step(self, a):
        """
        Perform specified action
        follow Sungka rules
        returns (state, reward, is_done, prob)
        """
        if a == 7 or a == 15:
            print('Invalid Move!')
        if a < 8:
            player = 1
        else:
            player = 2


        while True:
            # if empty
            if self.board[a] == 0:
                break
            stones = self.board[a]
            self.board[a] = 0 # take all stones
            stone_batch = 0
            if stones >= 15:
                stone_batch = stones//15
                stones %= 15

            if stone_batch:
                self.board += stone_batch
                if player == 1:
                    self.board[self.p2_score_ind] -= stone_batch
                elif player == 2:
                    self.board[self.p1_score_ind] -= stone_batch

            next = a+1
            last = a+stones
            ind = np.arange(next,last+1)
            ind[ind>15] -= 16
            p1_score = np.argwhere(ind==self.p1_score_ind)
            p2_score = np.argwhere(ind==self.p2_score_ind)
            # print(ind)
            # print(p1_score, p2_score)

            # Don't add on opponent score
            while p1_score and player == 2:
                ind = np.delete(ind, p1_score)
                last = last+1
                ind = np.append(ind, last)
                p1_score = np.argwhere(ind==self.p1_score_ind)
                ind[ind>=16] -= 16
            while p2_score and player == 1:
                ind = np.delete(ind, p2_score)
                last = last+1
                ind = np.append(ind, last)
                p2_score = np.argwhere(ind==self.p2_score_ind)
                ind[ind>=16] -= 16
            print(ind)

            # Distribute stones
            self.board[ind] += 1
            a = ind[-1]

            self.render()
            if self.board[a] == 1:
                break
            if a==self.p1_score_ind or a==self.p2_score_ind:
                break
            # Distribute stones

            # if (a+1) + stones <= 15:
            #     self.board[(a+1): (a+1) + stones] += 1
            #     a += stones
            # else:
            #     self.board[(a+1): 15] += 1
            #     self.board[0: (a+1) + stones - 15] += 1
            #     a += stones - 15
            #
            # # check if there is still next move
            # if self.board[a] == 1:
            #     # print(self.board[a],'break')
            #     break
            # elif a == 0  or a == 8:
            #     break
        # return (s, r, d, {"prob" : p})

    def render(self):
        print(self.board)
        outfile = sys.stdout
        output = ' ____________________________________________'
        outfile.write(output)

        for row in range(self.shape[0]):
            output = '\n|    |    |    |    |    |    |    |    |    |\n|    '
            outfile.write(output)

            if row == 0:
                for col in range(self.shape[1]-1):
                    output = "| %s " % str(self.board[col]).zfill(2)
                    outfile.write(output)
            elif row == 1:
                for col in range(self.shape[1]-1):
                    output = "| %s " % str(self.board[(self.shape[1]-1)*2-col]).zfill(2)
                    outfile.write(output)


            if row == 0:
                output = '|    |\n| %s |____|____|____|____|____|____|____| %s |' % \
                            (str(self.board[15]).zfill(2), str(self.board[7]).zfill(2))
                outfile.write(output)
            else:
                output = '|    |\n|____|____|____|____|____|____|____|____|____|'
                outfile.write(output)
        outfile.write('\n')


if __name__ == '__main__':

    env = SungkaEnv()
    env.render()
    # env.step(0)
    env.step(1)
    print('p2')
    env.step(9)
