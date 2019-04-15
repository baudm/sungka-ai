import sys
# from gym.envs.toy_text import discrete
from gym import Env, spaces
from gym.utils import seeding
import numpy as np


class SungkaEnv(Env):
    def __init__(self):
        self.shape = (2, 8)
        # Set to 7
        self.board = np.ones(np.prod(self.shape), dtype=np.int32)*7
        # self.board = np.arange(np.prod(self.shape), dtype=np.int32)
        # Set scores to 0
        self.board[0] = 0
        self.board[8] = 0

        self.num_states = self.shape[0]*self.shape[1]
        self.num_actions = 4 # up, down, left, right)

        # Declare action space and observation space
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_states)

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
        self.board = np.ones(self.shape)*7
        # Set scores to 0
        self.board[0] = 0
        self.board[8] = 0

        self.lastaction=None
        return self.board

    def step(self, a):
        """
        Perform specified action
        returns (state, reward, is_done, prob)
        """
        return (s, r, d, {"prob" : p})

    def render(self):
        print(self.board)
        outfile = sys.stdout
        output = ' ____________________________________________'
        outfile.write(output)

        for row in range(self.shape[0]):
            output = '\n|    |    |    |    |    |    |    |    |    |\n|    '
            outfile.write(output)

            if row == 0:
                for col in range(1,self.shape[1]):
                    output = "| %s " % str(self.board[row*self.shape[1] + col]).zfill(2)
                    outfile.write(output)
            elif row == 1:
                for col in range(1,self.shape[1]):
                    output = "| %s " % str(self.board[(row+1)*self.shape[1] - col]).zfill(2)
                    outfile.write(output)


            if row == 0:
                output = '|    |\n| %s |____|____|____|____|____|____|____| %s |' % \
                            (str(self.board[0]).zfill(2), str(self.board[8]).zfill(2))
                outfile.write(output)
            else:
                output = '|    |\n|____|____|____|____|____|____|____|____|____|'
                outfile.write(output)
        outfile.write('\n')


if __name__ == '__main__':

    env = SungkaEnv()
    env.render()
