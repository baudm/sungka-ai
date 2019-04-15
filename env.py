import sys
# from gym.envs.toy_text import discrete
from gym import Env, spaces
from gym.utils import seeding
import numpy as np


class SungkaEnv(Env):
    def __init__(self):
        self.shape = (2, 7)
        self.board = np.ones(self.shape)*7
        self.scores = np.array([0,0])

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
        self.board = np.ones(self.shape)*7

        self.lastaction=None
        return self.board

    def step(self, a):
        """
        Perform specified action
        returns (state, reward, is_done, prob)
        """
        return (s, r, d, {"prob" : p})

    def render(self):
        print(self.scores,self.board)
        outfile = sys.stdout
        output = ' ___________________________________'
        outfile.write(output)

        for row in range(self.board.shape[0]):
            output = '\n|   |   |   |   |   |   |   |   |   |\n|   '
            outfile.write(output)


            for col in range(self.board.shape[1]):
                output = "| %i " % self.board[row, col]
                outfile.write(output)

            if row == 0:
                output = '|   |\n| %i |___|___|___|___|___|___|___| %i |' % (self.scores[0], self.scores[1])
                outfile.write(output)
            else:
                output = '|   |\n|___|___|___|___|___|___|___|___|___|'
                outfile.write(output)

                
if __name__ == '__main__':

    env = SungkaEnv()
    env.render()
