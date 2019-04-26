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
        # self.board[0:7] = 0
        # self.board[5] = 1
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

        # self.is_done = np.zeros(self.num_states)
        self.is_done = False

        self.seed()
        self.lastaction=None

    def action_space(self):
        return self.action_space
    def observation_space(self):
        return self.observation_space
    def is_done(self):
        return self.is_done

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
        s = self.board[0:7]
        s = np.append(s, self.board[8:15])
        return s

    def step(self, a):
        """
        Perform specified action
        follow Sungka rules
        returns (state, reward, is_done, prob)
        """
        if a > 13:
            print('Invalid Move!')

        # a = 0 to 6
        if a < 7:
            player = 1
        # moves: 7-13 -> a = 8 to 14
        else:
            player = 2
            a += 1

        prev_r = [0,0]
        prev_r[0] = self.board[self.p1_score_ind]
        prev_r[1] = self.board[self.p2_score_ind]


        while True:
            # print(np.sum(self.board))
            ctr = 0
            if np.sum(self.board)<98:
                continue
            # print('action:', a)
            # print('player', player)
            # if empty
            if self.board[a] == 0:
                ind = [a]
                # print('ind',ind)
                break

            stones = self.board[a]
            self.board[a] = 0 # take all stones
            stone_batch = 0
            if stones >= 15:
                stone_batch = stones//15
                stones %= 15

            if stone_batch:
                self.board[0:7] += stone_batch
                self.board[8:15] += stone_batch
                if player == 1:
                    self.board[self.p1_score_ind] += stone_batch
                elif player == 2:
                    self.board[self.p2_score_ind] += stone_batch

            if stones == 0:
                continue

            next = a+1
            last = a+stones

            # if the next house is the enemy's score house, adjust
            if (player == 1 and next == self.p2_score_ind) or \
                    (player == 2 and next == self.p1_score_ind):
                next += 1
                last += 1

            ind = np.arange(next,last+1)
            # print('ind before loop', next,last,ind)
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
            # print(ind)

            # Distribute stones
            self.board[ind] += 1
            # set new last action as ind[-1]
            a = ind[-1]
            # print('a',a, 'ind',ind[-1])

            # self.render()

            # SUNOG mechanism
            if self.board[a] == 1 and a != self.p1_score_ind and a != self.p2_score_ind:
                # perform "SUNOG"
                if (player == 1 and a < 7):
                    # get yours
                    self.board[self.p1_score_ind] += self.board[a]
                    # get opposite side
                    self.board[self.p1_score_ind] += self.board[abs(14-a)]

                    # remove stones
                    self.board[a] = 0
                    self.board[abs(14-a)] = 0
                elif (player == 2 and a >= 7):
                    # get yours
                    self.board[self.p2_score_ind] += self.board[a]
                    # get opposite side
                    self.board[self.p2_score_ind] += self.board[abs(14-a)]

                    # remove stones
                    self.board[a] = 0
                    self.board[abs(14-a)] = 0
                # self.render()
                # print('meow')
                # print(ind[-1])
                break
            ### end SUNOG mechanism

            # if last stone goes to the score, stop the loop
            # current player is allowed another turn
            if a==self.p1_score_ind or a==self.p2_score_ind:
                # print('meow')
                # print(ind[-1])
                break

        # Next player info, change to other player after the move
        if player == 1:
            next_player = 2
        elif player == 2:
            next_player = 1

        # Except when last stone goes to score, allow player to move again
        if ind[-1] == self.p1_score_ind:
            next_player = 1
        elif ind[-1] == self.p2_score_ind:
            next_player = 2

        # create state vector
        s = self.board[0:self.p1_score_ind]
        s = np.append(s, self.board[self.p1_score_ind:-2])
        # print("s",s)

        # create reward vector
        if player == 1:
            r = self.board[self.p1_score_ind] - prev_r[0]
        elif player == 2:
            r = self.board[self.p2_score_ind] - prev_r[1]

        # if game is done
        # game is done if all stones are in the scores; board is empty
        if self.board[self.p1_score_ind] + self.board[self.p2_score_ind] == 98:
            d = True
            # self.render()
        else:
            d = False

        p = 1
        # print('d',d )
        return (s, r, d, {"prob" : p, "next_player" : next_player, "p1_score" : self.board[self.p1_score_ind], "p2_score" : self.board[self.p2_score_ind]})

    def render(self):
        # print(self.board)
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
    # env.render()
    # print(env.step(7))
    # env.step(1)
    # print('p2')
    # print(env.step(9))
