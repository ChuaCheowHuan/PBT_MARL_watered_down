"""# The environment: RockPaperScissorsEnv class"""

import ray
from gym.spaces import Discrete
from ray.rllib.env import BaseEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv

ROCK = 0
PAPER = 1
SCISSORS = 2

class RockPaperScissorsEnv(MultiAgentEnv):
    """Two-player environment for rock paper scissors.
    The observation is simply the last opponent action."""

    def __init__(self, _, population_size):
    #def __init__(self, population_size):
        self.population_size = population_size
        self.action_space = Discrete(3)
        self.observation_space = Discrete(3)
        self.player_A = None
        self.player_B = None
        #self.player_A = "agt_0"
        #self.player_B = "agt_1"
        self.last_move = None
        self.num_moves = 0

    def reset(self):
        g_helper = ray.get_actor("g_helper")
        agt_i, agt_j = ray.get(g_helper.get_pair.remote())
        self.player_A = agt_i
        self.player_B = agt_j
        self.last_move = (0, 0)
        self.num_moves = 0
        return {
            self.player_A: self.last_move[1],
            self.player_B: self.last_move[0],
        }

    def step(self, action_dict):
        move1 = action_dict[self.player_A]
        move2 = action_dict[self.player_B]
        self.last_move = (move1, move2)
        obs = {
            self.player_A: self.last_move[1],
            self.player_B: self.last_move[0],
        }

        r1, r2 = {
            (ROCK, ROCK): (0, 0),
            (ROCK, PAPER): (-1, 1),
            (ROCK, SCISSORS): (1, -1),
            (PAPER, ROCK): (1, -1),
            (PAPER, PAPER): (0, 0),
            (PAPER, SCISSORS): (-1, 1),
            (SCISSORS, ROCK): (-1, 1),
            (SCISSORS, PAPER): (1, -1),
            (SCISSORS, SCISSORS): (0, 0),
        }[move1, move2]
        rew = {
            self.player_A: r1,
            self.player_B: r2,
        }
        self.num_moves += 1
        done = {
            "__all__": self.num_moves >= 10,
        }

        #print('obs', obs)

        return obs, rew, done, {}
