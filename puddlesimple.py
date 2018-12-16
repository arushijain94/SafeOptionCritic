import gym
from gym import core, spaces
from gym.envs.registration import register
import numpy as np
from gym.utils import seeding
import copy

class PuddleSimpleEnv(gym.Env):

    def __init__(self, goal=[1.0, 1.0], goal_threshold=0.1,
            noise=0.025, thrust=0.05, puddle_center=[[.5, .5]],
            puddle_width=[[.3, .3]]):
        self.goal = np.array(goal)
        self.goal_threshold = goal_threshold
        self.noise = noise
        self.thrust = thrust
        self.puddle_center = [np.array(center) for center in puddle_center]
        self.puddle_width = [np.array(width) for width in puddle_width]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,))

        self.actions = [np.zeros(2) for i in range(4)]
        for i in range(4):
            self.actions[i][i//2] = thrust * (i%2 * 2 - 1)

        self._seed()
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        self.pos += self.actions[action] + self.np_random.uniform(low=-self.noise, high=self.noise, size=(2,))
        self.pos = np.clip(self.pos, 0.0, 1.0)

        reward = 0.

        done = np.linalg.norm((self.pos - self.goal), ord=1) < self.goal_threshold

        if done == True:
            reward = 50.0

        return self.pos, reward, done, {}

    def reset(self):
        self.pos = self.observation_space.sample()
        return self.pos
#
register(
    id='PuddleEnv-v0',
    entry_point='puddlesimple:PuddleSimpleEnv',
    timestep_limit=5000,
)

