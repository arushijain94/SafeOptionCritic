import numpy as np
from gym import core, spaces
from gym.envs.registration import register




class FourroomsTest(core.Env):
    def __init__(self):

        layout = """\
wwwwwwwwwwwww
w     w     w
w   ffwff   w
w  fffffff  w
w   ffwff   w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        """
        Direction:
        0:U
        1:D
        2:L
        3:R

        Deterministic Actions

        Introducing variable rewards in "frozen"/ "slippery" state in range U[-15, 15] where expected value is zero as another states
        Reward for Goal state : 50
        Reward for Normal state/ hits wall:0
        """

        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])
        self.frozen = np.array([list(map(lambda c: 1 if c=='f' else 0, line)) for line in layout.splitlines()])

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
        self.tocell = {v:k for k,v in self.tostate.items()}
        
        self.goal = 62
        self.init_states = [0, 43]#1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 21, 29, 30, 31, 32, 33, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102, 103]
        #all init states excluding frozen states

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        return state

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/5,
        the agent moves instead in one of the other three directions, each with 1/15 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        """
        reward = 0
        if self.rng.uniform() < 1/5.:
            empty_cells = self.empty_around(self.currentcell)
            nextcell = empty_cells[self.rng.randint(len(empty_cells))]
        else:
            nextcell = tuple(self.currentcell + self.directions[action])
        
        if not self.occupancy[nextcell]:
            self.currentcell = nextcell
        state = self.tostate[self.currentcell]
        

        if self.frozen[self.currentcell]:
            #frozen state
            reward = np.random.uniform(low=-15.0, high=15.0)
        elif state == self.goal:
            reward = 50 #goal state reward
        #remaining cases, reward is 0

        done = state == self.goal
        return state, reward, done, None

register(
    id='Fourrooms-v1',
    entry_point='fourrooms_test:FourroomsTest',
    timestep_limit=500,
)
