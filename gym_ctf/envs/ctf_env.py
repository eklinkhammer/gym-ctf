import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math

from ..state import agent
class CtfEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        """ Using single agent env, all mutliagent parameters are fixed for now
        """

        self.world_height = 100
        self.world_width = 100
        self.num_teams = 2
        self.number_agents_per_team = np.array([5, 5])
        self.number_flags = 10
        self.flag_radius = 2
        self.time_to_score = 5
        
        self.create_teams()

        self.set_observation_space()
        self.set_action_space()
        
        self.action_
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        
        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])
        
        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high)

        self.position = -1
        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action-1)*0.001 + math.cos(3*position)*(-0.0025)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position)
        reward = -1.0

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            clearance = 10

            robot = agent.Agent(np.array([0,0]),4.2)
            car = rendering.FilledPolygon(robot.triangle(30))
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)

        pos = self.state[0]
        print (pos)
        self.position = self.position + 0.1
        self.cartrans.set_translation((self.position-self.min_position)*scale, 200)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def create_teams(self):
        pass

    def set_observation_space():
        self.origin_x = 0
        self.origin_y = 0

        self.agent_low = np.array([self.origin_x, self.origin_y, 0])
        self.agent_high = np.array([self.world_width, self.world_height,
                                    2*math.pi])
        self.flag_low = np.array([self.origin_x, self.origin_y])
        self.flag_high = np.array([self.world_width, self.world_height])
        
        self.agent_box = spaces.Box(self.agent_low, self.agent_high)
        self.flag_box  = spaces.Box(self.flag_low, self.flag_high)
        self.team_discrete = spaces.Discrete(self.num_teams + 1)

        self.agent_obs = spaces.Tuple((self.agent_box, self.team_discrete))
        self.flag_obs = spaces.Tuple((self.flag_box, self.team_discrete))

        self.all_obs = []
        for _ in range(np.sum(self.number_agents_per_team)):
            self.all_obs.append(self.agent_obs)

        for _ in range(self.number_flags):
            self.all_obs.append(self.flag_obs)

        self.observation_space = spaces.Tuple(self.all_obs)

    def set_action_space():
        self.vector_box = spaces.Box(np.array([-1,-1]), np.array([1,1]))
        self.agent_action = spaces.Tuple((spaces.Discrete(1), # action type
                                          self.vector_box))   # move action

        self.all_actions = []

        for _ in range(np.sum(self.number_agents_per_team)):
            self.all_actions.append(self.agent_action)

        self.action_space = spaces.Tuple(self.all_actions)
