import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math

from ..state import agent
from ..state import flag
from ..state import world
from ..state import team

class CtfEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    COLOR = {1 : [255, 0, 0],
             2 : [0, 255, 0]}
    
    def __init__(self):
        """ Using single agent env, all mutliagent parameters are fixed for now
        """

        self.world_height = 100
        self.world_width = 100
        self.num_teams = 2
        self.number_agents_per_team = np.array([4,4])
        self.number_flags = 10
        self.flag_radius = 2
        self.time_to_score = 5
        self.time_limit = 10

        self.set_observation_space()
        self.set_action_space()
        
        self.create_teams()
        self.create_world()
        
        self.viewer = None

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        """ Execute actions. Returns new world state, reward, and done status.
                Reward is a vector (not a float). This means it cannot be processed
                directly by Open AI learning tools.

        Args:
            action (list of actions): A list of Tuples of action type and vector
        
        Returns:
            observation (location, heading, and team of all agents. location 
                status of all flags. Current time)
            reward (np array of floats): reward per team. 
               NOTE NOT A FLOAT. NOT WHAT OPENAI EXPECTS
            done (boolean): whether epoch is done
            _ (?): unknown
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        team_actions = self.to_team_actions(action)
            
        self.world.world_step(team_actions)
        done = bool(self.world.time >= self.time_limit)

        reward = self.world.get_reward()

        return self.world.get_observation(), reward, done, {}

    def _reset(self):
        self.world.reset()
        return self.world.get_observation()

    def _render(self, mode='human', close=False):
        """ Render incomplete, but entirely functional. Shouldn't have to reopen
            viewer. Transform translation not working. Causes jumpy behavior
        """
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        scale_w = screen_width / self.world_width
        scale_h = screen_height / self.world_height

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            clearance = 1

            i = 0
            self.robot_trans = {}
            # self.robot_locs = {}
            # self.robot_heading = {}
            for t in self.world.teams:
                c = self.COLOR[t.team]
                for a in t:
                    car_r = rendering.FilledPolygon(a.triangle(b=20, scale_x=scale_w, scale_y=scale_h))
                    cartrans = rendering.Transform()
                    car_r.add_attr(cartrans)
                    car_r.set_color(c[0], c[1], c[2])
                    self.viewer.add_geom(car_r)
                    self.robot_trans[i] = cartrans
                    # self.robot_locs[i] = a.loc
                    # self.robot_heading[i] = a.orientation
                    i = i + 1

            for f in self.world.flags:
                flag_r = rendering.make_circle()
                flag_r.set_color(128, 0, 128)
                x = f.position[0] * scale_w
                y = f.position[1] * scale_h
                flag_r.add_attr(rendering.Transform(translation=(x,y)))
                self.viewer.add_geom(flag_r)
            # robot = agent.Agent(np.array([0,0]),4.2)
            # car = rendering.FilledPolygon(robot.triangle(30))
            # car.add_attr(rendering.Transform(translation=(0, clearance)))
            # self.cartrans = rendering.Transform()
            # car.add_attr(self.cartrans)
            # self.viewer.add_geom(car)

        # i = 0
        # for t in self.world.teams:
        #     for a in t:
        #         print ("Step")
        #         # print (self.robot_locs[i])
        #         print (a.loc)
        #         print (a.orientation)
        #         # dx = a.loc[0] - self.robot_locs[i][0]
        #         # dy = a.loc[1] - self.robot_locs[i][1]
        #         # dt = a.orientation - self.robot_heading[i]
        #         # self.robot_locs[i] = a.loc
        #         self.robot_trans[i].set_translation(a.loc[0]*scale_w, a.loc[1]*scale_h)
        #         self.robot_trans[i].set_rotation(a.orientation)
        #         i = i + 1
                
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def to_team_actions(self, actions):
        team_actions = []
        for x in self.number_agents_per_team:
            team_actions.append(actions[:x])
            actions = actions[x:]
        return team_actions
    
    def create_teams(self):
        self.teams = []
        for t in range(self.num_teams):
            agents = []
            for _ in range(self.number_agents_per_team[t]):
                rLoc = flag.Flag.random_pos(self.origin_x, self.origin_y,
                                            self.world_width, self.world_height)
                
                agents.append(agent.Agent(rLoc, 0, t + 1))
            self.teams.append(team.Team(np.array(agents), t + 1))
        self.teams = np.array(self.teams)

    def create_world(self):
        self.world = world.World(self.world_height, self.world_width,
                                 self.teams, None, None, self.number_flags)
        
    def set_observation_space(self):
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

        self.all_obs.append(spaces.Discrete(self.time_limit))
        
        self.observation_space = spaces.Tuple(self.all_obs)

    def set_action_space(self):
        self.vector_box = spaces.Box(np.array([-1,-1]), np.array([1,1]))
        self.agent_action = spaces.Tuple((spaces.Discrete(1), # action type
                                          self.vector_box))   # move action

        self.all_actions = []

        for _ in range(np.sum(self.number_agents_per_team)):
            self.all_actions.append(self.agent_action)

        self.action_space = spaces.Tuple(self.all_actions)
