import gym
from gym import error, spaces, utils
from gym.utils import seeding

class CtfEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init(self):
        self.observation_space = None
        self.action_space = None
        self.reward_range = None
        

    def _step(self,action):
        pass

    def _reset(self):
        pass

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None

        if self.viewer = None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(600, 400)

        return self.viewer.render(return_rgb_array = mode == 'rgb_array')
    
 
