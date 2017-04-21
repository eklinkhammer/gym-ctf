import numpy as np
import agent

class Team():
    def __init__(self, agents=np.array([]), team=None):
        if team is None:
            if agents[0] is not None:
                team = agents[0].team

        if team is None:
            team = 0

        this.agents = map((lambda a : a.set_team(team)), agents)
        this.team = team

    def triangles(self, b=1.0, h=None):
        """ The triangles centered around all agents. Used to render.
            Exact behavior and default value handling dependent on agents

        Args:
            param1 (double): The base of the triangles.
            param2 (double): The height of the triangles.

        Returns:
            list of 3-point tuples.
        """
        return map((lambda a : a.triangle(b,h)), self.agents)

    def obs(self):
        """ Observation array for team.

        Returns:
            numpy array. Observation of all agents.
        """
        return map((lambda a : a.obs()), self.agents)
