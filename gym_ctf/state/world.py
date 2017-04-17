import numpy as np
import agent
import team

class World():
    def __init__(self, height, width, teams, flags=None, scoring_radius=None, time_to_score=None):
        self.height = height
        self.width = width
        self.teams = teams
        self.team_count = self.teams.size

        if scoring_radius is None:
            scoring_radius = height*width*0.01

        self.flag_radius = scoring_radius
        
        if flags is None:
            flags = self.create_flags()

        self.flags = flags
        self.flag_count = self.flags.size

        self.time = 0

    def get_observation(self):
        """ Returns the observation (in the format expected by the Gym Env)

        Returns:
            (teams (list of teams), flags (list of flags), and time (int))
            Teams is all agents in the world, organized by team.
            Flags is all flags, and their scoring status.
            Time is the current play time of the game.
        """
        teams_obs = map(obs, self.teams)
        flags_obs = map((lambda f : f.obs()), self.flags)
        return self.teams.obs(), self.flags, self.time

    def timestep(self):
        self.time += 1

    def reset(self):
        self.time = 0

    def create_flags(self):
        """ Generates random flags within the world area. 

        Returns:
            Flags. A list of Flag objects with random locations. None location
                       values determined by world size.
        """
        
        area = self.height * self.width
        num_flags = area / (30 * self.flag_radius * self.flag_radius)
        """ There will be enough flags that, if they are non-overlapping, will make 1/10 of
                the world scoring. 30 ~ 10Pi.
        """

        flags = []
        for i range(int(num_flags)):
            flags.append(Flag.random_flag(0, 0, self.width, self.height, self.scoring_radius))

        return flags

    def apply_actions(self, actions):
        """ Given the actions (as defined by the Gym Env) for each agent, 
               converts them into the Command format. Applies the command to
               all of the agents. Scores flags appropriately, and advances
               the game clock
        """
        pass
    
    def apply_commands(self, commands):
        """ Every agent applies the appropriate inputted command.

        Args:
            param1 ([[command]]): A list (one per team) of a list of commands
                                     (one per agent)
        
        Returns:
            None: Applies the command to each agent. Modifies World's teams.
        """
        
        self.teams = _zipZip(self.teams, commands, (lambda (a,c) : a.action(c)))
        

    def _zipZip(arr1, arr2, f=None):
        """ ZipZip zips two 2D arrays to produce a 2D array with each element
                being the result of an elementwise application of f to the 
                input arrays.
            Haskell Implementation
            zipZip xs ys f = zipWith (zipWith f) xs ys
        """
        return [zipWith(a,b,f) for (a,b) in zip(arr1, arr2)]

    def zipWith (arr1, arr2, f=None):
        """ zipWith :: [a] -> [b] -> (a -> b -> c) -> [c]
            Default behavior is zip: f a b = (a,b)
        """
        if f is None:
            return zip(arr1, arr2)

        return [f(a,b) for (a,b) in zip(arr1, arr2)]
