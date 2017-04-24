import numpy as np
from . import agent
from . import team
from . import flag
from . import command

class World():
    """ World is a the simulation environment for the capture the flag gym 
        environment. It is a container for the teams and flags.
    """
    def __init__(self, height, width, teams, flags=None, 
                 scoring_radius=None, time_to_score=5):
        self.height = height
        self.width = width
        self.teams = teams
        self.team_count = self.teams.size
        self.time_to_score = time_to_score

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
        return teams_obs, flags_obs, self.time

    def timestep(self):
        self.time += 1

    def reset(self):
        """ Resets the world. All flags are returned to original state. World 
                clock is reset.

        Returns:
            Nothing.

        Mutates:
            self.flags - Calls reset method
            self.time  - Zeros
        """
        for f in self.flags:
            f.reset()
            
        self.time = 0

    def create_flags(self):
        """ Generates random flags within the world area. 

        Returns:
            Flags. A list of Flag objects with random locations. None location
                       values determined by world size.
        """
        
        area = self.height * self.width
        num_flags = area / (30 * self.flag_radius * self.flag_radius)
        """ There will be enough flags that, if they are non-overlapping, will
                make 1/10 of the world scoring. 30 ~ 10Pi.
        """

        flags = []
        for i in range(int(num_flags)):
            flags.append(flag.Flag.random_flag(0, 0, self.width, self.height, self.flag_radius))

        return np.array(flags)

    @classmethod
    def to_commands(actions):
        """ Given the actions (as defined by the Gym Env) for each agent, 
               converts them into the Command format. Applies the command to
               all of the agents. 
        """
        return  map ((lambda t : map((lambda a : command.Command.from_action(a)), t)), actions)


    def score_flags(self):
        """ Score and update the flags based on current agent position.

        Returns:
            None

        Mutates:
            self.flags - Each flag will have the most up-to-date scoring status
        """
        for flag in self.flags:
            if flag.taken:
                continue
            max_team_score = 0
            team_id = None
            for team in self.teams:
                team_score = 0
                for agent in team.agents:
                    if flag.within_scoring_distance(agent.loc):
                        team_score += 1
                if team_score > max_team_score:
                    max_team_score = team_score
                    team_id = team.team
            if team_id is not None:
                if team_id != flag.scoring_team:
                    flag.reset()
                    flag.scoring_team = team_id
                flag.scoring_count += 1
                if flag.scoring_count >= self.time_to_score:
                    flag.take(team_id)
        
    def world_step(self, actions):
        """ Convert actions into command format, execute commands, score flags
            appropriately, and move time forward.

        Args:
            actions ([[action]]) - command for every agent.

        Returns:
            None

        Mutates:
            self.teams - applies commands to each member
            self.flags - updates scoring status
            self.time  - increments
        """
        commands = World.to_commands(actions)
        self.apply_commands(commands)
        self.score_flags()
        self.timestep()
    
    def apply_commands(self, commands):
        """ Every agent applies the appropriate inputted command.

        Args:
            commands ([[command]]): A list (one per team) of a list of commands
                                     (one per agent)
        
        Returns:
            Nothing.

        Mutates:
            self.teams - Applies the command to each agent.
        """
        
        self.teams = _zipZip(self.teams, commands, (lambda ac : ac[0].action(ac[1])))
        

    @classmethod
    def _zipZip(arr1, arr2, f=None):
        """ ZipZip zips two 2D arrays to produce a 2D array with each element
                being the result of an elementwise application of f to the 
                input arrays.
            Haskell Implementation
            zipZip :: [[a]] -> [[b]] -> (a -> b -> c) -> [[c]]
            zipZip xs ys f = zipWith (zipWith f) xs ys
        """
        return [zipWith(a,b,f) for (a,b) in zip(arr1, arr2)]

    @classmethod
    def zipWith (arr1, arr2, f=None):
        """ zipWith :: [a] -> [b] -> (a -> b -> c) -> [c]
            Default behavior is zip: f a b = (a,b)
        """
        if f is None:
            return zip(arr1, arr2)

        return [f(a,b) for (a,b) in zip(arr1, arr2)]
