import numpy as np
import random

class Flag():
    """ A flag is target that agents use to score in capture the flag.
        Once captured, it is marked as taken and stores the scoring team.
    """
    
    def __init__(self, pos, scoring_radius):
        self.position = pos
        self.scoring_radius = scoring_radius
        self.taken = False
        self.scoring_team = None
        self.scoring_count = 0

    def take(self, team_id):
        self.taken = True
        self.scoring_team = team_id

    def reset(self):
        self.taken = False
        self.scoring_team = None
        self.scoring_count = 0

    def random_pos(min_x, min_y, max_x, max_y):
        """ Generates a random tuple representing a 2D point within the box
                defined by the ranges.

        Args:
            param1 (double): Minimum x value (lower-left corner)
            param2 (double): Minimum y value (lower-left corner)
            param3 (double): Maximum x value (upper-right corner)
            param4 (double): Maximum y value (upper-right corner)

        Returns:
            (double, double). 2D point.
        """
        
        if max_y is None: max_y = max_x

        rand_x = random.randrange(min_x, max_x,1)
        rand_y = random.rangrange(min_y, max_y,1)

        position = (rand_x, rand_y)
        return position

    def random_flag(min_x, min_y, max_x, max_y, scoring_radius):
        """ Generates a random flag at a position within the bounding box 
                provided using the given scoring radius. Scoring radius is not
                random because it depends (for best results) on container size.
        
        Args:
            param1 (double): Minimum x value (lower-left corner)
            param2 (double): Minimum y value (lower-left corner)
            param3 (double): Maximum x value (upper-right corner)
            param4 (double): Maximum y value (upper-right corner)
            param5 (double): The radius within which agents capture the flag

        Returns:
            Flag. A flag object at a random 2D point.
        """
        return Flag(Flag.random_pos(min_x, min_y, max_x, max_y), scoring_radius)
        
    def obs(self):
        """ Returns the observation of a flag in format expected by gym env 

        Returns:
            numpy array of length 3. Contains position of flag and scoring
                team. Team is -1 if no team scored.
        """
        if self.taken:
            team = self.scoring_team
        else:
            team = -1
        return np.array([self.position[0], self.position[1], team])
