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
            min_x (double): Minimum x value (lower-left corner)
            min_y (double): Minimum y value (lower-left corner)
            max_x (double): Maximum x value (upper-right corner)
            max_y (double): Maximum y value (upper-right corner)

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
            min_x (double): Minimum x value (lower-left corner)
            min_y (double): Minimum y value (lower-left corner)
            max_x (double): Maximum x value (upper-right corner)
            max_y (double): Maximum y value (upper-right corner)
            scoring_radius (double): The radius within which agents can ctf

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

    def within_scoring_distance(self, position_other):
        """ Determine if other position is within the scoring radius of the flag

        Args:
            position_other (2-tuple of doubles): 2D point

        Returns:
            boolean. True iff position_other is within scoring radius.
        """
        distance = math.sqrt(math.pow(self.pos[0] - position_other[0], 2) +
                             math.pow(self.pos[1] - position_other[1], 2))
        return distance <= self.scoring_radius
