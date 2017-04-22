import numpy as np

class Agent():
    def __init__(self, location=(0,0), orientation=0, team=1):
        self.loc = location
        self.orientation = orientation
        self.team = team

    @classmethod
    def copy_agent(cls, instance):
        """ Copy constructor for Agent Class.
        
        Provided for completeness. In future, agent may be extended beyond a 
          (x,y,theta) tuple with move command.
        """
        loc = instance.loc
        ori = instance.orientation
        team = instance.team
        return cls(loc, ori, team)
    
    def action(self, command):
        """ Agent implements a command from the action space

        Args:
            param1 (Action): The action the agent should take.

        Returns:
            Agent. New agent with a changed location.
        
        """
        if command.action_label == 'MOVE':
            return self.move(command)
        else:
            return Agent.copy_agent(self)

    def move(self, move_command):
        """ Moves the agent as specified.
        
        Args:
            param1 (Action): Movement action. Specifies dx and dy.
        Returns:
            Agent. New agent with updated location and orientation.
        """
        vec = move_command.vector
        pos = np.add(self.loc, vec)
        ori = np.arctan2(vec[1], vec[0])
        return update_move(self, pos, ori)

    def update_move(self, pos, ori):
        """ Immutable agent position and orientation update. 

        Args:
            param1 (np.array): Position of agent. Agent Class places no 
                               restriction on bounds.
            param2 (double): Orientation of agent. Must be between [0, 2pi)

        Returns:
            Agent. New agent at position and orientation specified. Copies all
                other information as applicable.
        """
        a = Agent.copy_agent(self)
        a.loc = pos
        a.orientation = ori
        return a

    def set_team(self, team):
        """ Immutable setter of team.

        Args:
            param1 (int): New team.
        
        Returns:
            Agent. New instance of an agent, identical except on a new team.
        """
        
        a = Agent.copy_agent(self)
        a.team = team
        return a
    
    def triangle(self, b=1.0, h=None):
        """ Returns the points of a triangle pointed in the direction of
                orientation centered at position.
        
        Args:
            param1 (double): Base of triangle (distance between points 2 and 3 
                                 along direction perpendicular to orientation)
            param2 (double): Distance from base to top point. Colinear with 
                                 orientation. Default value is the same as base.

        Returns:
            Array of Double Tuples: CW points starting from forward point.
        """
        if h is None:
            h = b

        dx = np.cos(self.orientation)
        dy = np.sin(self.orientation)
        top = (self.loc[0] + dx * 0.5 * h, self.loc[1] + dy * 0.5 * h)
        base_midpoint = (self.loc[0] - dx * 0.5 * h, self.loc[1] - dy * 0.5 * h)
        base_line = (dy, -dx)
        base_dx = base_line[0] * 0.5 * b
        base_dy = base_line[1] * 0.5 * b
        pt2 = (base_midpoint[0] + base_dx, base_midpoint[1] + base_dy)
        pt3 = (base_midpoint[0] - base_dx, base_midpoint[1] - base_dy)
        return (top, pt2, pt3)

    def obs(self):
        """ Create observation numpy array.

        Returns:
            numpy array (length 4): X, Y, Theta, Team
        """
        return np.array([self.loc[0], self.loc[1], self.orientation, self.team])
