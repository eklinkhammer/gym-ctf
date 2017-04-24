import numpy as np

class Command():
    COMMANDS = {0 : 'MOVE'}
    
    def __init__(self, command_type, command_data):
        self.action_label = COMMANDS[command_type]
        if self.action_label == 'MOVE':
            self.create_move(command_data)

    def create_move(self, command_data):
        self.vector = np.array([command_data[0][0], command_data[0][1]])

    @classmethod
    def from_action(action):
        return Command(action[0], a[1:])
