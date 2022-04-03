# Student agent: Add your own agent here
from copy import deepcopy
from agents.agent import Agent
from store import register_agent
import sys

import numpy as np


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)   3-dimentional 
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        best_score = 0
        best_move = [(-1,-1), -1]
        
        my_new_pos = (-1,-1)
        dir_wall = -1

        # Stores all previous moves that we have searched
        previous_moves = [(-1,-1), -1]

        for step_size in range(1, max_step+1) # eg. when moving 4 steps
            for _ in range (0, 2)   # we make this number of different moves for our agent in this step_size
                                    # eg. we take 2 different 4-step moves of the agent as a possibility to search
                for each_step in range (1, step_size+1) # move to each square...

                    # decide a direction to move in
                    dir_move = np.random.randint(0, 4)
                    
                    # we place a wall only on the last step:
                    if each_step == step_size:
                        dir_wall = np.random.randint(0,4)

                    while not self.world.check_valid_step(my_pos, my_pos + moves[dir_move], dir_wall):
                        
                        # generate a valid next step:
                        dir_move = np.random.randint(0,4)
                        
                        # we place a wall only on the last step:
                        if each_step == step_size:
                            dir_wall = np.random.randint(0,4)

                            # If we have already checked this move, do new move... (back to while)
                            if (dir_move, dir_wall) in previous_moves:
                                dir_move = (-1,-1)
                                dir_wall = -1

                    my_new_pos = my_pos + moves[dir_move]
                    r, c = my_new_pos
                    if each_step == step_size:
                        self.world.set_barrier(r, c, dir_wall)  # ! and to unset_barrier at the same position later?
                # a new move of step-size steps has been generated here---

                # Run the minimax algo to check where to place our agent is the best
                score = minimax(chess_board, my_new_pos, adv_pos, False, 10)
                
                # Get the optimal ? position and wall direction
                r, c = my_new_pos
                self.world.unset_barrier(r, c, dir_wall)
                if score > best_score:
                    best_score = score
                    best_move = my_new_pos
                    best_dir_wall = dir_wall
                
        return best_move, best_dir_wall


    def minimax(chess_board, my_new_pos, adv_pos, is_maximizing, depth):
        """
        is_maximizing is a bool indicating whether we are at maximizing or minimizing step
        depth is the depth of search
        """
        
        return false
    # M x M board. max_step = floor((M+1)/2)
    # for each node in the search tree, it has children for every possible step number : 
    # some children after moving 1 step, moving 2 steps, ... 5 steps max.
    # to reduce the branching factor of the search tree but still cover each possible distance, 
    # we can calculate only 2 positions for each number <= max_step, and 4 positions for the wall.
    
    
    
        