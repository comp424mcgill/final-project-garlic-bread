# Student agent: Add your own agent here
from copy import deepcopy
from operator import is_
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

    def set_barrier(self, chess_board, r, c, dir):
        # Set the barrier to True
        chess_board[r, c, dir] = True
        
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        # Set the opposite barrier to True
        move = moves[dir]
        chess_board[r + move[0], c + move[1], opposites[dir]] = True

    def unset_barrier(self, chess_board, r, c, dir):
        # Set the barrier to False
        chess_board[r, c, dir] = False
       
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
         # Set the opposite barrier to False
        move = moves[dir]
        chess_board[r + move[0], c + move[1], opposites[dir]] = False

       
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

        for step_size in range(1, max_step+1): # eg. when moving 4 steps
            for _ in range (0, 2):   # we make this number of different moves for our agent in this step_size
                                    # eg. we take 2 different 4-step moves of the agent as a possibility to search
                for each_step in range (1, step_size+1): # move to each square...

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
                    previous_moves = previous_moves + [(my_new_pos, dir_wall)]
                    if each_step == step_size:
                        r, c = my_new_pos
                        self.set_barrier(chess_board, r, c, dir_wall)  # ! and to unset_barrier at the same position later?
                
                # a new move of step-size steps has been generated here---

                # Run the minimax algo to check where to place our agent is the best
                score = self.minimax(chess_board, my_new_pos, adv_pos, max_step, False, 10)
                
                # Get the optimal ? position and wall direction
                r, c = my_new_pos
                self.unset_barrier(chess_board, r, c, dir_wall) # ??? this is not right
                if score > best_score:
                    best_score = score
                    best_move = my_new_pos
                    best_dir_wall = dir_wall
                
        return best_move, best_dir_wall

    def minimax(self, chess_board, my_pos, adv_pos, max_step, is_maximizing, depth):
        """
        is_maximizing is a bool indicating whether we are at maximizing or minimizing step
        depth is the depth of search
        """
        # Check base cases:
        is_end, s1, s2 = self.world.check_endgame()
        if is_end & (not self.world.turn): # if we win. Is it correct to use self.world.turn?
            return 1
        elif is_end & self.world.turn: # if the other agent wins
            return 0
        elif is_end & s1==s2: # if a draw
            return 0.5

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        best_score = 0

        my_new_pos = (-1,-1)
        dir_wall = -1

        # Stores all previous moves that we have searched
        previous_moves = [(-1,-1), -1]

        if is_maximizing:

            for step_size in range(1, max_step+1): # eg. when moving 4 steps
                for _ in range (0, 2):   # we make this number of different moves for our agent in this step_size
                                        # eg. we take 2 different 4-step moves of the agent as a possibility to search
                    for each_step in range (1, step_size+1): # move to each square...

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
                        previous_moves = previous_moves + [(my_new_pos, dir_wall)]
                        if each_step == step_size:
                            r, c = my_new_pos
                            self.set_barrier(chess_board, r, c, dir_wall)  # ! and to unset_barrier at the same position later?
                    
                    # a new move of step-size steps has been generated here---

                    # Run the minimax algo to check where to place our agent is the best
                    score = self.minimax(chess_board, my_new_pos, adv_pos, max_step, False, depth+1)
                    
                    # Get the optimal ? position and wall direction
                    r, c = my_new_pos
                    self.unset_barrier(chess_board, r, c, dir_wall)
                    if score > best_score:
                        best_score = score
            return best_score
        
        else:  # if not is_maximizing:

            for step_size in range(1, max_step+1): # eg. when moving 4 steps
                for _ in range (0, 2):   # we make this number of different moves for our agent in this step_size
                                        # eg. we take 2 different 4-step moves of the agent as a possibility to search
                    for each_step in range (1, step_size+1): # move to each square...

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
                        previous_moves = previous_moves + [(my_new_pos, dir_wall)]
                        if each_step == step_size:
                            r, c = my_new_pos
                            self.set_barrier(chess_board, r, c, dir_wall)  # ! and to unset_barrier at the same position later?
                    
                    # a new move of step-size steps has been generated here---

                    # Run the minimax algo to check where to place our agent is the best
                    score = self.minimax(chess_board, my_new_pos, adv_pos, max_step, True, depth+1) # the TRUE here
                    
                    # Get the optimal ? position and wall direction
                    r, c = my_new_pos
                    self.unset_barrier(chess_board, r, c, dir_wall)

                    if score < best_score:  # the LESS THAN here
                        best_score = score
            return best_score

    # M x M board. max_step = floor((M+1)/2)
    # for each node in the search tree, it has children for every possible step number : 
    # some children after moving 1 step, moving 2 steps, ... 5 steps max.
    # to reduce the branching factor of the search tree but still cover each possible distance, 
    # we can calculate only 2 positions for each number <= max_step, and 4 positions for the wall.

    
        