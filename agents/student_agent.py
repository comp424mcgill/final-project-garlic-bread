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
        self.autoplay = True

    # functions set_barrier(), check_valid_step(), check_endgame() below copied from world.py:
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

    def check_valid_step(self, chess_board, my_start_pos, my_end_pos, barrier_dir, adv_pos, max_step):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Endpoint already has barrier or is boarder
        print(my_end_pos)
        r, c = my_end_pos
        if chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(my_start_pos, my_end_pos):
            return True

        # Get position of the adversary... deleted

        # BFS
        state_queue = [(my_start_pos, 0)]
        visited = {tuple(my_start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == max_step: 
                break
            for dir, move in enumerate(moves):
                if chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, my_end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    def check_endgame(self, chess_board, my_pos, adv_pos):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        board_size = chess_board.shape[1]    # array3d.shape ---> (layer,row,column)
        
        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        # 
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        '''
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
        if player_win >= 0:
            logging.info(
                f"Game ends! Player {self.player_names[player_win]} wins having control over {win_blocks} blocks!"
            )
        else:
            logging.info("Game ends! It is a Tie!")
        '''
        return True, p0_score, p1_score
       
    # THE ACTAUL IMPLEMENTATION...
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
        best_move = (-1,-1)
        
        my_new_pos = (-1,-1)
        dir_wall = -1

        # Stores all previous moves that we have searched
        previous_moves = [((-1,-1), -1)]

        for step_size in range(1, max_step+1): # eg. when moving 4 steps
            for _ in range (0, 3):   # we make this number of different moves for our agent in this step_size
                                    # eg. we take 2 different 4-step moves of the agent as a possibility to search
                for each_step in range (1, step_size+1): # move to each square...

                    # decide a direction to move in
                    dir_move = np.random.randint(0, 4)
                    
                    # we place a wall only on the last step:
                    if each_step == step_size:
                        dir_wall = np.random.randint(0,4)

                    # Go to a new position
                    r,c = my_pos
                    rd,cd = moves[dir_move]
                    my_new_pos = (r+rd, c+cd)
                    
                    while not self.check_valid_step(chess_board, my_pos, my_new_pos, dir_wall, adv_pos, max_step):

                        # generate a valid next step:
                        dir_move = np.random.randint(0,4)
                        
                        # we place a wall only on the last step:
                        if each_step == step_size:
                            dir_wall = np.random.randint(0,4)

                            # If we have already checked this move, do new move... (back to while)
                            if (dir_move, dir_wall) in previous_moves:
                                dir_move = (-1,-1)
                                dir_wall = -1
                        r,c = my_pos
                        rd,cd = moves[dir_move]
                        my_new_pos = (r+rd, c+cd)
                    
                    # If we are at the last step
                    if each_step == step_size:
                        r, c = my_new_pos
                        self.set_barrier(chess_board, r, c, dir_wall) # and unset_barrier at the same position later
                        # Add this move to the move history
                        previous_moves = previous_moves + [(my_new_pos, dir_wall)]
               
                # a new move of step-size steps has been generated here---

                # Run the minimax algo to check where to place our agent is the best
                # this is the minimizing node
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
        is_end, s1, s2 = self.check_endgame(chess_board, my_pos, adv_pos)
        
        # if a draw:
        if is_end & s1==s2:
            return 0.5
        
        # if not draw:
        # if the max player wins 
        if is_end & (not is_maximizing): return 1
        # if the min player wins
        elif is_end & is_maximizing: return 0

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        best_score = 0

        my_new_pos = (-1,-1)
        dir_wall = -1

        # Stores all previous moves that we searched in each for loop
        previous_moves = [((-1,-1), -1)]

        if is_maximizing:

            for step_size in range(1, max_step+1): # eg. when moving 4 steps
                for _ in range (0, 3):   # we make this number of different moves for our agent in this step_size
                                        # eg. we take 2 different 4-step moves of the agent as a possibility to search
                    for each_step in range (1, step_size+1): # move to each square...

                        # decide a direction to move in
                        dir_move = np.random.randint(0, 4)
                        
                        # we place a wall only on the last step:
                        if each_step == step_size:
                            dir_wall = np.random.randint(0,4)

                        # Go to a new position
                        r,c = my_pos
                        rd,cd = moves[dir_move]
                        my_new_pos = (r+rd, c+cd)
                        while not self.check_valid_step(chess_board, my_pos, my_new_pos, dir_wall, adv_pos, max_step):
                            
                            # generate a valid next step:
                            dir_move = np.random.randint(0,4)
                            
                            # we place a wall only on the last step:
                            if each_step == step_size:
                                dir_wall = np.random.randint(0,4)

                                # If we have already checked this move, do new move... (back to while)
                                if (dir_move, dir_wall) in previous_moves:
                                    dir_move = (-1,-1)
                                    dir_wall = -1
                            r,c = my_pos
                            rd,cd = moves[dir_move]
                            my_new_pos = (r+rd, c+cd)

                        # If we are at the last step
                        if each_step == step_size:
                            r, c = my_new_pos
                            self.set_barrier(chess_board, r, c, dir_wall)  # and unset_barrier at the same position later
                            # Add this move to the move history
                            previous_moves = previous_moves + [(my_new_pos, dir_wall)]
                        
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
                for _ in range (0, 3):   # we make this number of different moves for our agent in this step_size
                                        # eg. we take 2 different 4-step moves of the agent as a possibility to search
                    for each_step in range (1, step_size+1): # move to each square...

                        # decide a direction to move in
                        dir_move = np.random.randint(0, 4)
                        
                        # we place a wall only on the last step:
                        if each_step == step_size:
                            dir_wall = np.random.randint(0,4)
                        
                        # Go to a new position
                        r,c = adv_pos
                        rd,cd = moves[dir_move]
                        adv_new_pos = (r+rd, c+cd)
                        # Now we are taking a step for the adversary...
                                                 # The arguments are  pos,    new_pos,                ...,        adv_pos,
                        while not self.check_valid_step(chess_board, adv_pos, adv_new_pos, dir_wall, my_pos, max_step):
                            
                            # generate a valid next step:
                            dir_move = np.random.randint(0,4)
                            
                            # we place a wall only on the last step:
                            if each_step == step_size:
                                dir_wall = np.random.randint(0,4)

                                # If we have already checked this move, do new move... (back to while)
                                if (dir_move, dir_wall) in previous_moves:
                                    dir_move = (-1,-1)
                                    dir_wall = -1
                            r,c = adv_pos
                            rd,cd = moves[dir_move]
                            adv_new_pos = (r+rd, c+cd)

                        # If we are at the last step
                        if each_step == step_size:
                            r, c = adv_new_pos
                            self.set_barrier(chess_board, r, c, dir_wall)  # and unset_barrier at the same position later
                            # Add this move to the move history
                            previous_moves = previous_moves + [(adv_new_pos, dir_wall)]
                        
                    # a new move of step-size steps has been generated here---

                    # Run the minimax algo to check where to place our agent is the best
                    score = self.minimax(chess_board, my_pos, adv_new_pos, max_step, True, depth+1) # the TRUE here
                    
                    # Get the optimal ? position and wall direction
                    r, c = adv_new_pos
                    self.unset_barrier(chess_board, r, c, dir_wall)

                    if score < best_score:  # the LESS THAN here
                        best_score = score
            return best_score

    # M x M board. max_step = floor((M+1)/2)
    # for each node in the search tree, it has children for every possible step number : 
    # some children after moving 1 step, moving 2 steps, ... 5 steps max.
    # to reduce the branching factor of the search tree but still cover each possible distance, 
    # we can calculate only 2 positions for each number <= max_step, and 4 positions for the wall.

    
        