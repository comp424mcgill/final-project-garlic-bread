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
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1)) # moves as defined in world.py (useful for reusing world.py code)
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1} # opposite directions as defined in world.py
        self.autoplay = True

    # functions set_barrier(), check_valid_step(), check_endgame() below copied from world.py:
    def set_barrier(self, chess_board, r, c, dir): # adapted from world.py function of same name
        # Set the barrier to True
        chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

    def unset_barrier(self, chess_board, r, c, dir):
        # Set the barrier to False
        chess_board[r, c, dir] = False
       
         # Set the opposite barrier to False
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = False

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
        # Endpoint already has barrier or is boarder
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
            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue

                next_pos = (cur_pos[0] + move[0],cur_pos[1]+move[1])
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, my_end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    def check_endgame(self, chess_board, my_pos, adv_pos,board_size):
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
                    self.moves[1:3]
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
        board_size = chess_board.shape[1]

        best_score = -1

        candidate_steps = self.get_valid_steps(chess_board,my_pos,adv_pos,max_step,board_size)
            
        for step in candidate_steps:
            # a new move of step-size steps has been generated here---
            r,c,dir = (step[0],step[1],step[2])

            self.set_barrier(chess_board,r,c,dir)
            # Run the minimax algo to check where to place our agent is the best
            # this is the minimizing node
            score = self.minimax(chess_board, (r,c), adv_pos, max_step, board_size, False, 3)
            
            self.unset_barrier(chess_board, r, c, dir)
                
            if score == 1:
                best_move = ((r,c),dir)
                break
            elif score > best_score:
                best_move = ((r,c),dir)
                
        return best_move

    def minimax(self, chess_board, my_pos, adv_pos, max_step, board_size, is_maximizing, depth):
        """
        is_maximizing is a bool indicating whether we are at maximizing or minimizing step
        depth is the depth of search

        Returns
        score: int
            1 -> max player wins
            0.75 -> depth limit reached
            0.5 -> draw
            0 -> min player wins

        """
        # Check base cases:
        is_end, s1, s2 = self.check_endgame(chess_board, my_pos, adv_pos,board_size)
        
        # if a draw or depth limit reached:
        
        if is_end:
            if s1 > s2: return 1
            elif s1 < s2: return 0
            else: return 0.5

        if depth == 0: # if depth limit reached
            return 0.75

        best_score = 0



        # Stores all previous moves that we searched in each for loop
        previous_moves = [(-1,-1,-1)]

        if is_maximizing:

            candidate_steps = self.get_valid_steps(chess_board,my_pos,adv_pos,max_step,board_size)

            for step in candidate_steps: # move to each square...
                r,c,dir = (step[0],step[1],step[2])
                self.set_barrier(chess_board, r, c, dir)  # and unset_barrier at the same position later
                # Add this move to the move history
                previous_moves.append(step)
            

                # Run the minimax algo to check where to place our agent is the best
                score = self.minimax(chess_board, (r,c), adv_pos, max_step, board_size, False, depth-1)
                # end_found, result, = self.minimax(...)
                
                self.unset_barrier(chess_board, r, c, dir)
                
                if score == 1:
                    best_score = 1
                    break
                elif score > best_score:
                    best_score = score

                
            return score
        
        else:  # if not is_maximizing: (it is the adversary's turn)

            candidate_steps = self.get_valid_steps(chess_board,adv_pos,my_pos,max_step,board_size)

            for step in candidate_steps: # move to each square...
                r,c,dir = (step[0],step[1],step[2])
                self.set_barrier(chess_board, r, c, dir)  # and unset_barrier at the same position later
                # Add this move to the move history
                previous_moves.append(step)
            

                # Run the minimax algo to check where to place our agent is the best
                score = self.minimax(chess_board, my_pos, (r,c), max_step, board_size, True, depth-1)
                # end_found, result, = self.minimax(...)
                
                self.unset_barrier(chess_board, r, c, dir)
                
                if score == 0:
                    best_score = 0
                    break
                elif score < best_score:
                    best_score = score
            return best_score

    # M x M board. max_step = floor((M+1)/2)
    # for each node in the search tree, it has children for every possible step number : 
    # some children after moving 1 step, moving 2 steps, ... 5 steps max.
    # to reduce the branching factor of the search tree but still cover each possible distance, 
    # we can calculate only 2 positions for each number <= max_step, and 4 positions for the wall.

    def get_valid_steps(self, chess_board, my_pos, adv_pos, max_step, board_size): # returns set (can change to list if necessary) of valid steps from current position
        end_posits = [my_pos]
        valid_steps = []
        for n in range(1,max_step+1):
            for r_dist in range(0,n+1):
                c_dist = n - r_dist
                if r_dist == 0:
                    cur_steps = [(my_pos[0],my_pos[1] + c_dist),(my_pos[0],my_pos[1] - c_dist)]
                elif c_dist == 0:
                    cur_steps = [(my_pos[0] + r_dist,my_pos[1]),(my_pos[0] - r_dist,my_pos[1])]
                else:
                    cur_steps = [(my_pos[0] + r_dist,my_pos[1] + c_dist),(
                            my_pos[0] + r_dist,my_pos[1] - c_dist),(
                            my_pos[0] - r_dist,my_pos[1] + c_dist),(
                            my_pos[0] - r_dist,my_pos[1] - c_dist)]
                end_posits.extend(cur_steps)
        # filter steps which leave boundary or end in adversary's location
        end_posits = set(filter(lambda end_pos: end_pos[0] < board_size and end_pos[1] < board_size and end_pos[0] >= 0 and end_pos[1] >= 0 and end_pos != adv_pos, end_posits)) 
        for end_pos in end_posits:
            for dir in range(0,4):
                if chess_board[end_pos[0],end_pos[1],dir]:
                    continue
                valid_steps.append(tuple((end_pos[0],end_pos[1],dir)))
        valid_steps = set(filter(lambda step: self.check_valid_step(chess_board, my_pos,[step[0],step[1]],step[2], adv_pos, max_step),valid_steps))
        return valid_steps
        