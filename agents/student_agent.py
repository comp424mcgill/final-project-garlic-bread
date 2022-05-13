# Student agent: Add your own agent here
from copy import deepcopy
from operator import is_
from agents.agent import Agent
from store import register_agent
import sys

import numpy as np
import time

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
        self.first_turn = True
        self.timer = 0
        self.time_limit = 28
        # self.search_count = 0 # count number of times search_valid_pos is run per turn
        # self.step_count = 0 # count number of steps considered per turn
        # self.step_get_time = 0 # track total number of time spent getting valid steps
        # self.check_end_time = 0 # track time spent in checking endgames

    # THE ACTAUL IMPLEMENTATION...
    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)   3-dimentional 
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer       # M x M board. max_step = floor((M+1)/2)

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        self.search_count = 0
        self.step_count = 0
        self.step_get_time = 0
        self.check_end_time = 0

        board_size = chess_board.shape[1]
        self.timer = time.time()    # record the current time at the start of each step

        best_score = -1
        best_depth = 0
        # Heuristics to choose the move with a higher percentage of winning rate
        weighted_score = 0

        valid_steps = self.get_valid_steps(chess_board,my_pos,adv_pos,max_step)
        # stupid_steps_0 = set(filter(lambda step: self.check_stupid_step_0(chess_board,step),valid_steps)) # depth 0 stupid moves
        # candidate_steps = valid_steps - stupid_steps_0
        # filter depth 0 and 1 stupid moves
        stupid_steps = set(filter(lambda step: self.check_stupid_step(chess_board,step,adv_pos,max_step),valid_steps))
        candidate_steps = valid_steps - stupid_steps

        if len(candidate_steps) == 0:
            if(len(stupid_steps > 0)):
                best_move = stupid_steps.pop()

        count = 0
        reduce_depth = True
    
        for n in range(1,3): # iterative deepening (two times)
            # depth limit logic
            if n == 1: 
                 # check for winning move even though it would be more efficient to store this, it is only ~40-100 checks which is quite small relative the total
                 # number of checks that we do (~14000 before hitting the time limit)
                 # would be more efficient to implement a breadth-first search, but I don't have time to do that
                depth_limit = 0
            else:
                depth_limit = 1 # go to depth 1
                if board_size < 9: # if smaller board, go deeper
                    depth_limit = depth_limit + 1
                if self.first_turn: # if first turn, go deeper
                    depth_limit = depth_limit + 1
                if len(candidate_steps) < 20: # if few options available, go deeper
                    depth_limit = depth_limit + 1
            for step in candidate_steps:
                count += 1
                self.step_count += 1
                # a new move of step-size steps has been generated here---
                r,c,dir = (step[0],step[1],step[2])
                self.set_barrier(chess_board,r,c,dir)
                
                # Run the minimax algo to check where to place our agent is the best
                # this is the minimizing node
                score, ret_depth, results = self.minimax(chess_board, (r,c), adv_pos, max_step, board_size, False,0, depth_limit)
                
                self.unset_barrier(chess_board, r, c, dir)

                weighted_score = (results[0]-results[1])/(results[2])

                # explanation: the series of elif statements are essentiall a decision hierarchy of deciding the best move where score -> depth -> weighted_score is the priority of move decision.
                # therefore, whenever we are assigning the best move in the hierarchy, we need to also assign the best_depth and the best_weighted_score at that location so that the information
                # of the current best_move is stored.
                if score == 1:
                    best_move = ((r,c),dir)
                    break
                elif score > best_score:
                    best_move = ((r,c),dir)
                    best_depth = ret_depth
                    best_weighted_score = weighted_score  # don't need it here # yes we do!
                # losing or drawing closer to 0 depth (bottom of search tree) is preferred
                elif score == best_score and ret_depth > best_depth:
                    best_move = ((r,c),dir)
                    best_depth = ret_depth
                    best_weighted_score = weighted_score # the weighted score of the best move, not the best_weighted score of any move
                # a move which has a higher weighted score is better, all other things equal
                elif score ==  best_score and ret_depth == best_depth and weighted_score > best_weighted_score:
                    best_move = ((r,c),dir)
                    best_depth = ret_depth
                    best_weighted_score = weighted_score
                
                if best_score == 0.75 and reduce_depth: # if depth limit has been reached for an indeterminate step, do shallower search to allow us to check every possibility for a quick win
                    reduce_depth = False
                    depth_limit = depth_limit - 1

                if (time.time() - self.timer) > self.time_limit:
                    # print('time limit exceeded')
                    break

            # if (time.time() - self.timer) > self.time_limit:
                # print(time.time() - self.timer)

        if self.first_turn == True:
            self.first_turn = False
            self.time_limit = 1.9

        return best_move
        

    def minimax(self, chess_board, my_pos, adv_pos, max_step, board_size, is_maximizing, depth,depth_limit):
        """
        is_maximizing is a bool indicating whether we are at maximizing or minimizing step
        depth is the depth of search

        Returns
        score: int
            1 -> max player wins
            0.75 -> depth limit reached
            0.5 -> draw
            0 -> min player wins
        depth: depth reached for result (if we detect)
        results: [int,int,int]
            array of results discovered (wins-draws-losses)
        """
        # self.step_count += 1
        # Check base cases:
        is_end, s1, s2 = self.check_endgame(chess_board, my_pos, adv_pos,board_size)
        
        # if a draw or depth limit reached:
        results = np.array([0,0,1]) # wins, losses, moves (don't care about draws for this)

        if is_end:
            if s1 > s2: 
                results[0]=1
                return 1,depth,results
            elif s1 < s2:
                results[1]=1
                return 0,depth,results
            else: 
                return 0.5,depth,results

        if depth == depth_limit: # if depth limit reached (height limit actually)
            return 0.75,depth,results

        best_score = 0
        best_depth = depth


        if is_maximizing:

            valid_steps = self.get_valid_steps(chess_board,my_pos,adv_pos,max_step)
            #stupid_steps_0 = set(filter(lambda step: self.check_stupid_step_0(chess_board,step),valid_steps))
            # filter depth 0 and 1 stupid moves
            stupid_steps = set(filter(lambda step: self.check_stupid_step(chess_board,step,adv_pos,max_step),valid_steps))
            candidate_steps = valid_steps - stupid_steps  #(stupid_steps_0 | stupid_steps_1)

            for step in candidate_steps: # move to each square...
                
                r,c,dir = (step[0],step[1],step[2])
                self.set_barrier(chess_board, r, c, dir)  # and unset_barrier at the same position later

                # Run the minimax algo to check where to place our agent is the best
                score,ret_depth,ret_results = self.minimax(chess_board, (r,c), adv_pos, max_step, board_size, False, depth+1,depth_limit)
                
                self.unset_barrier(chess_board, r, c, dir)

                results = np.add(results,ret_results) # add results which were found
                

                if score == 1:
                    best_score = 1
                    best_depth = ret_depth
                    break
                elif score > best_score:
                    best_score = score
                    best_depth = ret_depth  #ADDED, Best_depth must be assigned within the elif statements
                elif score == best_score and ret_depth > best_depth:
                    best_depth = ret_depth

                if (time.time() - self.timer) > self.time_limit:
                    break
                
            return best_score,best_depth,results
        
        else:  # if not is_maximizing: (it is the adversary's turn)

            # valid_steps = self.get_valid_steps(chess_board,my_pos,adv_pos,max_step) 
            valid_steps = self.get_valid_steps(chess_board,adv_pos,my_pos,max_step) #CHANGED  get valid steps of the opposing agent
            
            # filter depth 0 and 1 stupid moves
            stupid_steps = set(filter(lambda step: self.check_stupid_step(chess_board,step,my_pos,max_step),valid_steps))
            candidate_steps = valid_steps - stupid_steps  # (stupid_steps_0 | stupid_steps_1)
            
            for step in candidate_steps: # move to each square...
                
                r,c,dir = (step[0],step[1],step[2])
                self.set_barrier(chess_board, r, c, dir)  # and unset_barrier at the same position later
               
                # Run the minimax algo to check where to place our agent is the best
                score,ret_depth,ret_results = self.minimax(chess_board, my_pos, (r,c), max_step, board_size, True, depth+1,depth_limit)

                self.unset_barrier(chess_board, r, c, dir)

                results = np.add(results,ret_results)

                if score == 0:
                    best_score = 0
                    best_depth = ret_depth
                    break
                elif score < best_score:  #CHANGED , Best_depth must be assigned within the elif statements
                    best_score = score
                    best_depth = ret_depth
                elif score == best_score and ret_depth > best_depth:
                    best_depth = ret_depth

                if (time.time() - self.timer) > self.time_limit:
                    break

            return best_score,best_depth,results


    def get_valid_steps(self, chess_board, my_pos, adv_pos, max_step): # returns set (can change to list if necessary) of valid steps from current position
        # start = time.time()
        end_posits = [my_pos]
        valid_steps = set()
        end_posits = self.search_valid_pos(chess_board,my_pos,adv_pos,max_step)
        for end_pos in end_posits:
            empty_edges = self.get_empty_edges(chess_board,end_pos)
            for dir in empty_edges:
                valid_steps.add((end_pos[0],end_pos[1],dir))
        # self.step_get_time += (time.time()-start)
        return valid_steps        

    def search_valid_pos(self, chess_board, my_start_pos, adv_pos, max_step):
        """
        Modified version of check_valid_step which returns all reachable positions from a given location with a given max_step
        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        """
        # BFS
        # self.search_count = self.search_count + 1
        state_queue = [(my_start_pos, 0)]
        visited = {tuple(my_start_pos)}
        cur_step = 0
        while state_queue:
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

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        return visited

    def get_empty_edges(self,chess_board,pos):
        empty_edges = []
        for dir in range(0,4):
            if not (chess_board[pos[0],pos[1],dir]):
                empty_edges.append(dir)
        return empty_edges

    # def check_stupid_step_0(self,chess_board,step): 
        """
        This is not exhaustive of moves in which make us lose
        Checks for if move puts us in a 1x1 box, forcing us to lose
        """
        r,c = (step[0],step[1])
        empty_edges = self.get_empty_edges(chess_board,(r,c))
        if (len(empty_edges) == 1):
            return True
        return False

    def check_stupid_step(self, chess_board,step,adv_pos,max_step): 
        """
        This is not exhaustive of moves in which we can lose next turn
        Checks for if move puts us in a box with 3 edges around it where the opponent can reach the empty edge in their next turn
        """
        r,c,dir = step
        empty_edges = self.get_empty_edges(chess_board,(r,c))
        
        # A 1-stupid step allows the opponent to beat us next turn 
        if (len(empty_edges) == 2): # might be stupid
            empty_edges.remove(dir)
        
        # A 0-stupid_step makes us lose this turn
        # filter the steps that put agent in a 3-wall position and within the reach of the opponent
        if (len(empty_edges) == 1):
            empty_dir = empty_edges[0]
            move = self.moves[empty_dir] # find the position on the other side of wall
            win_pos = (r+move[0],c+move[1])
            if self.check_valid_step(chess_board,adv_pos,win_pos,self.opposites[empty_dir],(r,c),max_step):
                return True
        return False

    def check_valid_step(self, chess_board, my_start_pos, my_end_pos, dir, adv_pos, max_step):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        dir : int
            The direction of the barrier.
        """

        # Endpoint already has barrier or is boarder
        r, c = my_end_pos
        if chess_board[r, c, dir]:
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
        # start = time.time()
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
        
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        
        # self.check_end_time += (time.time() - start)
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
       

    
    