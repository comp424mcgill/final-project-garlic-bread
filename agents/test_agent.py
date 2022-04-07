
# Student agent: Add your own agent here
from operator import truediv
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy

@register_agent("test_agent")
class TestAgent(Agent):
 
    def __init__(self):
        super(TestAgent, self).__init__()
        self.name = "TestAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1)) # moves as defined in world.py (useful for reusing world.py code)
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1} # opposite directions as defined in world.py

    def step(self, chess_board, my_pos, adv_pos, max_step):
        valid_steps = self.get_valid_steps(chess_board, my_pos, adv_pos, max_step)
        terminal_steps = set()
        stupid_steps = set()
        for step in valid_steps:
            is_endgame, is_winner = self.sim_move(chess_board,step,adv_pos)[1:3]
            num_edges = self.count_edges(chess_board,(step[0],step[1]))
            if(is_endgame == True):
                terminal_steps.add(tuple((step,is_winner)))
            if(num_edges == 3):
                stupid_steps.add(step)
        print("Valid Steps:")
        print(valid_steps)
        print("Terminal Steps:")
        print(terminal_steps)
        print("Stupid Steps:")
        print(stupid_steps)
        not_stupid_steps = self.get_stupid_steps(chess_board,valid_steps,adv_pos,max_step)
        print(not_stupid_steps)

        text = input("Your move (x,y,dir) or input q to quit: ")
        
        while len(text.split(",")) != 3 and "q" not in text.lower():
            print("Wrong Input Format!")
            text = input("Your move (x,y,dir) or input q to quit: ")
        if "q" in text.lower():
            print("Game ended by user!")
            sys.exit(0)
        x, y, dir = text.split(",")
        x, y, dir = x.strip(), y.strip(), dir.strip()
        x, y = int(x), int(y)
        while not self.check_valid_input(
            x, y, dir, chess_board.shape[0], chess_board.shape[1]
        ):
            print(
                "Invalid Move! (x, y) should be within the board and dir should be one of u,r,d,l."
            )
            text = input("Your move (x,y,dir) or input q to quit: ")
            while len(text.split(",")) != 3 and "q" not in text.lower():
                print("Wrong Input Format!")
                text = input("Your move (x,y,dir) or input q to quit: ")
            if "q" in text.lower():
                print("Game ended by user!")
                sys.exit(0)
            x, y, dir = text.split(",")
            x, y, dir = x.strip(), y.strip(), dir.strip()
            x, y = int(x), int(y)
        my_pos = (x, y)
        return my_pos, self.dir_map[dir]

    def check_valid_input(self, x, y, dir, x_max, y_max):
        return 0 <= x < x_max and 0 <= y < y_max and dir in self.dir_map

    def get_valid_steps(self, chess_board, my_pos, adv_pos, max_step): # returns set (can change to list if necessary) of valid steps from current position
        board_size = chess_board.shape[0]
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
        valid_steps = set(filter(lambda move: self.check_valid_step(np.asarray(my_pos),[move[0],move[1]],adv_pos, move[2], chess_board, max_step),valid_steps))
        return valid_steps
    
    def check_valid_step(self, start_pos, end_pos, adv_pos, barrier_dir, chess_board, max_step): # reused from world.py (modified to work in this context)
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
        r, c = end_pos
        if chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
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
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        return is_reached

    def sim_move(self,chess_board,move,adv_pos):
        """
        Assumption is that move the move is valid
        Will add the move to a copy of the chess_board, and check if the move ends the game
        
        Returns
        -------
        chess_board_copy : chess_board
            Copy of chess_board with move applied
        is_endgame : bool
            Whether the game ends
        is_winner : bool
            (if is_endgame == True) Whether the player who made the move wins
            (if is_endgame == False or if game is a tie) None
        """

        # create copy of the chess_board to view the new gamestate
        chess_board_copy = deepcopy(chess_board) 
        
        # apply move to copied chess_board
        self.set_barrier(chess_board_copy,move[0],move[1],move[2])

        # check stupid end
        if self.count_edges(chess_board_copy,(move[0],move[1])) == 4:
            return chess_board_copy, True, False

        # check if move ends game
        cur_pos = (move[0],move[1])
        is_endgame, is_winner = self.check_endgame(chess_board_copy,cur_pos,adv_pos)
        return chess_board_copy, is_endgame, is_winner

    def set_barrier(self, chess_board, r, c, dir): # adapted from world.py function of same name
        # Set the barrier to True
        chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

    def check_endgame(self,chess_board,p0_pos,p1_pos): # adapted from world.py function of same name
        """
        Check if a game state is terminal and return who wins if so
        Does not return score since it is not useful

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        is_p0_winner : bool
            Whether p0 wins (if is_endgame == false or if there is a tie, returns None)
        """
        # Union-Find
        father = dict()
        for r in range(chess_board.shape[0]):
            for c in range(chess_board.shape[0]):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(chess_board.shape[0]):
            for c in range(chess_board.shape[0]):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(chess_board.shape[0]):
            for c in range(chess_board.shape[0]):
                find((r, c))
        p0_r = find(tuple(p0_pos))
        p1_r = find(tuple(p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, None
        is_p0_winner = None
        if p0_score > p1_score:
            is_p0_winner = True
        elif p0_score < p1_score:
            is_p0_winner = False
        return True, is_p0_winner

    def count_edges(self,chess_board,pos):
        count = 0
        for dir in range(0,4):
            if chess_board[pos[0],pos[1],dir]:
                count = count + 1
        return count

    def get_empty_edges(self,chess_board,pos):
        empty_edges = []
        for dir in range(0,4):
            if not (chess_board[pos[0],pos[1],dir]):
                empty_edges.append(dir)
        return empty_edges



    def get_stupid_steps(self,chess_board,steps,adv_pos,max_step):
        """
        Returns stupid steps
        In implementation it will make more sense for it to return steps which aren't stupid,
        but for now this makes it easier to see if it is correctly identifying "stupid" steps
        """
        filtered_steps = deepcopy(steps)
        chess_board_copy = deepcopy(chess_board)

        # stupid condition 1: instantly makes us lose
        # No need to query this, since it will already be queried when checking for terminal steps

        # stupid condition 2: put 3 walls around us when adversary is in range
        def check_stupid_2(chess_board,step,adv_pos,max_step):
            r,c,dir = step
            empty_edges = self.get_empty_edges(chess_board,(r,c))
            if (self.count_edges(chess_board_copy,(r,c)) == 2): # might be stupid
                empty_edges.remove(dir)
                empty_dir = empty_edges[0]
                move = self.moves[empty_dir]
                win_pos = (r+move[0],c+move[1])
                if self.check_valid_step(tuple(adv_pos),win_pos,(r,c),self.opposites[empty_dir],chess_board,max_step):
                    return True
            return False

        filtered_steps = set(filter(lambda step: check_stupid_2(chess_board,step,adv_pos,max_step),steps))
        return filtered_steps

    



    

