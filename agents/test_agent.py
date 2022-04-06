
# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np

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

    def step(self, chess_board, my_pos, adv_pos, max_step):
        self.get_valid_moves(chess_board, my_pos, adv_pos, max_step)
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

    def get_valid_moves(self, chess_board, my_pos, adv_pos, max_step): # returns set (can change to list if necessary) of valid moves from current position
        print(my_pos)
        print(max_step)
        board_size = chess_board.shape[0]
        print('board size is %d' % (board_size))
        end_posits = []
        for n in range(1,max_step+1):
            for r_dist in range(0,n+1):

                c_dist = n - r_dist
                print('%d %d %d' % (n,r_dist,c_dist))
                if r_dist == 0:
                    cur_moves = [(my_pos[0],my_pos[1] + c_dist),(my_pos[0],my_pos[1] - c_dist)]
                elif c_dist == 0:
                    cur_moves = [(my_pos[0] + r_dist,my_pos[1]),(my_pos[0] - r_dist,my_pos[1])]
                else:
                    cur_moves = [(my_pos[0] + r_dist,my_pos[1] + c_dist),(
                            my_pos[0] + r_dist,my_pos[1] - c_dist),(
                            my_pos[0] - r_dist,my_pos[1] + c_dist),(
                            my_pos[0] - r_dist,my_pos[1] - c_dist)]
                print(cur_moves)
                end_posits.extend(cur_moves)
        
        print(end_posits)
        end_posits = set(filter(lambda end_pos: end_pos[0] < board_size and end_pos[1] < board_size and end_pos[0] >= 0 and end_pos[1] >= 0 and end_pos != adv_pos, end_posits)) # filter moves which leave boundary or end in adversary's location
        print('filtered')
        print(end_posits)
        moves = []
        for end_pos in end_posits:
            moves.append(tuple((end_pos[0],end_pos[1],0)))
            moves.append(tuple((end_pos[0],end_pos[1],1)))
            moves.append(tuple((end_pos[0],end_pos[1],2)))
            moves.append(tuple((end_pos[0],end_pos[1],3)))

        moves = set(filter(lambda move: self.check_valid_step(np.asarray(my_pos),[move[0],move[1]],adv_pos, move[2], chess_board, max_step),moves))
        print(moves)
        return moves
    
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
        print(type(adv_pos))
        print(type(start_pos))
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

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
            print(cur_pos)
            r, c = cur_pos
            if cur_step == max_step:
                break
            for dir, move in enumerate(moves):
                if chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached



    

