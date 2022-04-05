
# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys

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
        self.get_valid_moves(chess_board, my_pos, max_step)
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

    '''def get_valid_moves(self, chess_board, my_pos, max_step):
        print(my_pos)
        board_size = chess_board.shape[0]
        num_steps = max_step
        while (num_steps > 0)
            


        print(chess_board.shape)
    '''



    

