import simulator

args = simulator.get_args()
args.player_1 = "test_agent"
args.player_2 = "random_agent"
args.display = True
args.board_size = 5
s1 = simulator.Simulator(args)
result = s1.run()




