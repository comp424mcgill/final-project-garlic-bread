import simulator

args = simulator.get_args()
args.player_1 = "student_agent"
args.player_2 = "random_agent"
args.display = True
args.board_size = 5
args.autoplay = True
args.autoplay_runs = 100
args.board_size_min = 4
args.board_size_max = 6
s1 = simulator.Simulator(args)
#result = s1.autoplay()
result = s1.run()




