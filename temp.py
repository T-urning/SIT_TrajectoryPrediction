import pstats

p = pstats.Stats('outputs/cProfile_datset_out.txt')
p.sort_stats('time', 'cumulative').print_stats(80)