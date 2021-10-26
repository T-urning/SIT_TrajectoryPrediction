import pstats

p = pstats.Stats('outputs/transformer_profile.txt')
p.sort_stats('time', 'cumulative').print_stats(80)
print()