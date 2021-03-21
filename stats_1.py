import pstats 
from pstats import SortKey 

p = pstats.Stats('stats.txt') 
p.strip_dirs().sort_stats(SortKey.TIME).print_stats(100) 
