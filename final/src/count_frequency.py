import sys
from collections import defaultdict

d = defaultdict(int)

col = 1

with open('train', 'r') as f:
    for line in f:
        parts = line.rstrip().split(',')
        d[parts[col]] += 1
        
        
with open('test.in', 'r') as f:
    for line in f:
        parts = line.rstrip().split(',')
        d[parts[col]] += 1
        
        
print '\'', ('\', \'').join(d.keys()), '\''
